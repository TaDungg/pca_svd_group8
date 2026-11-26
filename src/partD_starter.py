"""partD_starter.py — Khung mã nguồn Phần D, Đề tài 8: phát hiện bất thường thiết bị quay từ tín hiệu rung.

Gồm: đọc CSV phyphox -> nội suy về lưới thời gian đều -> chia cửa sổ chồng lấp -> đặc trưng
(RMS, độ nhọn, hệ số đỉnh, năng lượng dải tần từ phổ biên độ Fourier) -> mô hình chỉ học trên dữ liệu
bình thường (PCAMonitor của Phần A + các mô hình so sánh) -> đánh giá theo nhóm phiên ghi (GroupKFold).

Chạy thử toàn bộ quy trình trên tín hiệu MÔ PHỎNG (chưa có dữ liệu thật):
    python partD_starter.py --demo
Khi đã có dữ liệu thật (thư mục data/raw + data/metadata.csv):
    python partD_starter.py --metadata data/metadata.csv --raw-dir data/raw
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pca_core import PCAMonitor, LinearAutoencoder

SEED = 42


# ----------------------------------------------------------------------------- 1. Đọc dữ liệu
def load_phyphox_csv(path):
    """Đọc file CSV xuất từ phyphox (thí nghiệm Acceleration with/without g).
    Tên cột khác nhau tùy phiên bản/ngôn ngữ, nên tìm theo từ khóa: cột thời gian chứa 'time',
    ba cột gia tốc kết thúc bằng tên trục x/y/z (ví dụ 'Acceleration x (m/s^2)')."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    t_col = next(c for c in cols if "time" in c.lower())
    axis_cols = []
    for ax in ["x", "y", "z"]:
        cand = [c for c in cols if c != t_col and (f" {ax} " in f" {c.lower()} " or c.lower().strip().endswith(f" {ax}")
                                                   or f" {ax} (" in c.lower())]
        if not cand:
            raise ValueError(f"Không tìm thấy cột trục {ax} trong {cols}")
        axis_cols.append(cand[0])
    t = df[t_col].to_numpy(float)
    acc = df[axis_cols].to_numpy(float)
    # Một số máy Android trả về một loạt mẫu đầu có thời gian = 0 hoặc không tăng: bỏ các mẫu đó
    keep = np.concatenate([[True], np.diff(t) > 0]) & (
        t > 0 if (t > 0).any() else True)
    t, acc = t[keep], acc[keep]
    fs_est = 1.0 / np.median(np.diff(t))
    return t, acc, fs_est


def resample_uniform(t, acc, fs):
    """Nội suy tuyến tính về lưới thời gian đều tần số fs (dấu thời gian của điện thoại không đều)."""
    t_new = np.arange(t[0], t[-1], 1.0 / fs)
    return t_new, np.column_stack([np.interp(t_new, t, acc[:, j]) for j in range(acc.shape[1])])


def load_cwru_mat(path, channel="DE"):
    """Đọc file .mat của CWRU Bearing Data Center; lấy biến kết thúc bằng '_DE_time' (hoặc FE/BA)."""
    mat = sio.loadmat(path)
    keys = [k for k in mat if k.endswith(f"_{channel}_time")]
    if not keys:
        raise KeyError(
            f"{path}: không có biến *_{channel}_time; các biến: {[k for k in mat if not k.startswith('__')]}")
    num = os.path.splitext(os.path.basename(path))[0]
    key = next((k for k in keys if k.startswith(
        f"X{int(num):03d}")), keys[0]) if num.isdigit() else keys[0]
    return mat[key].ravel().astype(float)


# ----------------------------------------------------------------------------- 2. Cửa sổ và đặc trưng
def sliding_windows(x, win, hop):
    """x: (N,) hoặc (N, C) -> (số cửa sổ, win, C)."""
    x = x[:, None] if x.ndim == 1 else x
    starts = np.arange(0, len(x) - win + 1, hop)
    return np.stack([x[s:s + win] for s in starts])


def band_edges(fs, n_bands=12, f_min=2.0):
    """Biên các dải tần chia đều theo thang log từ f_min đến tần số Nyquist."""
    return np.geomspace(f_min, fs / 2, n_bands + 1)


def window_features(w, fs, edges):
    """w: (win, C). Trả về vector đặc trưng và tên đặc trưng."""
    feats, names = [], []
    hann = np.hanning(len(w))
    freqs = np.fft.rfftfreq(len(w), 1.0 / fs)
    for c in range(w.shape[1]):
        # bỏ thành phần một chiều (trọng lực, độ lệch)
        x = w[:, c] - w[:, c].mean()
        rms = np.sqrt(np.mean(x**2))
        # độ nhọn (phân phối chuẩn = 3)
        kurt = stats.kurtosis(x, fisher=False)
        crest = np.max(np.abs(x)) / (rms + 1e-12)          # hệ số đỉnh
        # phổ công suất (bình phương biên độ Fourier)
        spec = np.abs(np.fft.rfft(x * hann))**2
        band_e = [spec[(freqs >= lo) & (freqs < hi)].sum()
                  for lo, hi in zip(edges[:-1], edges[1:])]
        # log để giảm độ lệch phải
        feats += [rms, kurt, crest] + list(np.log10(np.array(band_e) + 1e-12))
        names += [f"{c}_rms", f"{c}_kurt", f"{c}_crest"] + \
            [f"{c}_band{b}" for b in range(len(band_e))]
    return np.array(feats), names


def features_from_signal(acc, fs, win_sec=2.0, overlap=0.5, n_bands=12):
    win = int(round(win_sec * fs))
    hop = int(round(win * (1 - overlap)))
    edges = band_edges(fs, n_bands)
    W = sliding_windows(acc, win, hop)
    rows = [window_features(w, fs, edges) for w in W]
    return np.array([r[0] for r in rows]), rows[0][1]


# ----------------------------------------------------------------------------- 3. Mô hình (chỉ học trên dữ liệu bình thường)
class AEDetector:
    """Điểm bất thường = sai số tái tạo của bộ tự mã hóa tuyến tính (Phần A.3)."""

    def __init__(self, k=5, n_iter=3000):
        self.k, self.n_iter = k, n_iter

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        lam1 = np.linalg.svd(Z - Z.mean(0), compute_uv=False)[0]**2 / len(Z)
        self.ae = LinearAutoencoder(
            self.k, lr=0.4 / lam1, n_iter=self.n_iter, seed=SEED).fit(Z)
        return self

    def score(self, X):
        Z = self.sc.transform(X)
        return ((Z - self.ae.reconstruct(Z))**2).sum(1)


class SklearnDetector:
    def __init__(self, model):
        self.model = model

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        self.model.fit(self.sc.transform(X))
        return self

    def score(self, X):
        # lớn hơn = bất thường hơn
        return -self.model.score_samples(self.sc.transform(X))


def make_models(k_pca=None):
    return {
        # xử lý riêng vì có 2 thống kê
        "PCA T2+SPE (Phần A)": None,
        "Tự mã hóa tuyến tính (Phần A)": AEDetector(k=5),
        "SVM một lớp": SklearnDetector(OneClassSVM(nu=0.05, gamma="scale")),
        "Rừng cô lập": SklearnDetector(IsolationForest(n_estimators=200, random_state=SEED)),
    }


# ----------------------------------------------------------------------------- 4. Đánh giá
def detection_delay(alarms, onset, fs_win, consecutive=3):
    """Độ trễ (giây) từ thời điểm lỗi bắt đầu đến khi có `consecutive` báo động liên tiếp."""
    run = 0
    for i in range(onset, len(alarms)):
        run = run + 1 if alarms[i] else 0
        if run == consecutive:
            return (i - consecutive + 1 - onset) / fs_win
    return np.nan


def evaluate(X, state, speed, session, alpha=0.01, n_folds=5):
    """GroupKFold theo phiên ghi: các cửa sổ chồng lấp của cùng một phiên không bao giờ nằm ở cả train và test.
    Trong mỗi phần gập: huấn luyện trên cửa sổ BÌNH THƯỜNG của các phiên train; ngưỡng = phân vị (1-alpha)
    của điểm bất thường trên phiên bình thường dành riêng làm kiểm định; kiểm tra trên các phiên test."""
    normal = state == "normal"
    faults = sorted(set(state) - {"normal"})
    results = {}
    gkf = GroupKFold(n_splits=n_folds)
    norm_sessions = np.unique(session[normal])
    for fold, (_, test_norm_idx) in enumerate(gkf.split(norm_sessions, groups=norm_sessions)):
        test_sessions = set(norm_sessions[test_norm_idx])
        rest = [s for s in norm_sessions if s not in test_sessions]
        rng = np.random.default_rng(SEED + fold)
        val_sessions = set(rng.choice(
            rest, size=max(1, len(rest) // 5), replace=False))
        tr = normal & ~np.isin(session, list(test_sessions | val_sessions))
        va = normal & np.isin(session, list(val_sessions))
        te_norm = normal & np.isin(session, list(test_sessions))
        fault_sessions = np.unique(session[~normal])
        # 1/n_folds số phiên lỗi mỗi phần gập
        te_fault = ~normal & np.isin(session, fault_sessions[fold::n_folds])

        for name, model in make_models().items():
            if name.startswith("PCA"):
                mon = PCAMonitor(var_ratio=0.90, alpha=alpha).fit(
                    X[tr]).fit_empirical_limits(X[va])

                def predict(Xq): return mon.predict(
                    Xq, kind="empirical", rule="either")

                def score(Xq):
                    T2, SPE = mon.scores(Xq)
                    t_lim, q_lim = mon.limits("empirical")
                    return np.maximum(T2 / t_lim, SPE / q_lim)
            else:
                model.fit(X[tr])
                thr = np.quantile(model.score(X[va]), 1 - alpha)
                def predict(Xq, m=model, t=thr): return m.score(Xq) > t
                score = model.score
            r = results.setdefault(
                name, {"FAR": [], "AUROC": [], **{f"TPR_{f}": [] for f in faults}})
            r["FAR"].append(predict(X[te_norm]).mean())
            for f in faults:
                idx = te_fault & (state == f)
                if idx.any():
                    r[f"TPR_{f}"].append(predict(X[idx]).mean())
            yt = np.r_[np.zeros(te_norm.sum()), np.ones(te_fault.sum())]
            r["AUROC"].append(roc_auc_score(
                yt, np.r_[score(X[te_norm]), score(X[te_fault])]))
    return results


def print_results(results):
    def ms(v):
        v = np.asarray(v, float)
        return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"
    for name, r in results.items():
        print(f"\n{name}")
        for key, v in r.items():
            print(f"   {key:<16}: {ms(v)}")


# ----------------------------------------------------------------------------- 5. Tín hiệu mô phỏng (CHỈ để chạy thử quy trình)
def simulate_fan_session(state, speed_level, fs=400.0, duration=120.0, rng=None):
    """Mô hình đồ chơi của rung quạt: tần số quay f_r, tần số cánh 5 f_r, nhiễu. KHÔNG thay thế dữ liệu thật."""
    rng = np.random.default_rng(rng)
    f_r = {1: 15.0, 2: 18.3, 3: 21.7}[speed_level] * rng.uniform(0.98, 1.02)
    t = np.arange(0, duration, 1 / fs)
    acc = np.zeros((len(t), 3))
    phase = rng.uniform(0, 2 * np.pi, 8)
    # khác biệt giữa các lần gắn điện thoại
    gain = rng.uniform(0.85, 1.15, 3)
    for c in range(3):
        a1, abpf = 0.05 * gain[c], 0.08 * gain[c]
        x = a1 * np.sin(2 * np.pi * f_r * t +
                        phase[c]) + abpf * np.sin(2 * np.pi * 5 * f_r * t + phase[3 + c])
        if state == "imbalance":                               # mất cân bằng: tăng mạnh thành phần 1x
            x += 0.15 * gain[c] * np.sin(2 * np.pi * f_r * t + phase[6])
        if state == "loose":                                   # lỏng chân đế: điều hòa bậc cao + nửa tần số quay
            x += 0.05 * np.sin(2 * np.pi * 2 * f_r * t) + 0.04 * \
                np.sin(2 * np.pi * 3 * f_r * t) + \
                0.03 * np.sin(np.pi * f_r * t)
        if state == "rub":                                     # cọ xát: xung ngắn mỗi vòng quay + nhiễu dải rộng
            pulses = (np.sin(2 * np.pi * f_r * t +
                      phase[7]) > 0.995).astype(float)
            x += 0.6 * np.convolve(pulses, np.exp(-np.arange(20) / 3),
                                   mode="same") * rng.standard_normal()
            x += 0.03 * rng.standard_normal(len(t))
        acc[:, c] = x + 0.02 * rng.standard_normal(len(t))
    acc[:, 2] += 9.81
    # dấu thời gian không đều như điện thoại thật
    t_jitter = t + rng.normal(0, 0.1 / fs, len(t))
    return np.sort(t_jitter), acc


def demo():
    print("CHẠY THỬ TRÊN TÍN HIỆU MÔ PHỎNG — kết quả chỉ để kiểm tra mã, KHÔNG dùng trong báo cáo.\n")
    fs = 400.0
    X_all, state_all, speed_all, sess_all = [], [], [], []
    sid = 0
    plan = [("normal", 8), ("imbalance", 5), ("loose", 5),
            ("rub", 5)]     # số phiên mỗi tốc độ
    for state, n_sess in plan:
        for speed in [1, 2, 3]:
            for _ in range(n_sess):
                t, acc = simulate_fan_session(
                    state, speed, fs=fs, duration=60, rng=SEED + sid)
                t_u, acc_u = resample_uniform(t, acc, fs)
                F, names = features_from_signal(acc_u, fs)
                X_all.append(F)
                state_all += [state] * len(F)
                speed_all += [speed] * len(F)
                sess_all += [sid] * len(F)
                sid += 1
    X = np.vstack(X_all)
    state, speed, session = np.array(state_all), np.array(
        speed_all), np.array(sess_all)
    print(f"{X.shape[0]} cửa sổ x {X.shape[1]} đặc trưng, {sid} phiên ghi")
    print_results(evaluate(X, state, speed, session))


def run_real(metadata_csv, raw_dir, fs=None):
    """metadata.csv cần các cột: file, session_id, state, speed (xem tài liệu hướng dẫn Phần D)."""
    meta = pd.read_csv(metadata_csv)
    signals = [load_phyphox_csv(os.path.join(raw_dir, f))
               for f in meta["file"]]
    fs_est = np.array([s[2] for s in signals])
    print("Tần số lấy mẫu ước lượng từng phiên (Hz):", np.round(fs_est, 1))
    # MỌI phiên dùng chung một tần số -> dải tần giống nhau
    fs_use = fs or float(np.floor(fs_est.min()))
    print(f"Nội suy mọi phiên về fs = {fs_use} Hz")
    X_all, state_all, speed_all, sess_all = [], [], [], []
    for (t, acc, _), (_, row) in zip(signals, meta.iterrows()):
        t_u, acc_u = resample_uniform(t, acc, fs_use)
        # bỏ 10 s đầu và cuối (chạm tay, khởi động)
        trim = int(10 * fs_use)
        F, names = features_from_signal(acc_u[trim:-trim], fs_use)
        X_all.append(F)
        state_all += [row["state"]] * len(F)
        speed_all += [row["speed"]] * len(F)
        sess_all += [row["session_id"]] * len(F)
    X = np.vstack(X_all)
    print_results(evaluate(X, np.array(state_all),
                  np.array(speed_all), np.array(sess_all)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--metadata")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--fs", type=float, default=None,
                    help="tần số nội suy chung cho mọi phiên (Hz)")
    args = ap.parse_args()
    if args.demo or not args.metadata:
        demo()
    else:
        run_real(args.metadata, args.raw_dir, args.fs)
