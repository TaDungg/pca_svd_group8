"""pca_core.py — cài đặt của Phần A (Đề tài 8), xuất nguyên văn từ các ô [CORE] của notebooks/01_PhanA_CaiDat.ipynb
(riêng block_power_svd từ bài C.4). Notebook 2, 3 và Phần D import từ file này.

    from pca_core import pca_svd, PCAMonitor, LinearAutoencoder, StreamingPCA, randomized_svd
"""
import numpy as np
from scipy import stats
from scipy.linalg import subspace_angles


def mean_std(values, fmt=".3g"):
    """Chuỗi 'trung bình ± độ lệch chuẩn' (độ lệch chuẩn mẫu, ddof=1)."""
    v = np.asarray(values, dtype=float)
    return f"{v.mean():{fmt}} ± {v.std(ddof=1):{fmt}}"


def max_principal_angle_deg(A, B):
    """Góc chính lớn nhất (độ) giữa span(A) và span(B)."""
    return np.degrees(subspace_angles(A, B).max())


def pca_eig(X, k):
    # PCA bằng phân rã trị riêng của ma trận hiệp phương sai
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    C = Xc.T @ Xc / (n - 1)                  # ma trận hiệp phương sai d x d
    # eigh: ma trận đối xứng -> trị riêng thực, tăng dần
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]            # đổi sang giảm dần
    lam, V = lam[order], V[:, order]
    # trung bình, d x k hướng chính, mọi trị riêng
    return mu, V[:, :k], lam


def pca_svd(X, k):
    # PCA bằng SVD của ma trận dữ liệu đã quy tâm (cách sklearn dùng)
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # Xc = U diag(S) Vt
    lam = S**2 / (n - 1)                     # lambda_i = sigma_i^2 / (n-1)
    return mu, Vt[:k].T, lam


def fix_sign(W):
    # Vector riêng chỉ xác định đến dấu: chuẩn hóa để phần tử có |giá trị| lớn nhất của mỗi cột là dương
    idx = np.argmax(np.abs(W), axis=0)
    return W * np.sign(W[idx, np.arange(W.shape[1])])


def randomized_svd(A, k, p=10, q=2, rng=None):
    # SVD ngẫu nhiên (Halko-Martinsson-Tropp 2011, Thuật toán 4.4 + 5.1)
    rng = np.random.default_rng(rng)
    m, n = A.shape
    Omega = rng.standard_normal((n, k + p))      # (1) ma trận thử Gauss
    Q, _ = np.linalg.qr(A @ Omega)  # Y = A Omega, trực chuẩn hóa
    for _ in range(q):                           # (2) lặp lũy thừa có trực chuẩn hóa lại
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    B = Q.T @ A                                  # (3) ma trận nhỏ (k+p) x n
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], S[:k], Vt[:k]


def make_spectrum_matrix(m, n, s, rng):
    # Ma trận m x n có giá trị kỳ dị cho trước s (dài min(m, n))
    U, _ = np.linalg.qr(rng.standard_normal((m, len(s))))
    V, _ = np.linalg.qr(rng.standard_normal((n, len(s))))
    return (U * s) @ V.T


class StreamingPCA:
    # PCA tăng dần: chỉ lưu n, mu và phác thảo hạng k (S_k, V_k^T)
    def __init__(self, k):
        self.k, self.n, self.mean_ = k, 0, None

    def partial_fit(self, Xb):
        nb, mb = Xb.shape[0], Xb.mean(axis=0)
        if self.n == 0:
            M = Xb - mb
            new_mean = mb
        else:
            assert self.mean_ is not None
            # hiệu chỉnh do trung bình dịch chuyển
            corr = np.sqrt(self.n * nb / (self.n + nb)) * (self.mean_ - mb)
            M = np.vstack([self.S_[:, None] * self.Vt_, Xb - mb, corr])
            new_mean = (self.n * self.mean_ + nb * mb) / (self.n + nb)
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        self.S_, self.Vt_ = S[:self.k], Vt[:self.k]
        self.n += nb
        self.mean_ = new_mean
        self.explained_variance_ = self.S_**2 / (self.n - 1)
        return self

    @property
    def components(self):          # d x k
        return self.Vt_.T


def rbf_kernel(X, Y, gamma):
    # k(x, y) = exp(-gamma ||x - y||^2), tính ||x - y||^2 bằng khai triển để tránh vòng lặp
    sq = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def linear_kernel(X, Y, gamma=None):
    return X @ Y.T


class KernelPCA_:
    def __init__(self, k, kernel=rbf_kernel, gamma=1.0):
        self.k, self.kernel, self.gamma = k, kernel, gamma

    def fit(self, X):
        self.X_ = X
        K = self.kernel(X, X, self.gamma)
        # lưu để quy tâm điểm mới
        self.col_mean_ = K.mean(axis=0)
        self.all_mean_ = K.mean()
        Kc = K - self.col_mean_[None, :] - self.col_mean_[:,
                                                          # (1) quy tâm
                                                          None] + self.all_mean_
        # (2) phân rã
        ell, A = np.linalg.eigh(Kc)
        order = np.argsort(ell)[::-1][:self.k]
        self.ell_, A = ell[order], A[:, order]
        self.alpha_ = A / np.sqrt(self.ell_)            # chuẩn hóa ||v_j|| = 1
        self.train_scores_ = A * np.sqrt(self.ell_)     # = Kc @ alpha
        return self

    def transform(self, Y):                             # (3) chiếu điểm mới
        Ky = self.kernel(Y, self.X_, self.gamma)
        Kyc = Ky - Ky.mean(axis=1, keepdims=True) - \
            self.col_mean_[None, :] + self.all_mean_
        return Kyc @ self.alpha_


class LinearAutoencoder:
    # Tự mã hóa tuyến tính d -> k -> d, huấn luyện bằng hạ gradient toàn lô, gradient tự dẫn xuất
    def __init__(self, k, lr=0.05, n_iter=3000, seed=0):
        self.k, self.lr, self.n_iter, self.seed = k, lr, n_iter, seed

    def loss_and_grads(self, Xc, E, D):
        n = Xc.shape[0]
        Z = Xc @ E.T                       # n x k
        R = Z @ D.T - Xc                   # phần dư n x d
        loss = (R**2).sum() / n
        gD = 2.0 / n * R.T @ Z             # d x k
        gE = 2.0 / n * D.T @ R.T @ Xc      # k x d
        return loss, gE, gD

    def fit(self, X, track=None):
        rng = np.random.default_rng(self.seed)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        d = X.shape[1]
        E = 0.1 * rng.standard_normal((self.k, d))
        D = 0.1 * rng.standard_normal((d, self.k))
        self.history_ = []
        for it in range(self.n_iter):
            loss, gE, gD = self.loss_and_grads(Xc, E, D)
            E -= self.lr * gE
            D -= self.lr * gD
            if track is not None and (it % 20 == 0 or it == self.n_iter - 1):
                self.history_.append(
                    (it, loss, max_principal_angle_deg(D, track)))
        self.E_, self.D_ = E, D
        return self

    def reconstruct(self, X):
        return (X - self.mean_) @ self.E_.T @ self.D_.T + self.mean_


class PCAMonitor:
    # Giám sát quá trình bằng PCA: T^2 Hotelling + SPE (Q) với giới hạn lý thuyết và thực nghiệm
    def __init__(self, k=None, var_ratio=0.90, alpha=0.01):
        self.k, self.var_ratio, self.alpha = k, var_ratio, alpha

    def fit(self, X):
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1.0
        # chuẩn hóa bằng thống kê của dữ liệu bình thường
        Z = (X - self.mean_) / self.std_
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        lam = S**2 / (n - 1)
        if self.k is None:                                    # chọn k theo tỉ lệ phương sai tích lũy
            self.k = int(np.searchsorted(
                np.cumsum(lam) / lam.sum(), self.var_ratio) + 1)
        k, a = self.k, self.alpha
        self.P_, self.lam_, self.n_ = Vt[:k].T, lam, n
        # Giới hạn T^2
        self.T2_lim_chi2_ = stats.chi2.ppf(1 - a, k)
        self.T2_lim_F_ = k * (n - 1) * (n + 1) / \
            (n * (n - k)) * stats.f.ppf(1 - a, k, n - k)
        # Giới hạn SPE
        res = lam[k:]
        th1, th2, th3 = res.sum(), (res**2).sum(), (res**3).sum()
        h0 = 1 - 2 * th1 * th3 / (3 * th2**2)
        # c_alpha lấy cùng dấu với h0 (Jackson-Mudholkar 1979)
        ca = np.sign(h0) * stats.norm.ppf(1 - a)
        self.SPE_lim_JM_ = th1 * \
            (ca * np.sqrt(2 * th2 * h0**2) / th1 + 1 +
             th2 * h0 * (h0 - 1) / th1**2) ** (1 / h0)
        g, h = th2 / th1, th1**2 / th2
        self.SPE_lim_box_ = g * stats.chi2.ppf(1 - a, h)
        return self

    def scores(self, X):
        Z = (X - self.mean_) / self.std_
        # tọa độ trên các thành phần chính
        T = Z @ self.P_
        T2 = (T**2 / self.lam_[:self.k]).sum(axis=1)
        # phần dư ngoài không gian con chính
        E = Z - T @ self.P_.T
        SPE = (E**2).sum(axis=1)
        return T2, SPE

    def fit_empirical_limits(self, X_val):
        T2, SPE = self.scores(X_val)
        self.T2_lim_emp_ = np.quantile(T2, 1 - self.alpha)
        self.SPE_lim_emp_ = np.quantile(SPE, 1 - self.alpha)
        return self

    def limits(self, kind="theory"):
        if kind == "theory":
            return self.T2_lim_F_, self.SPE_lim_JM_
        return self.T2_lim_emp_, self.SPE_lim_emp_

    def predict(self, X, kind="theory", rule="either"):
        # Trả về mảng bool: True = báo động. rule: "T2", "SPE" hoặc "either"
        T2, SPE = self.scores(X)
        t_lim, q_lim = self.limits(kind)
        a_T, a_Q = T2 > t_lim, SPE > q_lim
        return {"T2": a_T, "SPE": a_Q, "either": a_T | a_Q}[rule]


def block_power_svd(A, k, max_iter=200, tol=1e-6, rng=None):
    # Lặp lũy thừa khối (subspace iteration) cho k thành phần kỳ dị đầu
    rng = np.random.default_rng(rng)
    Q, _ = np.linalg.qr(A @ rng.standard_normal((A.shape[1], k)))
    s_old = np.zeros(k)
    it = -1
    for it in range(max_iter):
        # trực chuẩn hóa ở cả hai phía để ổn định
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, Rq = np.linalg.qr(A @ Z)               # Rq = Q^T A Z (k x k)
        # ước lượng rẻ các sigma để kiểm tra hội tụ
        s = np.linalg.svd(Rq, compute_uv=False)
        if np.max(np.abs(s - s_old) / s) < tol:
            break
        s_old = s
    Ub, s, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    return Q @ Ub, s, Vt, it + 1
