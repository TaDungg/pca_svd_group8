# Đề tài 8 — Phân tích thành phần chính, SVD và xấp xỉ hạng thấp

## Cấu trúc

```
notebooks/
  01_PhanA_CaiDat.ipynb      A.1–A.4: cài đặt (nguồn gốc của src/pca_core.py) + thí nghiệm
  02_PhanB_ChungMinh.ipynb   B.1–B.6: phát biểu, chứng minh, kiểm chứng số
  03_PhanC_BaiTap.ipynb      C.1–C.4: bài tập
src/
  pca_core.py                cài đặt Phần A dùng chung (notebook 2, 3 và Phần D import từ đây)
  partD_starter.py           khung mã Phần D
docs/
  HuongDan_DeTai8_ABC.pdf    hướng dẫn đọc hiểu 3 notebook (kèm chứng minh LaTeX)
  DeTai8_ChuanBi_PhanD.pdf   tài liệu chuẩn bị Phần D
  latex/                     nguồn LaTeX: chứng minh b_B1.tex … b_B6.tex, lời giải c_C1.tex, hình img/
```

## Chạy lại

```
pip install -r requirements.txt
bash run_all.sh                       # khoảng 2–3 phút trên CPU
python src/partD_starter.py --demo    # chạy thử quy trình Phần D trên tín hiệu mô phỏng
```

## Quy ước tái lập

- `SEED = 42`; mọi kết quả ngẫu nhiên: trung bình ± độ lệch chuẩn trên 5 lần chạy.
- Thuật toán cốt lõi tự cài bằng NumPy; `sklearn`/`scipy` chỉ để đối chiếu và lấy phân vị phân phối.
- Các ô `# [CORE]` trong notebook 1 được chép nguyên văn sang `src/pca_core.py`; ô cuối notebook 1 kiểm tra hai bản khớp nhau.
  Sửa một cài đặt thì sửa ở cả hai nơi rồi chạy lại ô kiểm tra.
