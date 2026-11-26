#!/usr/bin/env bash
# Chạy lại toàn bộ 3 notebook (Phần A, B, C), ghi đè kết quả vào chính các notebook.
set -e
cd "$(dirname "$0")/notebooks"
for nb in 01_PhanA_CaiDat 02_PhanB_ChungMinh 03_PhanC_BaiTap; do
    echo ">> $nb"
    jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" --ExecutePreprocessor.timeout=1800
done
echo "Xong. Phần D (chạy thử): python ../src/partD_starter.py --demo"
