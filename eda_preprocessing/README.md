# eda_preprocessing/

**Bước 1** — Khám phá dữ liệu (EDA) và tiền xử lý. Toàn bộ code là script `.py`, không dùng Jupyter Notebook.

| File / thư mục | Nội dung |
|---|---|
| `eda.py` | Tổng quan, missingness, phân phối & outlier, so sánh theo `Outcome`, tương quan (pandas, numpy, matplotlib) |
| `preprocessing.py` | Chia train/test stratified, giá trị 0 phi sinh học → NaN, `StandardScaler` → `KNNImputer(k=5)` |
| `figures/` | Biểu đồ do `eda.py` sinh ra |
| `tables/` | Bảng số liệu do `eda.py` sinh ra (CSV) |

## Cách chạy

```bash
.venv/bin/python eda_preprocessing/eda.py
.venv/bin/python eda_preprocessing/preprocessing.py
```

Nếu `data/raw/train_raw.csv` chưa có, `eda.py` sẽ tự tạo split, nên có thể chạy EDA trước bước xử lý dữ liệu.

## Đầu ra

| File | Nội dung |
|---|---|
| `figures/outcome_bar.png` | Phân phối nhãn `Outcome` |
| `figures/pred_dis_on_train.png` | Boxplot + histogram từng biến (tập train) |
| `figures/pred_dis_by_outcome.png` | Boxplot từng biến theo `Outcome` |
| `figures/pred_pred_pair.png` | Pair plot giữa các biến, tô màu theo `Outcome` |
| `figures/pred_pred_heatmap.png` | Ma trận tương quan Spearman |
| `tables/missingness_overview.csv` | Số giá trị 0 / missing theo biến |
| `tables/distribution_summary.csv` | Thống kê mô tả + số điểm ngoài 1.5×IQR |
| `tables/outcome_summary.csv` | Median, IQR theo `Outcome` |
| `tables/spearman_correlation.csv` | Ma trận tương quan |
| `data/raw/{train,test}_raw.csv` | Split 80/20 còn NaN (từ `preprocessing.py`) |
| `data/processed/{train,test}.csv` | Đã imputation + chuẩn hóa (từ `preprocessing.py`) |

Imputer và scaler chỉ được fit trên tập train. `build_preprocessor()` được dùng lại bên trong pipeline của từng mô hình ở `model/train.py`, nên mô hình đã lưu tự thực hiện đúng các bước tiền xử lý này khi dự đoán.

Kết quả và nhận xét EDA: xem [README chính §2–3](../README.md#2-eda--khám-phá-dữ-liệu).
