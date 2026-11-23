# eda_preprocessing/

**Bước 1** — Khám phá dữ liệu (EDA) và tiền xử lý.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `*.py` | Các script Python EDA và tiền xử lý (đặt trực tiếp trong thư mục này) |
| `figures/` | Biểu đồ xuất ra (histogram, boxplot, heatmap tương quan, ...) dùng cho báo cáo |

Các script dự kiến:

| File | Nội dung |
|---|---|
| `eda.py` | Thống kê mô tả, vẽ và lưu biểu đồ vào `figures/` |
| `preprocessing.py` | Xử lý giá trị phi thực tế, imputation, chuẩn hóa, chia train/test, lưu ra `data/processed/` |

Cách chạy (từ thư mục gốc project):

```bash
python eda_preprocessing/eda.py
python eda_preprocessing/preprocessing.py
```

## Nội dung cần thực hiện

### EDA
- Thống kê mô tả, kiểu dữ liệu, phân phối từng thuộc tính
- Kiểm tra mất cân bằng lớp (class imbalance)
- Ma trận tương quan, phân phối theo nhãn
- Phát hiện outlier

### Tiền xử lý
- Phát hiện giá trị sinh học phi thực tế (huyết áp, glucose, insulin, BMI = 0) và chuyển thành missing
- Imputation bằng **KNNImputer** hoặc **MICE** (`IterativeImputer`), so sánh hai cách
- Chuẩn hóa bằng **StandardScaler**
- Chia tập train/test (stratified)

## Đầu vào / đầu ra

- **Đầu vào:** `data/raw/`
- **Đầu ra:** `data/processed/`, `eda_preprocessing/figures/`

Toàn bộ code viết dưới dạng file `.py`, không dùng Jupyter Notebook.

> Imputer và scaler phải được *fit chỉ trên tập train* để tránh rò rỉ dữ liệu, và cần lưu lại (vào `model/saved_models/`) để app dùng khi dự đoán.
