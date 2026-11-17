# model/

**Bước 2 + 3** — Huấn luyện và đánh giá mô hình phân loại.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `*.py` | Các script Python huấn luyện và đánh giá (đặt trực tiếp trong thư mục này) |
| `saved_models/` | Mô hình đã huấn luyện cùng imputer/scaler (`.pkl` / `.joblib`) |
| `reports/` | Bảng so sánh metric, ROC curve, confusion matrix |

Các script dự kiến:

| File | Nội dung |
|---|---|
| `train.py` | Huấn luyện 4 mô hình, tinh chỉnh siêu tham số, lưu vào `saved_models/` |
| `evaluate.py` | Tính ROC-AUC, Sensitivity, Specificity, vẽ ROC curve, lưu kết quả vào `reports/` |

Cách chạy (từ thư mục gốc project):

```bash
python model/train.py
python model/evaluate.py
```

## Mô hình

| Mô hình | Ghi chú |
|---|---|
| Naive Bayes | Gaussian NB |
| Decision Tree | ID3 (`criterion="entropy"`) / CART (`criterion="gini"`) |
| SVM | Kernel RBF, tinh chỉnh `C`, `gamma` |
| Multi-layer Perceptron | Tinh chỉnh số lớp ẩn, learning rate |

## Đánh giá

- **ROC-AUC**
- **Sensitivity** (Recall lớp dương) — **ưu tiên hàng đầu** để không bỏ sót người mắc bệnh
- **Specificity**
- Confusion matrix, có thể điều chỉnh ngưỡng quyết định để tăng Sensitivity

## Đầu vào / đầu ra

- **Đầu vào:** `data/processed/`
- **Đầu ra:** `model/saved_models/`, `model/reports/`
