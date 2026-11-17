# data/

Chứa toàn bộ dữ liệu của project.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `raw/` | Dữ liệu gốc tải về, **giữ nguyên không chỉnh sửa** |
| `processed/` | Dữ liệu sau tiền xử lý (imputation, chuẩn hóa, chia train/test) — được sinh ra từ `eda_preprocessing/` |

## Nguồn dữ liệu

| Dataset | Nguồn | Số mẫu | Biến mục tiêu |
|---|---|---|---|
| Pima Indians Diabetes Database | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | `Outcome` (0/1) |
| Heart Disease UCI | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | `target` / `num` |

## Lưu ý

- Không sửa trực tiếp file trong `raw/`; mọi biến đổi phải đi qua các script `.py` trong `eda_preprocessing/` để có thể tái lập.
- Với Pima: các cột `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` có giá trị `0` là phi thực tế, cần xem như missing.
