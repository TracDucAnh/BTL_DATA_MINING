# app/

**Bước 4** — Ứng dụng web hỗ trợ nhân viên y tế sàng lọc nguy cơ.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `backend/` | API nhẹ (Flask/FastAPI) gọi `model.inference.load_model()` để nạp checkpoint từ `model/checkpoints/` và trả kết quả dự đoán |
| `frontend/` | Giao diện tĩnh HTML / CSS / JavaScript |
| `frontend/css/` | File style |
| `frontend/js/` | Logic gọi API, hiển thị kết quả, vẽ biểu đồ radar |
| `frontend/assets/` | Hình ảnh, icon |

## Chức năng

- Form nhập chỉ số xét nghiệm (glucose, huyết áp, insulin, BMI, tuổi, ...)
- Chọn loại bệnh cần sàng lọc: Tiểu đường / Tim mạch
- Hiển thị xác suất và mức độ nguy cơ
- **Biểu đồ radar** thể hiện các chỉ số vượt ngưỡng cảnh báo
- Xuất báo cáo nguy cơ

## Luồng hoạt động

```
frontend (form) ──POST chỉ số──► backend ──► imputer → scaler → model ──► xác suất ──► frontend (kết quả + radar)
```

> Trình duyệt không chạy trực tiếp được mô hình scikit-learn, vì vậy cần `backend/` làm cầu nối giữa giao diện và mô hình.

## Gọi mô hình từ backend

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # thư mục gốc repo

from model.inference import available_models, load_model

model = load_model()                 # nạp một lần khi khởi động server (có cache)
result = model.predict(payload)[0]   # payload: JSON object với các chỉ số từ form
# -> {"probability", "threshold", "high_risk", "imputed_fields", "model", "display_name"}
```

- Ở endpoint, bắt `ValueError` từ `load_model()` / `predict()` và trả HTTP 400 kèm nội dung lỗi (tên mô hình sai, field lạ, giá trị không phải số, số âm).
- Có thể dùng `available_models()` để frontend cho phép chọn mô hình.
- Hướng dẫn đầy đủ kèm ví dụ server Flask đã chạy thử: [model/checkpoints/README.md](../model/checkpoints/README.md).
