# app/

**Bước 4** — Ứng dụng web hỗ trợ nhân viên y tế sàng lọc nguy cơ.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `backend/` | API nhẹ (Flask/FastAPI) nạp mô hình từ `model/saved_models/` và trả kết quả dự đoán |
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
