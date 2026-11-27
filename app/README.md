# Bước 4 — Ứng dụng sàng lọc nguy cơ tiểu đường Pima

Ứng dụng demo học thuật dùng các checkpoint đã lưu ở [Bước 2 và 3](../model/checkpoints/README.md). Backend gọi `model.inference.load_model()` để nhận chỉ số thô và trả điểm xác suất, ngưỡng của mô hình cùng các chỉ số đã được ước lượng.

## Chạy trên WSL

Mở terminal WSL tại thư mục gốc `BTL_DATA_MINING`. App dùng Python 3.12 và các checkpoint có sẵn trong repo.

### Chuẩn bị môi trường

Khối lệnh này tạo `.venv` nếu cần, rồi cài các thư viện theo phiên bản trong `requirements.txt`.

```bash
if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
```

### Khởi động app

```bash
.venv/bin/python -m app.backend.server
```

Giữ terminal này mở và truy cập [http://127.0.0.1:5000/](http://127.0.0.1:5000/) trên trình duyệt. Chọn **Điền mẫu** rồi **Chạy dự đoán** để kiểm tra. Có thể chọn **In / lưu PDF** khi kết quả hiện ra.

Nhấn `Ctrl+C` trong terminal chạy Flask để dừng app.

## Cách đọc kết quả

- Form dùng tám chỉ số theo đơn vị trong [hướng dẫn checkpoint](../model/checkpoints/README.md#5-dữ-liệu-đầu-vào). MLP là mô hình mặc định theo manifest. Có thể chọn một trong năm checkpoint để dự đoán.
- Pipeline trong checkpoint đổi các số 0 được quy ước là thiếu thành `NaN`, chạy `StandardScaler` → `KNNImputer(k=5)`, rồi dự đoán. Kết quả cho biết chỉ số nào được pipeline ước lượng.
- Radar hiển thị phân vị thực nghiệm của các chỉ số đã đo so với `train_raw.csv`. Bảng đánh dấu chỉ số nằm ngoài khoảng P5–P95 của tập train. Khoảng này mô tả dữ liệu huấn luyện.
- Kết quả phục vụ demo học thuật. Việc chẩn đoán tiểu đường cần đánh giá y tế và xét nghiệm phù hợp. [NIDDK](https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis)

## API

| Endpoint | Chức năng |
|---|---|
| `GET /api/models` | Trả danh sách mô hình và mô hình mặc định. |
| `POST /api/predict` | Nhận một bộ chỉ số đo và trả kết quả dự đoán cùng dữ liệu cho radar. |

```bash
curl -s http://127.0.0.1:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"measurements":{"Glucose":148,"BMI":33.6,"Age":50}}'
```

Thêm `"model": "svm_rbf"` ở cấp ngoài `measurements` để chọn checkpoint khác.

Frontend nằm ở `frontend/` và được Flask phục vụ cùng API trong `backend/server.py`.
