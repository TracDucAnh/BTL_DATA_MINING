# Bước 4 — Ứng dụng sàng lọc nguy cơ tiểu đường Pima

Ứng dụng demo học thuật dùng các checkpoint đã lưu ở [Bước 2 và 3](../model/checkpoints/README.md). Backend gọi `model.inference.load_model()` để nhận chỉ số thô và trả điểm xác suất, ngưỡng của mô hình cùng các chỉ số đã được ước lượng.

## Setup và chạy web app

Mở terminal ở thư mục gốc `BTL_DATA_MINING`, nơi có `requirements.txt`. Dùng [Python 3.12](https://www.python.org/downloads/) để khớp với môi trường của các checkpoint trong repo.

Nếu chạy cả trên Windows và WSL, dùng checkout và `.venv` riêng cho từng môi trường. Khi đã có `.venv` tạo bằng Python 3.12 trên cùng hệ điều hành, có thể bắt đầu từ lệnh cài thư viện.

### Windows (PowerShell)

Kiểm tra Python bằng `py -3.12 --version`. Các lệnh sau gọi trực tiếp Python trong `.venv`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.backend.server
```

### WSL, macOS và Linux

Kiểm tra Python bằng `python3.12 --version`. Trên WSL, chạy các lệnh trong terminal của bản phân phối Linux.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m app.backend.server
```

### Mở và thử ứng dụng

Giữ terminal chạy Flask và truy cập [http://127.0.0.1:5000/](http://127.0.0.1:5000/) trên trình duyệt. Chọn **Điền mẫu** rồi **Chạy dự đoán** để kiểm tra. Chọn **In / lưu PDF** nếu cần lưu kết quả.

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
