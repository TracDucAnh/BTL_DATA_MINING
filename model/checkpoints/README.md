# Hướng dẫn sử dụng checkpoint

Thư mục này chứa các mô hình **đã huấn luyện sẵn** cho bài toán sàng lọc nguy cơ tiểu đường (dữ liệu Pima). Có thể dùng ngay để dự đoán mà **không cần train lại**.

- [1. Có gì trong thư mục](#1-có-gì-trong-thư-mục)
- [2. Chuẩn bị môi trường](#2-chuẩn-bị-môi-trường)
- [3. Dùng nhanh bằng dòng lệnh](#3-dùng-nhanh-bằng-dòng-lệnh)
- [4. Dùng trong code Python](#4-dùng-trong-code-python)
- [5. Dữ liệu đầu vào](#5-dữ-liệu-đầu-vào)
- [6. Kết quả trả về & cách đọc](#6-kết-quả-trả-về--cách-đọc)
- [7. Chọn mô hình](#7-chọn-mô-hình)
- [8. Tích hợp vào backend](#8-tích-hợp-vào-backend)
- [9. Train lại / cập nhật checkpoint](#9-train-lại--cập-nhật-checkpoint)
- [10. Xử lý lỗi thường gặp](#10-xử-lý-lỗi-thường-gặp)

---

## 1. Có gì trong thư mục

| File | Nội dung |
|---|---|
| `naive_bayes.joblib` | Gaussian Naive Bayes |
| `decision_tree_id3.joblib` | Cây quyết định ID3 (entropy) |
| `decision_tree_cart.joblib` | Cây quyết định CART (gini) |
| `svm_rbf.joblib` | SVM kernel RBF (có calibration để ra xác suất) |
| `mlp.joblib` | Multi-layer Perceptron, **mô hình mặc định** |
| `manifest.json` | Thông tin đi kèm: mô hình mặc định, ngưỡng sàng lọc của từng mô hình, danh sách cột đầu vào, tham số, điểm CV, phiên bản thư viện lúc lưu |

Mỗi file `.joblib` là **một pipeline hoàn chỉnh**:

```
chỉ số thô  →  0 thành NaN  →  StandardScaler  →  KNNImputer(k=5)  →  mô hình  →  xác suất mắc bệnh
```

Vì vậy **không cần tự chuẩn hóa hay điền giá trị thiếu**: chỉ cần đưa vào chỉ số gốc như trên phiếu xét nghiệm.

> Không nên gọi `joblib.load()` trực tiếp. Hãy dùng `model/inference.py`: module này tự xử lý đường dẫn import, ngưỡng sàng lọc, validate đầu vào và kiểm tra phiên bản thư viện.

---

## 2. Chuẩn bị môi trường

Checkpoint được lưu bằng **scikit-learn 1.9.1, Python 3.12**. Nếu load bằng phiên bản khác, kết quả có thể sai lệch hoặc lỗi; xem phiên bản chính xác trong `manifest.json → environment`.

```bash
# từ thư mục gốc repo
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Checkpoint cần import được package `eda_preprocessing` (chứa hàm tiền xử lý), nên **thư mục gốc repo phải nằm trong `sys.path`**:
- Khi chạy `python model/inference.py`, việc này đã được xử lý tự động.
- Khi dùng từ code khác (ví dụ backend), hãy thêm thư mục gốc repo vào `sys.path` như ở mục 4.

---

## 3. Dùng nhanh bằng dòng lệnh

Chạy từ thư mục gốc repo.

**Liệt kê checkpoint** (dấu `*` đánh dấu mô hình mặc định):

```bash
.venv/bin/python model/inference.py --list
```

```
  naive_bayes          threshold 0.147  CV AUC 0.831  naive_bayes.joblib
  decision_tree_id3    threshold 0.360  CV AUC 0.814  decision_tree_id3.joblib
  decision_tree_cart   threshold 0.281  CV AUC 0.807  decision_tree_cart.joblib
  svm_rbf              threshold 0.254  CV AUC 0.850  svm_rbf.joblib
* mlp                  threshold 0.296  CV AUC 0.854  mlp.joblib
```

**Dự đoán cho một bệnh nhân** (truyền vào một JSON object):

```bash
.venv/bin/python model/inference.py '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}'
```

```json
[
  {
    "model": "mlp",
    "display_name": "MLP",
    "probability": 0.6282518744914964,
    "threshold": 0.2962902401652347,
    "high_risk": true,
    "imputed_fields": ["SkinThickness", "Insulin"]
  }
]
```

**Chọn mô hình khác, dự đoán nhiều bệnh nhân** (truyền vào một JSON list):

```bash
.venv/bin/python model/inference.py '[{"Glucose": 85, "BMI": 26.6, "Age": 31}, {"Glucose": 183, "BMI": 23.3, "Age": 32}]' --model svm_rbf
```

---

## 4. Dùng trong code Python

```python
import sys
sys.path.insert(0, "/đường/dẫn/tới/BTL_DATA_MINING")   # thư mục gốc repo

from model.inference import available_models, load_model

model = load_model()                  # mô hình mặc định (mlp)
# model = load_model("svm_rbf")       # hoặc chọn theo tên

# Một bệnh nhân
report = model.predict({
    "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
    "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
})[0]
print(f"{report['probability']:.1%}", "NGUY CƠ CAO" if report["high_risk"] else "nguy cơ thấp")

# Nhiều bệnh nhân
reports = model.predict([patient_1, patient_2, patient_3])

# So sánh tất cả mô hình cho cùng một bệnh nhân
for name in available_models():
    r = load_model(name).predict(patient_1)[0]
    print(name, round(r["probability"], 3), r["high_risk"])
```

Các hàm chính:

| Hàm | Mô tả |
|---|---|
| `load_model(name=None)` | Nạp checkpoint, mặc định là `default_model` trong manifest. **Có cache**: gọi lại nhiều lần không đọc đĩa lại. |
| `RiskModel.predict(records)` | `dict` hoặc `list[dict]` → `list[dict]` kết quả (mục 6) |
| `RiskModel.predict_proba(df)` | `pandas.DataFrame` 8 cột → `numpy` array xác suất; dùng khi xử lý cả bảng dữ liệu |
| `RiskModel.threshold` | Ngưỡng sàng lọc của mô hình |
| `available_models()` | Danh sách tên checkpoint |
| `load_manifest()` | Đọc toàn bộ `manifest.json` |

Ví dụ chấm điểm cả một file CSV:

```python
import pandas as pd
df = pd.read_csv("benh_nhan.csv")              # có 8 cột như mục 5
df["probability"] = model.predict_proba(df)
df["high_risk"] = df["probability"] >= model.threshold
```

---

## 5. Dữ liệu đầu vào

Mỗi bệnh nhân là một object với **tối đa 8 trường**. Tên trường phân biệt hoa thường và phải đúng như bảng dưới.

| Trường | Ý nghĩa | Đơn vị | Khoảng trong dữ liệu train |
|---|---|---|---|
| `Pregnancies` | Số lần mang thai | lần | 0 – 17 |
| `Glucose` | Glucose huyết tương 2 giờ sau nghiệm pháp dung nạp glucose (OGTT) | mg/dL | 56 – 199 |
| `BloodPressure` | Huyết áp tâm trương | mmHg | 24 – 122 |
| `SkinThickness` | Độ dày nếp gấp da cơ tam đầu | mm | 7 – 99 |
| `Insulin` | Insulin huyết thanh 2 giờ | µU/mL | 15 – 744 |
| `BMI` | Chỉ số khối cơ thể | kg/m² | 18.2 – 67.1 |
| `DiabetesPedigreeFunction` | Chỉ số tiền sử gia đình mắc tiểu đường | điểm | 0.08 – 2.33 |
| `Age` | Tuổi | năm | 21 – 81 |

**Quy tắc:**

- **Giá trị thiếu được chấp nhận.** Bỏ trường, hoặc truyền `null` hay `""`, thì mô hình sẽ ước lượng giá trị đó bằng KNNImputer.
- **Số `0` ở `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` được hiểu là "chưa đo"**, giống quy ước của dữ liệu gốc. `Pregnancies = 0` là giá trị thật.
- **Bị từ chối (`ValueError`):** trường không có trong bảng, giá trị không phải số, số âm, `inf`.
- **Giá trị nằm ngoài khoảng của dữ liệu train** (ví dụ Age = 15, Glucose = 400) vẫn được chấp nhận, nhưng mô hình chưa từng thấy trường hợp như vậy nên kết quả kém tin cậy.
- **Càng thiếu nhiều chỉ số, kết quả càng phụ thuộc vào giá trị ước lượng.** Nên có tối thiểu **Glucose, BMI, Age**, là các yếu tố phân biệt rõ nhất theo EDA.

---

## 6. Kết quả trả về & cách đọc

```json
{
  "model": "mlp",
  "display_name": "MLP",
  "probability": 0.628,
  "threshold": 0.296,
  "high_risk": true,
  "imputed_fields": ["SkinThickness", "Insulin"]
}
```

| Trường | Ý nghĩa |
|---|---|
| `probability` | Xác suất ước lượng mắc tiểu đường, trong khoảng 0–1 |
| `threshold` | Ngưỡng sàng lọc của mô hình, được chọn trên tập train để Sensitivity ≥ 0.85 |
| `high_risk` | `true` nếu `probability ≥ threshold`, khi đó nên làm xét nghiệm khẳng định (HbA1c, OGTT) |
| `imputed_fields` | Các chỉ số không được nhập và đã được mô hình ước lượng; nên hiển thị cho người dùng biết |

**Vì sao ngưỡng thấp hơn 0.5?**
- Mục tiêu là **sàng lọc**, tức là ưu tiên không bỏ sót người bệnh. Với ngưỡng 0.5, MLP bỏ sót tới 50% người bệnh trên tập test. Với ngưỡng 0.296, MLP phát hiện 45/54 ca, đổi lại khoảng 35% người khỏe bị cảnh báo nhầm.
- Vì vậy `high_risk = true` nghĩa là **"cần kiểm tra thêm"**, không phải chẩn đoán xác định.
- `probability` là ước lượng từ dữ liệu Pima (phụ nữ người Pima, ≥ 21 tuổi). Không nên coi đây là xác suất chính xác cho các quần thể khác.

---

## 7. Chọn mô hình

Hiệu năng trên tập test hold-out (154 bệnh nhân), tại ngưỡng sàng lọc của từng mô hình:

| Tên | ROC-AUC | Recall (Sensitivity) | Specificity | Precision | F1 | Gợi ý |
|---|---|---|---|---|---|---|
| `mlp` | 0.805 | 0.833 | 0.650 | 0.562 | 0.672 | **Mặc định**: CV AUC cao nhất trên train |
| `svm_rbf` | 0.815 | 0.870 | 0.640 | 0.566 | 0.686 | Phương án thay thế; bỏ sót ít nhất trên test |
| `naive_bayes` | 0.783 | 0.815 | 0.580 | 0.512 | 0.629 | Đơn giản, dễ giải thích |
| `decision_tree_id3` | 0.783 | 0.759 | 0.680 | 0.562 | 0.646 | Dễ giải thích bằng luật |
| `decision_tree_cart` | 0.745 | 0.815 | 0.510 | 0.473 | 0.599 | Kém nhất, chỉ dùng để so sánh |

Phân tích đầy đủ: [../SPEC.md §5](../SPEC.md#5-kết-quả-trên-dữ-liệu-pima).

---

## 8. Tích hợp vào backend

Ví dụ với **Flask** (cần `pip install flask`; file đặt tại `app/backend/server.py`):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # thư mục gốc repo

from flask import Flask, jsonify, request
from model.inference import available_models, load_model

app = Flask(__name__)
load_model()                                                    # nạp sẵn mô hình mặc định khi khởi động


@app.get("/api/models")
def models():
    return jsonify(available_models())


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Body must be a JSON object with the patient's measurements."}), 400
    try:
        model = load_model(request.args.get("model"))           # ?model=svm_rbf, bỏ trống = mặc định
        return jsonify(model.predict(payload)[0])
    except ValueError as error:                                 # dữ liệu sai / tên mô hình sai
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(port=5000)
```

Gọi thử:

```bash
curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" \
     -d '{"Glucose": 148, "BloodPressure": 72, "BMI": 33.6, "Age": 50}'
```

Lưu ý:
- `load_model()` có cache, nên gọi lại trong mỗi request không bị chậm.
- Một process có thể phục vụ nhiều request đồng thời, vì mô hình chỉ được đọc mà không bị thay đổi.
- Backend phải chạy bằng cùng môi trường Python (`.venv`) với lúc train.

---

## 9. Train lại / cập nhật checkpoint

```bash
.venv/bin/python model/train.py       # ghi đè *.joblib và manifest.json (~30 giây)
.venv/bin/python model/evaluate.py    # đánh giá lại trên test, đồng thời kiểm tra checkpoint load được
```

- Quá trình train **tất định** (seed 42): chạy lại với cùng dữ liệu và cùng `requirements.txt` sẽ cho checkpoint và `manifest.json` giống hệt.
- Cần train lại khi: thay đổi dữ liệu hoặc tiền xử lý, đổi phiên bản scikit-learn, hoặc đổi `TARGET_SENSITIVITY` trong `train.py`.
- Nếu backend đang chạy, cần khởi động lại để nạp checkpoint mới, vì mô hình đã được cache trong bộ nhớ.

---

## 10. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'eda_preprocessing'` hoặc `'model'` | Thư mục gốc repo chưa có trong `sys.path` | Thêm `sys.path.insert(0, "<thư mục gốc repo>")` trước khi import (mục 4) |
| `FileNotFoundError: model/checkpoints/manifest.json not found` | Chưa train hoặc thiếu file | Chạy `.venv/bin/python model/train.py` |
| `UserWarning: Checkpoints were saved with scikit-learn X but Y is installed` | Phiên bản thư viện khác lúc train | `pip install -r requirements.txt`, hoặc train lại bằng môi trường hiện tại |
| `ValueError: Unknown model 'xxx'` | Sai tên mô hình | Xem tên đúng bằng `--list` hoặc `available_models()` |
| `ValueError: Record 0: unknown fields ['Weight']` | Trường không thuộc 8 chỉ số | Sửa tên trường theo mục 5 (phân biệt hoa thường) |
| `ValueError: ... must be a number` / `non-negative` | Giá trị chữ, số âm hoặc `inf` | Kiểm tra dữ liệu nhập ở form |
