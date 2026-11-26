# SPEC — Bước 2 (Mô hình hóa) & Bước 3 (Đánh giá)

Tài liệu mô tả **kiến trúc**, **cách hiện thực** và **kết quả áp dụng trên dữ liệu Pima Indians Diabetes** của hai bước:

- **Bước 2** — Xây dựng Naive Bayes, Decision Tree (ID3/CART), SVM kernel RBF, Multi-layer Perceptron → [`train.py`](train.py)
- **Bước 3** — Đánh giá ROC-AUC, Sensitivity, Specificity, ưu tiên Sensitivity → [`evaluate.py`](evaluate.py)

---

## 1. Dữ liệu đầu vào

| Tập | File | Số dòng | Dương tính (`Outcome = 1`) |
|---|---|---|---|
| Train | `data/raw/train_raw.csv` | 614 | 214 (34.9%) |
| Test (hold-out) | `data/raw/test_raw.csv` | 154 | 54 (35.1%) |

- Chia **stratified 80/20**, `random_state = 42` (sinh bởi `eda_preprocessing/preprocessing.py`).
- 8 biến đầu vào: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`.
- Giá trị `0` ở `Glucose, BloodPressure, SkinThickness, Insulin, BMI` là phi sinh học và được coi là missing (xem kết quả EDA trong README chính).
- **Tập test chỉ được dùng một lần duy nhất, ở bước 3.** Mọi quyết định (siêu tham số, ngưỡng, chọn mô hình) đều đưa ra trên tập train.

---

## 2. Kiến trúc

### 2.1. Pipeline của một mô hình

Mỗi mô hình là một `sklearn.pipeline.Pipeline` hoàn chỉnh, nhận **chỉ số thô** (như nhân viên y tế nhập vào) và trả về **xác suất mắc bệnh**:

```
 chỉ số thô (8 cột, có thể có 0 / NaN)
        │
        ▼
┌───────────────────────────── preprocess (build_preprocessor) ─────────────────────────────┐
│ prepare_predictors   : chọn 8 cột, 0 → NaN ở 5 biến sinh học                              │
│ StandardScaler       : z-score; fit bỏ qua NaN                                            │
│ KNNImputer(k = 5)    : điền NaN bằng trung bình 5 láng giềng gần nhất (khoảng cách trên    │
│                        dữ liệu đã chuẩn hóa, nên các biến có đóng góp ngang nhau)          │
└────────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
     model  : GaussianNB | DecisionTree (entropy / gini) | Calibrated SVC-RBF | MLPClassifier
        │
        ▼
  P(Outcome = 1)  ──►  so với ngưỡng sàng lọc t*  ──►  nguy cơ cao / thấp
```

Phần `preprocess` chính là `build_preprocessor()` trong `eda_preprocessing/preprocessing.py` (bước 1), được dùng lại nguyên vẹn chứ không viết lại.

### 2.2. Vì sao huấn luyện trên pipeline thay vì đọc thẳng `data/processed/train.csv`

`data/processed/train.csv` là kết quả của imputer và scaler đã **fit trên toàn bộ 614 dòng train**. Nếu đem file này đi cross-validation thì ở mỗi fold, thông tin của fold validation đã "rò" vào giá trị trung bình, độ lệch chuẩn và láng giềng KNN, khiến điểm CV bị lạc quan hơn thực tế (data leakage).

Vì vậy `train.py` đưa **cả bộ tiền xử lý vào trong pipeline**, để mỗi fold CV fit lại imputer và scaler chỉ trên phần train của fold đó. Sau khi CV xong, `GridSearchCV` refit pipeline tốt nhất trên toàn bộ 614 dòng. Lúc này bước `preprocess` **trùng khớp hoàn toàn** với dữ liệu trong `data/processed/`. `evaluate.py` kiểm tra điều này mỗi lần chạy:

```
Preprocessor matches data/processed/test.csv (max |diff| = 0.0e+00)
```

Lợi ích thêm: file `.joblib` lưu lại là **một đối tượng duy nhất** (tiền xử lý + mô hình), nên app ở bước 4 chỉ cần nạp và gọi `predict_proba`.

### 2.3. Luồng xử lý

```
train_raw.csv ──► train.py ──┬─► checkpoints/<model>.joblib    (5 checkpoint: tiền xử lý + mô hình)
                             ├─► checkpoints/manifest.json     (ngưỡng t*, tham số, điểm CV, mô hình mặc định, môi trường)
                             └─► reports/cv_results.csv

checkpoints/ ──► inference.py (load_model) ──┬─► evaluate.py ─┬─► reports/test_metrics.csv
                                             │   + test_raw   └─► reports/figures/*.png
                                             └─► app/backend (bước 4)
```

---

## 3. Hiện thực bước 2 — `train.py`

### 3.1. Các mô hình và lưới siêu tham số

| Khóa | Mô hình | Hiện thực scikit-learn | Lưới tìm kiếm | Số cấu hình |
|---|---|---|---|---|
| `naive_bayes` | Gaussian Naive Bayes | `GaussianNB()` | `var_smoothing` ∈ 10⁻¹²…10⁻² | 11 |
| `decision_tree_id3` | Cây quyết định ID3 | `DecisionTreeClassifier(criterion="entropy")` | `max_depth` ∈ {3,4,5,6,8,None} × `min_samples_leaf` ∈ {1,5,10,20,40,60} × `class_weight` ∈ {None, balanced} | 72 |
| `decision_tree_cart` | Cây quyết định CART | `DecisionTreeClassifier(criterion="gini")` | giống ID3 | 72 |
| `svm_rbf` | SVM kernel RBF | `CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", ensemble=False)` | `C` ∈ {0.1…100} × `gamma` ∈ {scale, 0.001…0.1} × `class_weight` | 84 |
| `mlp` | Multi-layer Perceptron | `MLPClassifier(max_iter=2000)` | `hidden_layer_sizes` ∈ {(8),(16),(32),(16,8),(32,16)} × `alpha` ∈ {10⁻³…10} × `learning_rate_init` ∈ {10⁻³, 10⁻²} | 60 |

Ghi chú hiện thực:

- **ID3 và CART.** scikit-learn chỉ cài đặt cây nhị phân kiểu CART. "ID3" ở đây được hiện thực bằng **tiêu chí Information Gain (entropy)**, đúng tiêu chí chia của ID3; còn "CART" dùng **Gini impurity**. ID3 nguyên bản chia nhiều nhánh trên biến rời rạc, nhưng cả 8 biến Pima đều là biến số liên tục nên cách chia nhị phân theo ngưỡng là hợp lý.
- **SVM RBF.** SVM bản chất chỉ cho ra khoảng cách tới siêu phẳng, không cho xác suất. Để chấm điểm nguy cơ, mô hình được bọc trong `CalibratedClassifierCV` (Platt scaling, sigmoid). `SVC(probability=True)` đã bị deprecate từ scikit-learn 1.9 nên không dùng.
- **MLP.** Dùng optimizer Adam, hàm kích hoạt ReLU, đầu ra logistic. `alpha` là hệ số L2 regularization; với dữ liệu nhỏ (614 dòng), giá trị `alpha` lớn giúp tránh overfit.
- **Mất cân bằng lớp (65/35).** Tham số `class_weight="balanced"` được đưa vào lưới tìm kiếm để GridSearch tự quyết định có dùng hay không. Việc ưu tiên Sensitivity được xử lý riêng bằng ngưỡng quyết định (mục 3.3).
- Lưới được nới rộng ở những chỗ tham số tốt nhất rơi vào biên. Riêng `var_smoothing` của Naive Bayes nằm ở biên dưới, nhưng các giá trị trong lưới đều rất nhỏ so với phương sai ≈ 1 sau chuẩn hóa nên ảnh hưởng không đáng kể.

### 3.2. Tìm siêu tham số

- `GridSearchCV` với `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Chọn cấu hình theo **ROC-AUC** (`scoring="roc_auc"`). AUC không phụ thuộc ngưỡng, nên đo khả năng **xếp hạng** bệnh nhân theo nguy cơ.
- Với cấu hình tốt nhất, dùng dự đoán out-of-fold (mục 3.3) để tính **Accuracy, Precision, Recall, F1, Specificity** ở ngưỡng 0.5 và ở t\* (hàm `classification_metrics`), lưu vào `cv_results.csv` / `manifest.json`.

### 3.3. Chọn ngưỡng sàng lọc (ưu tiên Sensitivity)

Ngưỡng mặc định 0.5 cho Sensitivity thấp (≈ 0.48–0.76 trên test), nghĩa là bỏ sót tới một nửa người bệnh, không phù hợp cho sàng lọc. Vì vậy với mỗi mô hình tốt nhất:

1. Tính xác suất **out-of-fold** trên tập train bằng `cross_val_predict` (5 fold, cùng cách chia). Mỗi dòng được dự đoán bởi mô hình không nhìn thấy dòng đó.
2. Từ đường ROC out-of-fold, chọn **ngưỡng cao nhất t\* sao cho Sensitivity ≥ 0.85** (`TARGET_SENSITIVITY`). Ngưỡng cao nhất thỏa ràng buộc đồng nghĩa với Specificity cao nhất có thể.
3. Lưu t\* vào `checkpoints/manifest.json`, dùng ở bước 3 và ở app.

Ngưỡng được chọn **hoàn toàn trên tập train**, không nhìn tập test, nên kết quả trên test là ước lượng khách quan.

---

## 4. Hiện thực bước 3 — `evaluate.py`

Mỗi mô hình được đánh giá trên 154 dòng test ở **hai ngưỡng**: mặc định 0.5 và ngưỡng sàng lọc t\*.

| Chỉ số | Công thức | Ý nghĩa lâm sàng |
|---|---|---|
| ROC-AUC | diện tích dưới đường ROC | Xác suất một người bệnh ngẫu nhiên được chấm điểm cao hơn một người khỏe ngẫu nhiên |
| **Recall = Sensitivity** | TP / (TP + FN) | Tỷ lệ người bệnh được phát hiện; **ưu tiên số 1** |
| Specificity | TN / (TN + FP) | Tỷ lệ người khỏe được loại trừ đúng |
| Precision (PPV) | TP / (TP + FP) | Trong số ca báo dương, bao nhiêu ca thật sự bệnh |
| F1 score | 2 · Precision · Recall / (Precision + Recall) | Trung bình điều hòa, cân bằng giữa bỏ sót và báo nhầm |
| Accuracy | (TP + TN) / N | Tỷ lệ dự đoán đúng; dễ gây hiểu nhầm khi dữ liệu mất cân bằng |
| NPV | TN / (TN + FN) | Trong số ca báo âm, bao nhiêu ca thật sự khỏe |

- **Khoảng tin cậy 95% của AUC** được tính bằng bootstrap 1000 lần (seed 42). Tập test chỉ có 154 dòng nên phải xem độ bất định trước khi kết luận mô hình nào "tốt hơn".
- **Chọn mô hình cho app:** lấy mô hình có **CV ROC-AUC cao nhất trên train**, không chọn theo điểm test, để tránh chọn mô hình dựa trên tập test.

Đầu ra:

| File | Nội dung |
|---|---|
| `reports/test_metrics.csv` | Toàn bộ chỉ số (kèm TP/FN/FP/TN) cho 5 mô hình × 2 ngưỡng |
| `reports/figures/roc_curves.png` | Đường ROC của 5 mô hình, chấm tròn đánh dấu điểm làm việc tại t\* |
| `reports/figures/confusion_matrices.png` | Ma trận nhầm lẫn ở ngưỡng 0.5 và t\* |
| `reports/figures/sensitivity_specificity.png` | So sánh Sensitivity/Specificity ở hai ngưỡng |
| `reports/figures/classification_metrics.png` | So sánh Accuracy, Precision, Recall, F1 ở hai ngưỡng |
| `reports/figures/threshold_tradeoff.png` | Sensitivity và Specificity thay đổi theo ngưỡng |

---

## 5. Kết quả trên dữ liệu Pima

### 5.1. Cross-validation trên train (bước 2)

| Mô hình | Tham số tốt nhất | CV ROC-AUC | Ngưỡng t\* |
|---|---|---|---|
| Naive Bayes | `var_smoothing=1e-12` | 0.831 ± 0.009 | 0.147 |
| Decision Tree (ID3) | `max_depth=6, min_samples_leaf=20, balanced` | 0.814 ± 0.021 | 0.360 |
| Decision Tree (CART) | `max_depth=5, min_samples_leaf=40, balanced` | 0.807 ± 0.008 | 0.281 |
| SVM (RBF) | `C=30, gamma=0.003, balanced` | 0.850 ± 0.010 | 0.254 |
| **MLP** | `(16, 8), alpha=3.0, lr=0.001` | **0.854 ± 0.009** | 0.296 |

Chỉ số phân loại trên dự đoán out-of-fold của tập train (614 dòng):

| Mô hình | Ngưỡng | Accuracy | Precision | Recall (Sens.) | F1 | Specificity |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.5 | 0.757 | 0.663 | 0.617 | 0.639 | 0.833 |
| | t\* | 0.721 | 0.567 | 0.855 | 0.682 | 0.650 |
| Decision Tree (ID3) | 0.5 | 0.741 | 0.604 | 0.748 | 0.668 | 0.738 |
| | t\* | 0.708 | 0.552 | 0.864 | 0.674 | 0.625 |
| Decision Tree (CART) | 0.5 | 0.762 | 0.644 | 0.710 | 0.676 | 0.790 |
| | t\* | 0.630 | 0.483 | 0.860 | 0.618 | 0.508 |
| SVM (RBF) | 0.5 | **0.788** | **0.741** | 0.603 | 0.665 | **0.888** |
| | t\* | 0.710 | 0.555 | 0.850 | 0.672 | 0.635 |
| **MLP** | 0.5 | **0.788** | 0.728 | 0.626 | 0.673 | 0.875 |
| | t\* | 0.738 | 0.585 | 0.850 | **0.693** | 0.678 |

### 5.2. Tập test hold-out (bước 3)

| Mô hình | ROC-AUC [95% CI] | Ngưỡng | Accuracy | Precision | Recall (Sens.) | F1 | Specificity | NPV | Bỏ sót (FN) |
|---|---|---|---|---|---|---|---|---|---|
| Naive Bayes | 0.783 [0.708, 0.850] | 0.50 | 0.688 | 0.548 | 0.630 | 0.586 | 0.720 | 0.783 | 20 / 54 |
| | | **0.147** | 0.662 | 0.512 | **0.815** | 0.629 | 0.580 | 0.853 | 10 / 54 |
| Decision Tree (ID3) | 0.783 [0.706, 0.851] | 0.50 | 0.708 | 0.562 | 0.759 | 0.646 | 0.680 | 0.840 | 13 / 54 |
| | | **0.360** | 0.708 | 0.562 | **0.759** | 0.646 | 0.680 | 0.840 | 13 / 54 |
| Decision Tree (CART) | 0.745 [0.658, 0.822] | 0.50 | 0.701 | 0.574 | 0.574 | 0.574 | 0.770 | 0.770 | 23 / 54 |
| | | **0.281** | 0.617 | 0.473 | **0.815** | 0.599 | 0.510 | 0.836 | 10 / 54 |
| SVM (RBF) | **0.815** [0.745, 0.876] | 0.50 | 0.688 | 0.565 | 0.481 | 0.520 | 0.800 | 0.741 | 28 / 54 |
| | | **0.254** | **0.721** | 0.566 | **0.870** | **0.686** | 0.640 | 0.901 | **7 / 54** |
| **MLP** | 0.805 [0.733, 0.868] | 0.50 | 0.701 | 0.587 | 0.500 | 0.540 | 0.810 | 0.750 | 27 / 54 |
| | | **0.296** | 0.714 | 0.562 | **0.833** | 0.672 | **0.650** | 0.878 | 9 / 54 |

### 5.3. Nhận xét

1. **SVM-RBF và MLP là hai mô hình mạnh nhất**, cả trên CV (AUC ≈ 0.85) lẫn trên test (AUC ≈ 0.81). Naive Bayes và cây ID3 xếp sau, CART yếu nhất. Tuy vậy khoảng tin cậy AUC của các mô hình **chồng lấn nhiều** (n = 154), nên chênh lệch 0.01–0.03 giữa SVM, MLP, NB và ID3 **chưa có ý nghĩa thống kê**.
2. **Ngưỡng 0.5 không phù hợp cho sàng lọc.** Ở ngưỡng này SVM và MLP bỏ sót khoảng 50% người bệnh. Sau khi hạ ngưỡng xuống t\*, Sensitivity tăng lên 0.81–0.87 và **số ca bỏ sót giảm 2–4 lần** (SVM: 28 → 7, MLP: 27 → 9), đổi lại Specificity giảm từ khoảng 0.8 xuống 0.65. Đây là đánh đổi chấp nhận được trong sàng lọc: ca dương tính giả sẽ được làm xét nghiệm khẳng định (HbA1c, OGTT), còn ca âm tính giả thì bị bỏ qua hoàn toàn.
3. **Sensitivity trên test (0.81–0.87) bám sát mục tiêu 0.85 đặt trên train.** Nghĩa là cách chọn ngưỡng bằng dự đoán out-of-fold tổng quát hóa tốt. Mức sai lệch ±0.04 là hợp lý với 54 ca dương tính.
4. **SVM với `class_weight="balanced"` vẫn có Sensitivity thấp ở ngưỡng 0.5**, vì bước calibration ánh xạ điểm SVM về xác suất thực, tức là theo tỷ lệ bệnh khoảng 35%. Điều này cho thấy việc chọn ngưỡng là bắt buộc, không thể thay bằng trọng số lớp.
5. **Cây ID3 cho kết quả giống hệt nhau ở hai ngưỡng** vì xác suất của cây là rời rạc (tỷ lệ lớp trong từng lá): 0.36 và 0.5 nằm trong cùng một khoảng trống giữa hai mức xác suất của lá. Cây ID3 tự nhiên đã có Sensitivity 0.76 ở ngưỡng 0.5 nhờ `class_weight="balanced"`.
6. **Accuracy, Precision và F1.**
   - Trên test, hạ ngưỡng xuống t\* **không làm giảm F1** mà còn tăng mạnh với SVM (0.520 → 0.686) và MLP (0.540 → 0.672). Accuracy cũng tăng nhẹ (SVM 0.688 → 0.721, MLP 0.701 → 0.714). Nguyên nhân: Recall tăng gần gấp đôi trong khi Precision gần như đứng yên (≈ 0.56).
   - Riêng CART và Naive Bayes thì Accuracy và Precision giảm, vì đường ROC của hai mô hình này kém dốc hơn nên phải chấp nhận nhiều dương tính giả hơn để đạt Recall mục tiêu.
   - Trên train (OOF) ở ngưỡng 0.5, SVM và MLP có Accuracy cao nhất (0.788) và Precision cao (0.73–0.74), nhưng Recall chỉ ≈ 0.6. Đây là ví dụ rõ cho thấy **Accuracy cao chưa chắc là mô hình sàng lọc tốt**.
   - Mốc so sánh: một mô hình luôn dự đoán "không bệnh" đạt Accuracy 0.649 trên test (100/154) với Recall = 0. Vì vậy Accuracy chỉ dùng để tham khảo.
7. **EDA cho thấy Glucose tách hai lớp rõ nhất** (median 108 so với 141.5 mg/dL), sau đó là Insulin, BMI và Age. Một biến đơn lẻ đã phân biệt tốt như vậy giải thích vì sao ngay cả mô hình đơn giản như Naive Bayes cũng đạt AUC ≈ 0.78–0.83.

**Mô hình đề xuất cho app (bước 4): MLP** với ngưỡng t\* = 0.296, được chọn theo CV ROC-AUC cao nhất. Trên test, mô hình phát hiện 45/54 ca bệnh (Sensitivity 0.833) và loại trừ đúng 65% người khỏe; NPV 0.878. SVM-RBF là lựa chọn thay thế ngang hàng, thậm chí có Sensitivity trên test cao hơn (0.870).

---

## 6. Reproducibility

| Yếu tố | Cách đảm bảo |
|---|---|
| Chia dữ liệu | `train_test_split(stratify, random_state=42)`, lưu ra `train_raw.csv` / `test_raw.csv` và tái sử dụng |
| Cross-validation | `StratifiedKFold(5, shuffle=True, random_state=42)` dùng chung cho GridSearch và out-of-fold |
| Mô hình có yếu tố ngẫu nhiên | `random_state=42` cho DecisionTree, SVC, MLP |
| Bootstrap | `np.random.default_rng(42)` |
| Thư viện | Phiên bản cố định trong `requirements.txt` (Python 3.12) |
| Kiểm chứng | Chạy `run_pipeline.sh` hai lần cho `manifest.json`, `cv_results.csv`, `test_metrics.csv` giống nhau từng byte; bộ tiền xử lý đã lưu tái tạo chính xác `data/processed/test.csv` |

Thời gian chạy `train.py` khoảng 30 giây (song song hóa bằng `n_jobs=-1`; kết quả không phụ thuộc số nhân).

---

## 7. Checkpoint & inference (giao tiếp với bước 4)

### 7.1. Thư mục `model/checkpoints/`

| File | Nội dung |
|---|---|
| `<model>.joblib` | Checkpoint = `Pipeline` đã fit hoàn chỉnh (`prepare_predictors → StandardScaler → KNNImputer → classifier`). Nhận chỉ số thô, trả xác suất. |
| `manifest.json` | `default_model` (mô hình có CV ROC-AUC cao nhất, hiện là `mlp`), `feature_columns`, `zero_as_missing_columns`, `target_sensitivity`, `environment` (phiên bản Python/scikit-learn/numpy/pandas/joblib), và với từng mô hình: `file`, `threshold`, `best_params`, điểm CV/OOF |

Checkpoint được tạo lại mỗi lần chạy `train.py` và hoàn toàn tất định (mục 6). Tổng dung lượng khoảng 300 KB, nên được commit vào repo để backend dùng ngay mà không cần train lại.

### 7.2. API — `model/inference.py`

| Hàm / lớp | Chức năng |
|---|---|
| `load_model(name=None) -> RiskModel` | Nạp một checkpoint (mặc định là `default_model`). Kết quả được cache, nên backend gọi nhiều lần cũng chỉ đọc đĩa một lần. Cảnh báo nếu phiên bản scikit-learn khác với manifest. |
| `available_models()` | Danh sách tên checkpoint |
| `load_manifest()` | Đọc `manifest.json` (kiểm tra `manifest_version`) |
| `RiskModel.predict(records)` | Nhận `dict` hoặc `list[dict]` (JSON từ frontend), trả về danh sách `{model, display_name, probability, threshold, high_risk, imputed_fields}` |
| `RiskModel.predict_proba(df)` | Nhận DataFrame 8 cột thô, trả `np.ndarray` xác suất (dùng trong `evaluate.py`) |
| `to_frame(records)` | Validate đầu vào: từ chối field lạ, giá trị không phải số, số âm hoặc vô hạn; field thiếu / `null` / `""` → NaN để KNNImputer ước lượng |

`imputed_fields` liệt kê các chỉ số không được nhập (hoặc nhập 0 ở 5 biến sinh học) và đã được mô hình ước lượng. Frontend nên hiển thị danh sách này để nhân viên y tế biết kết quả dựa trên giá trị ước lượng.

```python
import sys
sys.path.insert(0, "<thư mục gốc repo>")   # checkpoint tham chiếu eda_preprocessing.preprocessing

from model.inference import load_model

model = load_model()                         # hoặc load_model("svm_rbf")
report = model.predict({
    "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
    "Insulin": 0,                            # 0 / null / bỏ trống = chưa xét nghiệm
    "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
})[0]
# {"model": "mlp", "display_name": "MLP", "probability": 0.635, "threshold": 0.296,
#  "high_risk": True, "imputed_fields": ["Insulin"]}
```

Dòng lệnh:

```bash
.venv/bin/python model/inference.py --list                       # * đánh dấu mô hình mặc định
.venv/bin/python model/inference.py '{"Glucose": 148, "BMI": 33.6, "Age": 50}' --model svm_rbf
```

`evaluate.py` cũng load mô hình qua `load_model()`, nên mỗi lần đánh giá đồng thời kiểm tra luôn đường load mà backend sẽ dùng.

Hướng dẫn sử dụng cho người dùng checkpoint (đơn vị đầu vào, cách đọc kết quả, ví dụ Flask, xử lý lỗi): [checkpoints/README.md](checkpoints/README.md).

---

## 8. Hạn chế và hướng phát triển

- **Dữ liệu nhỏ và đặc thù.** 768 phụ nữ người Pima, từ 21 tuổi trở lên. Kết quả chưa thể khái quát cho quần thể khác nếu chưa kiểm định ngoài (external validation).
- **Tập test nhỏ** (54 ca dương tính) làm khoảng tin cậy rộng. Có thể bổ sung **repeated nested CV** để có ước lượng ổn định hơn.
- **Missing không ngẫu nhiên.** Gần một nửa số dòng thiếu Insulin. Có thể so sánh thêm KNNImputer với MICE (`IterativeImputer`) như đề bài gợi ý, hoặc thêm biến chỉ báo "đã đo / chưa đo".
- **Dataset Heart Disease UCI** chưa được đưa vào. Pipeline hiện tại tổng quát (tiền xử lý → mô hình → ngưỡng sàng lọc), nên có thể áp dụng lại bằng cách thay hàm load dữ liệu và danh sách cột.
