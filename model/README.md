# model/

**Bước 2 + 3** — Huấn luyện và đánh giá mô hình phân loại. Đặc tả đầy đủ: [SPEC.md](SPEC.md).

| File / thư mục | Nội dung |
|---|---|
| `SPEC.md` | Kiến trúc, hiện thực, kết quả trên dữ liệu Pima |
| `train.py` | Bước 2: GridSearchCV (5-fold, ROC-AUC) cho 5 mô hình, chọn ngưỡng sàng lọc (Sensitivity ≥ 0.85) |
| `evaluate.py` | Bước 3: ROC-AUC (CI bootstrap), Accuracy, Precision, Recall (= Sensitivity), F1, Specificity, NPV trên tập test |
| `inference.py` | API load checkpoint + dự đoán cho backend: `load_model()`, `RiskModel.predict()`; có CLI |
| `checkpoints/<model>.joblib` | Checkpoint = pipeline hoàn chỉnh (tiền xử lý + mô hình): `naive_bayes`, `decision_tree_id3`, `decision_tree_cart`, `svm_rbf`, `mlp` |
| `checkpoints/manifest.json` | Mô hình mặc định, cột đầu vào, phiên bản thư viện; với mỗi mô hình: file, ngưỡng t\*, tham số, điểm CV |
| `reports/cv_results.csv` | Tóm tắt CV trên tập train: ROC-AUC, Accuracy/Precision/Recall/F1/Specificity (OOF) ở ngưỡng 0.5 và t\* |
| `reports/test_metrics.csv` | Chỉ số trên tập test ở ngưỡng 0.5 và t\* |
| `reports/figures/` | `roc_curves.png`, `confusion_matrices.png`, `classification_metrics.png`, `sensitivity_specificity.png`, `threshold_tradeoff.png` |

## Cách chạy

```bash
.venv/bin/python model/train.py      # ~30 giây, ghi checkpoints/
.venv/bin/python model/evaluate.py
.venv/bin/python model/inference.py --list
.venv/bin/python model/inference.py '{"Glucose": 148, "BMI": 33.6, "Age": 50}'
```

**Hướng dẫn sử dụng checkpoint đã train:** [checkpoints/README.md](checkpoints/README.md) · Đặc tả API: [SPEC.md §7](SPEC.md#7-checkpoint--inference-giao-tiếp-với-bước-4).

## Mô hình

| Mô hình | Hiện thực |
|---|---|
| Naive Bayes | `GaussianNB` |
| Decision Tree ID3 | `DecisionTreeClassifier(criterion="entropy")` |
| Decision Tree CART | `DecisionTreeClassifier(criterion="gini")` |
| SVM | `SVC(kernel="rbf")` + Platt calibration để ra xác suất |
| Multi-layer Perceptron | `MLPClassifier` |

**Đầu vào:** `data/raw/train_raw.csv` / `test_raw.csv`. Tiền xử lý nằm bên trong pipeline nên mỗi fold CV fit lại imputer và scaler, tránh leakage. Kết quả refit trùng khớp với `data/processed/`.

**Mô hình đề xuất:** MLP @ t\* = 0.296. Trên test: AUC 0.805, Accuracy 0.714, Precision 0.562, Recall 0.833, F1 0.672, Specificity 0.650.
