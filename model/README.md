# model/

**Bước 2 + 3** — Huấn luyện và đánh giá mô hình phân loại. Đặc tả đầy đủ: [SPEC.md](SPEC.md).

| File / thư mục | Nội dung |
|---|---|
| `SPEC.md` | Kiến trúc, hiện thực, kết quả trên dữ liệu Pima |
| `train.py` | Bước 2: GridSearchCV (5-fold, ROC-AUC) cho 5 mô hình, chọn ngưỡng sàng lọc (Sensitivity ≥ 0.85) |
| `evaluate.py` | Bước 3: ROC-AUC (CI bootstrap), Accuracy, Precision, Recall (= Sensitivity), F1, Specificity, NPV trên tập test |
| `saved_models/<model>.joblib` | Pipeline hoàn chỉnh (tiền xử lý + mô hình): `naive_bayes`, `decision_tree_id3`, `decision_tree_cart`, `svm_rbf`, `mlp` |
| `saved_models/thresholds.json` | Ngưỡng t\*, tham số tốt nhất, điểm CV của từng mô hình |
| `reports/cv_results.csv` | Tóm tắt CV trên tập train: ROC-AUC, Accuracy/Precision/Recall/F1/Specificity (OOF) ở ngưỡng 0.5 và t\* |
| `reports/test_metrics.csv` | Chỉ số trên tập test ở ngưỡng 0.5 và t\* |
| `reports/figures/` | `roc_curves.png`, `confusion_matrices.png`, `classification_metrics.png`, `sensitivity_specificity.png`, `threshold_tradeoff.png` |

## Cách chạy

```bash
.venv/bin/python model/train.py      # ~30 giây
.venv/bin/python model/evaluate.py
```

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
