"""Step 2: tune and train the four classifier families on the Pima training split."""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eda_preprocessing.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_DIR,
    TARGET_COLUMN,
    build_preprocessor,
    export_splits,
    load_pima_csv,
)

MODELS_DIR = PROJECT_ROOT / "model" / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "model" / "reports"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

CV_FOLDS = 5
# Screening target: the decision threshold is lowered until cross-validated
# sensitivity reaches this value, keeping as much specificity as possible.
TARGET_SENSITIVITY = 0.85

DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "decision_tree_id3": "Decision Tree (ID3)",
    "decision_tree_cart": "Decision Tree (CART)",
    "svm_rbf": "SVM (RBF)",
    "mlp": "MLP",
}

_TREE_GRID = {
    "model__max_depth": [3, 4, 5, 6, 8, None],
    "model__min_samples_leaf": [1, 5, 10, 20, 40, 60],
    "model__class_weight": [None, "balanced"],
}

MODEL_SPECS = {
    "naive_bayes": (
        GaussianNB(),
        {"model__var_smoothing": np.logspace(-12, -2, 11)},
    ),
    "decision_tree_id3": (
        DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE),
        _TREE_GRID,
    ),
    "decision_tree_cart": (
        DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
        _TREE_GRID,
    ),
    "svm_rbf": (
        # Platt scaling turns the SVM margin into a probability for risk scoring.
        CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), method="sigmoid", ensemble=False),
        {
            "model__estimator__C": [0.1, 0.3, 1, 3, 10, 30, 100],
            "model__estimator__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1],
            "model__estimator__class_weight": [None, "balanced"],
        },
    ),
    "mlp": (
        MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
        {
            "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)],
            "model__alpha": [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0],
            "model__learning_rate_init": [1e-3, 1e-2],
        },
    ),
}


def build_pipeline(estimator) -> Pipeline:
    """Raw predictors -> invalid zeros to NaN -> StandardScaler -> KNNImputer -> model."""
    return Pipeline([("preprocess", build_preprocessor(n_neighbors=5)), ("model", clone(estimator))])


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def select_threshold(y_true: pd.Series, scores: np.ndarray, target: float = TARGET_SENSITIVITY) -> float:
    """Highest threshold whose sensitivity reaches the target (i.e. best specificity)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    reaching = np.flatnonzero(tpr >= target)
    return float(thresholds[reaching[0]])


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall (= sensitivity), F1 and specificity of hard predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0)),
    }


def load_training_split() -> tuple[pd.DataFrame, pd.Series]:
    if not (RAW_DIR / "train_raw.csv").exists():
        export_splits()
    train = load_pima_csv(RAW_DIR / "train_raw.csv")
    return train[FEATURE_COLUMNS], train[TARGET_COLUMN]


def train_model(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Grid-search one model on ROC-AUC, pick its screening threshold, save it."""
    estimator, grid = MODEL_SPECS[name]
    search = GridSearchCV(
        build_pipeline(estimator),
        grid,
        scoring="roc_auc",
        cv=cv_splitter(),
        n_jobs=-1,
    )
    search.fit(X, y)
    best = search.best_index_
    cv_results = search.cv_results_

    # Out-of-fold probabilities of the tuned model give an unbiased view of how
    # the classification metrics behave on unseen rows of the training split.
    oof_scores = cross_val_predict(
        clone(search.best_estimator_), X, y, cv=cv_splitter(), method="predict_proba", n_jobs=-1,
    )[:, 1]
    threshold = select_threshold(y, oof_scores)
    oof_metrics = {}
    for label, cutoff in [("0.5", 0.5), ("threshold", threshold)]:
        metrics = classification_metrics(y, (oof_scores >= cutoff).astype(int))
        oof_metrics.update({f"oof_{metric}_at_{label}": value for metric, value in metrics.items()})

    joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")

    # Short names ("C", not "model__estimator__C") and plain Python numbers for JSON.
    params = {key.split("__")[-1]: getattr(value, "item", lambda: value)() for key, value in search.best_params_.items()}
    return {
        "model": name,
        "display_name": DISPLAY_NAMES[name],
        "best_params": params,
        "candidates": len(cv_results["params"]),
        "cv_roc_auc_mean": float(cv_results["mean_test_score"][best]),
        "cv_roc_auc_std": float(cv_results["std_test_score"][best]),
        "oof_roc_auc": float(roc_auc_score(y, oof_scores)),
        "threshold": threshold,
        **oof_metrics,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    X, y = load_training_split()
    print(f"Training split: {len(X):,} rows | positives: {int(y.sum())} ({y.mean():.1%})")
    print(f"{CV_FOLDS}-fold stratified CV, refit on ROC-AUC, target sensitivity {TARGET_SENSITIVITY:.0%}\n")

    summaries = []
    for name in MODEL_SPECS:
        summary = train_model(name, X, y)
        summaries.append(summary)
        print(
            f"{summary['display_name']:<22} CV AUC {summary['cv_roc_auc_mean']:.3f} ± {summary['cv_roc_auc_std']:.3f}"
            f" | threshold {summary['threshold']:.3f}"
            f" -> acc {summary['oof_accuracy_at_threshold']:.3f}, prec {summary['oof_precision_at_threshold']:.3f},"
            f" recall {summary['oof_recall_at_threshold']:.3f}, f1 {summary['oof_f1_at_threshold']:.3f},"
            f" spec {summary['oof_specificity_at_threshold']:.3f}"
            f" | {summary['best_params']}"
        )

    THRESHOLDS_PATH.write_text(json.dumps(
        {
            "target_sensitivity": TARGET_SENSITIVITY,
            "models": {summary["model"]: summary for summary in summaries},
        },
        indent=2,
        default=str,
    ))
    table = pd.DataFrame(summaries).set_index("model")
    table["best_params"] = table["best_params"].map(lambda params: json.dumps(params, default=str))
    table.round(4).to_csv(REPORTS_DIR / "cv_results.csv")
    print(f"\nSaved models to {MODELS_DIR.relative_to(PROJECT_ROOT)}/ and CV summary to "
          f"{(REPORTS_DIR / 'cv_results.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
