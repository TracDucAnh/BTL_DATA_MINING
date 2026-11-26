"""Step 3: evaluate the trained models on the held-out Pima test split."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eda_preprocessing.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    TARGET_COLUMN,
    load_pima_csv,
)
from model.train import DISPLAY_NAMES, MODELS_DIR, REPORTS_DIR, THRESHOLDS_PATH  # noqa: E402

FIGURES_DIR = REPORTS_DIR / "figures"
DEFAULT_THRESHOLD = 0.5
BOOTSTRAP_ROUNDS = 1000
MODEL_COLORS = {
    "naive_bayes": "#4477AA",
    "decision_tree_id3": "#66CCEE",
    "decision_tree_cart": "#228833",
    "svm_rbf": "#CCBB44",
    "mlp": "#EE6677",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
})


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure {(FIGURES_DIR / name).relative_to(PROJECT_ROOT)}")


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, rounds: int = BOOTSTRAP_ROUNDS) -> tuple[float, float]:
    """95% percentile bootstrap interval of the test ROC-AUC."""
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    for _ in range(rounds):
        sample = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[sample])) == 2:
            aucs.append(roc_auc_score(y_true[sample], scores[sample]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": sensitivity,
        "f1": 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0,
        "sensitivity": sensitivity,
        "specificity": tn / (tn + fp),
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def check_processed_consistency(pipeline, test: pd.DataFrame) -> None:
    """The saved preprocessor must reproduce data/processed/test.csv exactly."""
    processed = pd.read_csv(PROCESSED_DIR / "test.csv")[FEATURE_COLUMNS].to_numpy()
    transformed = pipeline.named_steps["preprocess"].transform(test[FEATURE_COLUMNS])
    max_diff = np.abs(processed - transformed).max()
    if max_diff > 1e-9:
        raise RuntimeError(f"Saved preprocessor differs from data/processed/test.csv (max diff {max_diff:.2e}).")
    print(f"Preprocessor matches data/processed/test.csv (max |diff| = {max_diff:.1e})")


def plot_roc_curves(y_true: np.ndarray, scores: dict, results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7), layout="constrained")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Chance")
    for name, model_scores in scores.items():
        fpr, tpr, _ = roc_curve(y_true, model_scores)
        row = results.loc[(name, "screening")]
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=1.8,
                label=f"{DISPLAY_NAMES[name]} (AUC = {row['roc_auc']:.3f})")
        ax.scatter(1 - row["specificity"], row["sensitivity"], color=MODEL_COLORS[name],
                   edgecolor="black", zorder=3, s=40)
    ax.set(
        title="ROC curves on the held-out test set\n(dots = screening thresholds chosen on training CV)",
        xlabel="False positive rate (1 − Specificity)", ylabel="True positive rate (Sensitivity)",
        xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
    )
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "roc_curves.png")


def plot_confusion_matrices(results: pd.DataFrame, model_names: list[str]) -> None:
    modes = [("default", "default threshold"), ("screening", "screening threshold")]
    fig, axes = plt.subplots(len(modes), len(model_names), figsize=(3.2 * len(model_names), 6.6), layout="constrained")
    fig.suptitle("Confusion matrices on the test set (rows = actual, columns = predicted)", fontsize=13)

    for row, (mode, label) in enumerate(modes):
        for column, name in enumerate(model_names):
            ax = axes[row, column]
            metrics = results.loc[(name, mode)]
            matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.sum())
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value}", ha="center", va="center", fontsize=13,
                        color="white" if value > matrix.sum() / 2 else "#222222")
            ax.set_xticks([0, 1], labels=["0", "1"])
            ax.set_yticks([0, 1], labels=["0", "1"])
            ax.grid(False)
            ax.set_title(f"{DISPLAY_NAMES[name]}\n{label} ({metrics['threshold']:.2f})", fontsize=9.5)
            ax.set_xlabel(f"Sens {metrics['sensitivity']:.2f} · Spec {metrics['specificity']:.2f}", fontsize=9)

    save_figure(fig, "confusion_matrices.png")


def plot_sensitivity_specificity(results: pd.DataFrame, model_names: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
    fig.suptitle("Sensitivity vs specificity on the test set", fontsize=13)
    positions = np.arange(len(model_names))
    width = 0.38

    for ax, mode, title in zip(axes, ["default", "screening"], [f"Threshold = {DEFAULT_THRESHOLD}", "Screening threshold"]):
        frame = results.xs(mode, level="mode").loc[model_names]
        for offset, metric, color in [(-width / 2, "sensitivity", "#CC6600"), (width / 2, "specificity", "#4477AA")]:
            bars = ax.bar(positions + offset, frame[metric], width, color=color, label=metric.capitalize())
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(positions, labels=[DISPLAY_NAMES[name] for name in model_names], rotation=20, ha="right")
        ax.set(title=title, ylim=(0, 1.08))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="upper right", frameon=False)
    save_figure(fig, "sensitivity_specificity.png")


def plot_classification_metrics(results: pd.DataFrame, model_names: list[str]) -> None:
    metrics = [
        ("accuracy", "Accuracy", "#4477AA"),
        ("precision", "Precision", "#66CCEE"),
        ("recall", "Recall (Sensitivity)", "#CC6600"),
        ("f1", "F1 score", "#228833"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True, layout="constrained")
    fig.suptitle("Accuracy, precision, recall and F1 on the test set", fontsize=13)
    positions = np.arange(len(model_names))
    width = 0.2

    for ax, mode, title in zip(axes, ["default", "screening"], [f"Threshold = {DEFAULT_THRESHOLD}", "Screening threshold"]):
        frame = results.xs(mode, level="mode").loc[model_names]
        for index, (metric, label, color) in enumerate(metrics):
            offset = (index - (len(metrics) - 1) / 2) * width
            bars = ax.bar(positions + offset, frame[metric], width, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        ax.set_xticks(positions, labels=[DISPLAY_NAMES[name] for name in model_names], rotation=20, ha="right")
        ax.set(title=title, ylim=(0, 1.08))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, frameon=False)
    save_figure(fig, "classification_metrics.png")


def plot_threshold_tradeoff(y_true: np.ndarray, scores: dict, thresholds: dict) -> None:
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, len(scores), figsize=(3.3 * len(scores), 3.6), sharey=True, layout="constrained")
    fig.suptitle("Sensitivity / specificity as the decision threshold moves (test set)", fontsize=13)

    for ax, (name, model_scores) in zip(axes, scores.items()):
        predictions = model_scores[None, :] >= grid[:, None]
        ax.plot(grid, predictions[:, y_true == 1].mean(axis=1), color="#CC6600", label="Sensitivity")
        ax.plot(grid, (~predictions[:, y_true == 0]).mean(axis=1), color="#4477AA", label="Specificity")
        ax.axvline(thresholds[name], color="#222222", linestyle="--", linewidth=1, label="Screening threshold")
        ax.axvline(DEFAULT_THRESHOLD, color="#999999", linestyle=":", linewidth=1, label=f"Default {DEFAULT_THRESHOLD}")
        ax.set(title=DISPLAY_NAMES[name], xlabel="Threshold", xlim=(0, 1))
    axes[0].set_ylabel("Score")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, frameon=False)
    save_figure(fig, "threshold_tradeoff.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    training = json.loads(THRESHOLDS_PATH.read_text())
    model_names = list(training["models"])
    thresholds = {name: training["models"][name]["threshold"] for name in model_names}

    test = load_pima_csv(RAW_DIR / "test_raw.csv")
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN].to_numpy()
    print(f"Test split: {len(test):,} rows | positives: {int(y_test.sum())} ({y_test.mean():.1%})")

    pipelines = {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in model_names}
    check_processed_consistency(pipelines[model_names[0]], test)
    scores = {name: pipeline.predict_proba(X_test)[:, 1] for name, pipeline in pipelines.items()}

    rows = []
    for name in model_names:
        auc = roc_auc_score(y_test, scores[name])
        auc_low, auc_high = bootstrap_auc_ci(y_test, scores[name])
        for mode, threshold in [("default", DEFAULT_THRESHOLD), ("screening", thresholds[name])]:
            rows.append({
                "model": name, "mode": mode,
                "roc_auc": auc, "roc_auc_ci_low": auc_low, "roc_auc_ci_high": auc_high,
                **threshold_metrics(y_test, scores[name], threshold),
            })
    results = pd.DataFrame(rows).set_index(["model", "mode"])
    results.round(4).to_csv(REPORTS_DIR / "test_metrics.csv")

    columns = ["threshold", "roc_auc", "accuracy", "precision", "recall", "f1", "specificity"]
    print("\n" + results[columns].round(3).to_string())

    best = max(model_names, key=lambda name: training["models"][name]["cv_roc_auc_mean"])
    best_row = results.loc[(best, "screening")]
    print(
        f"\nModel selected by training CV ROC-AUC: {DISPLAY_NAMES[best]} -> test AUC {best_row['roc_auc']:.3f} "
        f"[{best_row['roc_auc_ci_low']:.3f}, {best_row['roc_auc_ci_high']:.3f}], "
        f"accuracy {best_row['accuracy']:.3f}, precision {best_row['precision']:.3f}, "
        f"recall {best_row['recall']:.3f}, F1 {best_row['f1']:.3f}, specificity {best_row['specificity']:.3f}"
    )
    print(f"Saved metrics to {(REPORTS_DIR / 'test_metrics.csv').relative_to(PROJECT_ROOT)}")

    plot_roc_curves(y_test, scores, results)
    plot_confusion_matrices(results, model_names)
    plot_sensitivity_specificity(results, model_names)
    plot_classification_metrics(results, model_names)
    plot_threshold_tradeoff(y_test, scores, thresholds)


if __name__ == "__main__":
    main()
