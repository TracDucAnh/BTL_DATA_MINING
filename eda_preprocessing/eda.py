"""Exploratory data analysis of the Pima dataset; saves tables and figures."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eda_preprocessing.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    INVALID_ZERO_COLUMNS,
    RAW_DATA_PATH,
    RAW_DIR,
    TARGET_COLUMN,
    export_splits,
    load_pima_csv,
    prepare_predictors,
)

FIGURES_DIR = PROJECT_ROOT / "eda_preprocessing" / "figures"
TABLES_DIR = PROJECT_ROOT / "eda_preprocessing" / "tables"

IQR_MULTIPLIER = 1.5
CLASS_COLORS = {0: "#4477AA", 1: "#CC6600"}
CLASS_MARKERS = {0: "o", 1: "^"}
SHORT_LABELS = {"DiabetesPedigreeFunction": "DPF"}
PLOT_UNITS = {
    "Pregnancies": "count",
    "Glucose": "mg/dL",
    "BloodPressure": "mmHg",
    "SkinThickness": "mm",
    "Insulin": "µU/mL",
    "BMI": "kg/m²",
    "DiabetesPedigreeFunction": "score",
    "Age": "years",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
})


def short_label(feature: str) -> str:
    return SHORT_LABELS.get(feature, feature)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure {FIGURES_DIR.relative_to(PROJECT_ROOT) / name}")


def save_table(table: pd.DataFrame, name: str) -> None:
    table.to_csv(TABLES_DIR / name)
    print(f"  saved table  {TABLES_DIR.relative_to(PROJECT_ROOT) / name}")


def plot_outcome_distribution(data: pd.DataFrame) -> None:
    counts = data[TARGET_COLUMN].value_counts().sort_index()
    shares = counts / counts.sum()
    labels = [f"{count:,} ({share:.1%})" for count, share in zip(counts, shares)]

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.bar(counts.index.map(str), counts, width=0.5, color=CLASS_COLORS[0])
    ax.set(title="Outcome distribution", xlabel="Outcome", ylabel="Number of records")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    ax.bar_label(bars, labels=labels, padding=4)
    save_figure(fig, "outcome_bar.png")


def missingness_overview(raw: pd.DataFrame, eda: pd.DataFrame) -> pd.DataFrame:
    """Count missing values before and after treating invalid zeros as missing."""
    predictors = raw[FEATURE_COLUMNS]
    missing_on_load = predictors.isna().sum()
    total_missing = eda[FEATURE_COLUMNS].isna().sum()
    return pd.DataFrame({
        "Missing on load": missing_on_load,
        "Zeros in CSV": predictors.eq(0).sum(),
        "Zeros treated as missing": total_missing - missing_on_load,
        "Total missing": total_missing,
        "Missing (%)": (100 * total_missing / len(raw)).round(2),
    }).rename_axis("Predictor")


def distribution_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Describe each predictor and count values outside the boxplot fences."""
    columns = {"count": "Observed", "mean": "Mean", "50%": "Median", "std": "SD", "25%": "Q1", "75%": "Q3"}
    summary = features.describe().T[list(columns)].rename(columns=columns)
    summary["Observed"] = summary["Observed"].astype(int)
    summary["IQR"] = summary["Q3"] - summary["Q1"]

    lower = summary["Q1"] - IQR_MULTIPLIER * summary["IQR"]
    upper = summary["Q3"] + IQR_MULTIPLIER * summary["IQR"]
    summary["IQR flags"] = (features.lt(lower) | features.gt(upper)).sum()
    return summary.round(2).rename_axis("Predictor")


def plot_predictor_distributions(features: pd.DataFrame) -> None:
    """Boxplot stacked on a histogram for every predictor."""
    fig = plt.figure(figsize=(11, 13), layout="constrained")
    fig.suptitle("Predictor distributions in the training set", fontsize=15)
    grid = fig.add_gridspec((len(FEATURE_COLUMNS) + 1) // 2, 2)

    for position, feature in enumerate(FEATURE_COLUMNS):
        row, column = divmod(position, 2)
        panel = grid[row, column].subgridspec(2, 1, height_ratios=(1, 3), hspace=0.03)
        box_ax = fig.add_subplot(panel[0])
        hist_ax = fig.add_subplot(panel[1], sharex=box_ax)
        observed = features[feature].dropna()

        box_ax.boxplot(
            observed, orientation="horizontal", whis=IQR_MULTIPLIER, widths=0.55,
            patch_artist=True, boxprops={"facecolor": "#B7CCE1"},
            medianprops={"color": "#222222"}, flierprops={"markersize": 2.5},
        )
        if feature == "Pregnancies":
            bins = np.arange(observed.min() - 0.5, observed.max() + 1.5)
        else:
            bins = "auto"
        hist_ax.hist(observed, bins=bins, color=CLASS_COLORS[0], edgecolor="white", linewidth=0.6)

        box_ax.set_title(f"{feature} (n = {len(observed):,})", loc="left", fontsize=11)
        box_ax.set(yticks=[])
        box_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        box_ax.grid(False)
        for side in ("left", "bottom"):
            box_ax.spines[side].set_visible(False)

        hist_ax.set(ylabel="Rows")
        hist_ax.grid(axis="x", visible=False)

    save_figure(fig, "pred_dis_on_train.png")


def outcome_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Observed count, median and IQR of each predictor per Outcome class."""
    groups = train.groupby(TARGET_COLUMN)[FEATURE_COLUMNS]
    class_sizes = train[TARGET_COLUMN].value_counts().sort_index()
    observed = groups.count()
    median = groups.median()
    iqr = groups.quantile(0.75) - groups.quantile(0.25)
    return pd.concat(
        {
            f"Outcome = {label} (N = {class_sizes[label]:,})": pd.DataFrame({
                "Observed": observed.loc[label],
                "Median": median.loc[label],
                "IQR": iqr.loc[label],
            })
            for label in class_sizes.index
        },
        axis=1,
    ).round(2).rename_axis("Predictor")


def plot_predictors_by_outcome(train: pd.DataFrame) -> None:
    classes = sorted(train[TARGET_COLUMN].unique())
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.5), layout="constrained")
    fig.suptitle("Predictor distributions by Outcome in the training set", fontsize=14)
    fig.supxlabel("Outcome (n = observed measurements)", fontsize=11)

    for ax, feature in zip(axes.flat, FEATURE_COLUMNS, strict=True):
        groups = [train.loc[train[TARGET_COLUMN] == label, feature].dropna() for label in classes]
        boxes = ax.boxplot(
            groups, whis=IQR_MULTIPLIER, widths=0.5, patch_artist=True,
            tick_labels=[f"{label}\n(n = {len(group):,})" for label, group in zip(classes, groups)],
            medianprops={"color": "#222222"}, flierprops={"markersize": 2.5},
        )
        for patch, label in zip(boxes["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS[label])
            patch.set_alpha(0.6)
        ax.set(title=feature, ylabel=PLOT_UNITS[feature])
        ax.grid(axis="x", visible=False)

    save_figure(fig, "pred_dis_by_outcome.png")


def pairwise_observed_counts(features: pd.DataFrame) -> pd.DataFrame:
    """Number of rows where both predictors of each pair are observed."""
    observed = features.notna().astype("int64")
    return observed.T @ observed


def plot_pair_grid(train: pd.DataFrame) -> None:
    """Corner pair plot: scatter below the diagonal, class histograms on it."""
    classes = sorted(train[TARGET_COLUMN].unique())
    pair_counts = pairwise_observed_counts(train[FEATURE_COLUMNS])
    size = len(FEATURE_COLUMNS)
    fig, axes = plt.subplots(size, size, figsize=(14, 13.2))
    fig.suptitle("Predictor relationships in the training set", y=0.995, fontsize=15)

    for row, y_feature in enumerate(FEATURE_COLUMNS):
        for column, x_feature in enumerate(FEATURE_COLUMNS):
            ax = axes[row, column]
            if column > row:
                ax.set_visible(False)
                continue

            for label in classes:
                subset = train[train[TARGET_COLUMN] == label]
                if row == column:
                    values = subset[x_feature].dropna()
                    ax.hist(
                        values, bins=20, density=True, histtype="step",
                        color=CLASS_COLORS[label], linewidth=1.2,
                    )
                else:
                    ax.scatter(
                        subset[x_feature], subset[y_feature], s=10, alpha=0.45, linewidths=0,
                        color=CLASS_COLORS[label], marker=CLASS_MARKERS[label],
                    )

            ax.text(
                0.04, 0.94, f"n = {pair_counts.loc[y_feature, x_feature]:,}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
            )
            ax.tick_params(labelsize=7)
            ax.grid(False)
            ax.set_xlabel(short_label(x_feature) if row == size - 1 else "", fontsize=9)
            ax.set_ylabel(short_label(y_feature) if column == 0 else "", fontsize=9)
            if row != size - 1:
                ax.tick_params(labelbottom=False)
            if column != 0:
                ax.tick_params(labelleft=False)

    handles = [
        plt.Line2D([], [], linestyle="", marker=CLASS_MARKERS[label], color=CLASS_COLORS[label], label=str(label))
        for label in classes
    ]
    fig.legend(handles=handles, title=TARGET_COLUMN, loc="upper right", bbox_to_anchor=(0.92, 0.9), frameon=False)
    fig.subplots_adjust(top=0.955, wspace=0.08, hspace=0.08)
    save_figure(fig, "pred_pred_pair.png")


def plot_correlation_heatmap(train: pd.DataFrame) -> pd.DataFrame:
    """Lower-triangle Spearman correlation heatmap; returns the matrix."""
    features = train[FEATURE_COLUMNS]
    spearman = features.corr(method="spearman")
    pair_counts = pairwise_observed_counts(features)
    upper_triangle = np.triu(np.ones(spearman.shape, dtype=bool), k=1)
    labels = [short_label(feature) for feature in FEATURE_COLUMNS]

    fig, ax = plt.subplots(figsize=(9.0, 7.4), layout="constrained")
    image = ax.imshow(np.ma.masked_where(upper_triangle, spearman), cmap="RdBu_r", vmin=-1, vmax=1)
    for row in range(len(FEATURE_COLUMNS)):
        for column in range(row + 1):
            rho = spearman.iat[row, column]
            ax.text(
                column, row, f"{rho:.2f}\nn = {pair_counts.iat[row, column]}",
                ha="center", va="center", fontsize=8.5,
                color="white" if abs(rho) > 0.6 else "#222222",
            )
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.85, label="Spearman ρ")
    ax.set_title("Spearman correlation between predictors\nTraining set · n = rows with both values observed", pad=14)
    save_figure(fig, "pred_pred_heatmap.png")
    return spearman.round(3)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset overview on the full raw file.
    raw = load_pima_csv(RAW_DATA_PATH)
    print(f"[1] Loaded {len(raw):,} rows x {raw.shape[1]} columns from {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(raw[TARGET_COLUMN].value_counts().sort_index().to_string())
    plot_outcome_distribution(raw)

    # 2.1 Missingness: zeros in the five physiological measurements are invalid.
    eda = prepare_predictors(raw)
    eda[TARGET_COLUMN] = raw[TARGET_COLUMN]
    missingness = missingness_overview(raw, eda)
    print("\n[2.1] Missingness overview\n" + missingness.to_string())
    complete_rows = eda[INVALID_ZERO_COLUMNS].notna().all(axis=1).sum()
    both_missing = eda[["Insulin", "SkinThickness"]].isna().all(axis=1).sum()
    print(f"Rows with all five measurements recorded: {complete_rows:,} of {len(eda):,}")
    print(f"Rows missing both Insulin and SkinThickness: {both_missing:,}")
    save_table(missingness, "missingness_overview.csv")

    # 2.2 Outliers, on the stratified training split (created once, then reused).
    if not (RAW_DIR / "train_raw.csv").exists():
        export_splits()
    train = load_pima_csv(RAW_DIR / "train_raw.csv")
    print(f"\n[2.2] Training split: {len(train):,} rows")
    summary = distribution_summary(train[FEATURE_COLUMNS])
    print(summary.to_string())
    save_table(summary, "distribution_summary.csv")
    plot_predictor_distributions(train[FEATURE_COLUMNS])

    # 3. Predictor vs Outcome.
    by_outcome = outcome_summary(train)
    print("\n[3] Predictors by Outcome\n" + by_outcome.to_string())
    save_table(by_outcome, "outcome_summary.csv")
    plot_predictors_by_outcome(train)

    # 4. Predictor vs predictor.
    print("\n[4] Pairwise relationships")
    plot_pair_grid(train)
    spearman = plot_correlation_heatmap(train)
    save_table(spearman, "spearman_correlation.csv")


if __name__ == "__main__":
    main()
