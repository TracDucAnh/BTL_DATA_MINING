"""Prepare the Pima train/test split and build the KNN preprocessing pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DIR / "diabetes.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"
DATA_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_pima_csv(csv_path: Path) -> pd.DataFrame:
    """Read the Pima CSV with or without its column header."""
    first_record = pd.read_csv(csv_path, header=None, nrows=1).iloc[0]
    has_header = TARGET_COLUMN in first_record.tolist()
    data = pd.read_csv(csv_path, header=0 if has_header else None)

    if data.shape[1] != len(DATA_COLUMNS) or not isinstance(data.index, pd.RangeIndex):
        raise ValueError(f"{csv_path.name} must contain exactly {len(DATA_COLUMNS)} columns.")
    if has_header:
        if set(data.columns) != set(DATA_COLUMNS):
            raise ValueError(f"{csv_path.name} must use the Pima predictor names and Outcome.")
    else:
        data.columns = DATA_COLUMNS

    data = data[DATA_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    return data


def prepare_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """Select the eight predictors and mark invalid zeros as missing."""
    predictors = data[FEATURE_COLUMNS].copy()
    measurements = predictors[INVALID_ZERO_COLUMNS]
    predictors[INVALID_ZERO_COLUMNS] = measurements.mask(measurements.eq(0))
    return predictors


def build_preprocessor(n_neighbors: int = 5) -> Pipeline:
    """Build a fresh preprocessor for a DataFrame with the Pima predictors."""
    return Pipeline([
        ("prepare_predictors", FunctionTransformer(prepare_predictors)),
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer(n_neighbors=n_neighbors)),
    ])


def _write_csv_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    train_name: str,
    test_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as staging:
        staging_dir = Path(staging)
        train.to_csv(staging_dir / train_name, index=False)
        test.to_csv(staging_dir / test_name, index=False)
        (staging_dir / train_name).replace(output_dir / train_name)
        (staging_dir / test_name).replace(output_dir / test_name)


def export_splits(
    raw_path: Path = RAW_DATA_PATH,
    output_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export the stratified train/test split with missing values preserved."""
    data = load_pima_csv(raw_path)
    prepared = prepare_predictors(data)
    prepared[TARGET_COLUMN] = data[TARGET_COLUMN]
    train, test = train_test_split(
        prepared,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=prepared[TARGET_COLUMN],
    )

    _write_csv_pair(train, test, output_dir, "train_raw.csv", "test_raw.csv")
    return train, test


def export_processed(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on the raw train split and export both transformed splits."""
    train = load_pima_csv(raw_dir / "train_raw.csv")
    test = load_pima_csv(raw_dir / "test_raw.csv")
    preprocessor = build_preprocessor(n_neighbors=5)

    train_values = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    test_values = preprocessor.transform(test[FEATURE_COLUMNS])
    train_processed = pd.DataFrame(train_values, columns=FEATURE_COLUMNS, index=train.index)
    test_processed = pd.DataFrame(test_values, columns=FEATURE_COLUMNS, index=test.index)
    train_processed[TARGET_COLUMN] = train[TARGET_COLUMN]
    test_processed[TARGET_COLUMN] = test[TARGET_COLUMN]

    _write_csv_pair(train_processed, test_processed, output_dir, "train.csv", "test.csv")
    return train_processed, test_processed


if __name__ == "__main__":
    if not (RAW_DIR / "train_raw.csv").exists() and not (RAW_DIR / "test_raw.csv").exists():
        export_splits()
    train, test = export_processed()
    print(f"Saved {len(train):,} training rows to {PROCESSED_DIR / 'train.csv'}")
    print(f"Saved {len(test):,} test rows to {PROCESSED_DIR / 'test.csv'}")
