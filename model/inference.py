"""Load trained checkpoints and score patients; shared by evaluate.py and the app backend.

Usage from the backend:

    from model.inference import load_model
    model = load_model()                    # default model from the manifest
    model.predict({"Glucose": 148, "BMI": 33.6, "Age": 50, ...})

Command line:

    python model/inference.py --list
    python model/inference.py '{"Glucose": 148, "BMI": 33.6, "Age": 50}' --model svm_rbf
"""

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Checkpoints reference eda_preprocessing.preprocessing.prepare_predictors, so the
# project root must be importable before they are unpickled.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eda_preprocessing.preprocessing import FEATURE_COLUMNS, prepare_predictors  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
MANIFEST_PATH = CHECKPOINT_DIR / "manifest.json"
MANIFEST_VERSION = 1


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH.relative_to(PROJECT_ROOT)} not found; run `python model/train.py` first.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported checkpoint manifest version {manifest.get('manifest_version')!r}.")
    return manifest


def available_models() -> list[str]:
    return list(load_manifest()["models"])


def to_frame(records: dict | list[dict]) -> pd.DataFrame:
    """Validate patient records and return them as a DataFrame of the eight predictors.

    Absent or null fields become NaN and are imputed by the checkpoint, exactly like
    the invalid zeros of the training data.
    """
    rows = [records] if isinstance(records, dict) else list(records)
    if not rows:
        raise ValueError("At least one record is required.")

    cleaned = []
    for position, row in enumerate(rows):
        unknown = sorted(set(row) - set(FEATURE_COLUMNS))
        if unknown:
            raise ValueError(f"Record {position}: unknown fields {unknown}; expected {FEATURE_COLUMNS}.")
        values = {}
        for feature in FEATURE_COLUMNS:
            raw = row.get(feature)
            if raw is None or raw == "":
                values[feature] = np.nan
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"Record {position}: {feature} must be a number, got {raw!r}.") from None
            if math.isnan(value):
                values[feature] = np.nan
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Record {position}: {feature} must be a finite non-negative number, got {raw!r}.")
            values[feature] = value
        cleaned.append(values)
    return pd.DataFrame(cleaned, columns=FEATURE_COLUMNS)


@dataclass(frozen=True)
class RiskModel:
    """A trained pipeline (preprocessing + classifier) with its screening threshold."""

    name: str
    display_name: str
    threshold: float
    pipeline: Pipeline

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """P(Outcome = 1) for a DataFrame holding the eight raw predictors."""
        return self.pipeline.predict_proba(features[FEATURE_COLUMNS])[:, 1]

    def predict(self, records: dict | list[dict]) -> list[dict]:
        """Risk report for each patient record."""
        features = to_frame(records)
        probabilities = self.predict_proba(features)
        imputed = prepare_predictors(features).isna()
        return [
            {
                "model": self.name,
                "display_name": self.display_name,
                "probability": float(probability),
                "threshold": self.threshold,
                "high_risk": bool(probability >= self.threshold),
                "imputed_fields": [feature for feature in FEATURE_COLUMNS if imputed.at[index, feature]],
            }
            for index, probability in enumerate(probabilities)
        ]


@lru_cache(maxsize=None)
def load_model(name: str | None = None) -> RiskModel:
    """Load one checkpoint (the manifest's default model when name is None); cached."""
    manifest = load_manifest()
    name = name or manifest["default_model"]
    if name not in manifest["models"]:
        raise ValueError(f"Unknown model {name!r}; available: {list(manifest['models'])}.")

    trained_with = manifest["environment"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        warnings.warn(
            f"Checkpoints were saved with scikit-learn {trained_with} but {sklearn.__version__} is installed; "
            "install requirements.txt or retrain.",
            stacklevel=2,
        )

    entry = manifest["models"][name]
    return RiskModel(
        name=name,
        display_name=entry["display_name"],
        threshold=entry["threshold"],
        pipeline=joblib.load(CHECKPOINT_DIR / entry["file"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score patients with a trained checkpoint.")
    parser.add_argument("record", nargs="?", help="JSON object (or list of objects) with the eight predictors")
    parser.add_argument("--model", help="checkpoint name; defaults to the manifest's default model")
    parser.add_argument("--list", action="store_true", help="list available checkpoints and exit")
    args = parser.parse_args()

    if args.list:
        manifest = load_manifest()
        for name, entry in manifest["models"].items():
            marker = "*" if name == manifest["default_model"] else " "
            print(f"{marker} {name:<20} threshold {entry['threshold']:.3f}  CV AUC {entry['cv_roc_auc_mean']:.3f}  {entry['file']}")
        return
    if args.record is None:
        parser.error("a JSON record is required unless --list is given")

    print(json.dumps(load_model(args.model).predict(json.loads(args.record)), indent=2))


if __name__ == "__main__":
    main()
