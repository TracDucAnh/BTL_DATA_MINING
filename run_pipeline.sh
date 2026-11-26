#!/usr/bin/env bash
# Run the whole pipeline: EDA -> preprocessing -> training (step 2) -> evaluation (step 3).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "== [1/4] EDA ==";              "$PYTHON" eda_preprocessing/eda.py
echo "== [2/4] Preprocessing ==";    "$PYTHON" eda_preprocessing/preprocessing.py
echo "== [3/4] Training (step 2) =="; "$PYTHON" model/train.py
echo "== [4/4] Evaluation (step 3) =="; "$PYTHON" model/evaluate.py
