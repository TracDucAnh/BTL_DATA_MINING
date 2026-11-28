"""Serve the Step 4 frontend and score raw measurements with saved checkpoints."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from eda_preprocessing.preprocessing import FEATURE_COLUMNS, RAW_DIR, load_pima_csv, prepare_predictors
from model.inference import load_manifest, load_model, to_frame

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
WHOLE_NUMBER_FIELDS = {"Age", "Pregnancies"}
MAX_MODEL_INPUT = float(np.finfo(np.float32).max)

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 4_096


@lru_cache(maxsize=1)
def training_reference() -> dict[str, tuple[np.ndarray, float, float]]:
    """Cache the observed training distribution and its central 90% for each feature."""
    train = prepare_predictors(load_pima_csv(RAW_DIR / "train_raw.csv"))
    reference = {}
    for feature in FEATURE_COLUMNS:
        observed = np.sort(train[feature].dropna().to_numpy(dtype=float))
        lower, upper = np.quantile(observed, [0.05, 0.95])
        reference[feature] = (observed, float(lower), float(upper))
    return reference


def compare_measurements(measurements) -> list[dict]:
    """Compare measured inputs to training values and omit model-imputed inputs."""
    reference = training_reference()
    comparisons = []
    for feature, measurement in measurements.dropna().items():
        value = float(measurement)
        observed, lower, upper = reference[feature]
        left = np.searchsorted(observed, value, side="left")
        right = np.searchsorted(observed, value, side="right")
        percentile = 50 * (int(left) + int(right)) / observed.size
        status = "within"
        if value < lower:
            status = "below"
        elif value > upper:
            status = "above"
        comparisons.append({
            "feature": feature,
            "value": value,
            "percentile": percentile,
            "lower": lower,
            "upper": upper,
            "status": status,
        })
    return comparisons


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/models")
def models():
    manifest = load_manifest()
    return jsonify(
        default_model=manifest["default_model"],
        models=[
            {"id": name, "label": entry["display_name"]}
            for name, entry in manifest["models"].items()
        ],
    )


@app.post("/api/predict")
def predict():
    if not request.is_json:
        return jsonify(error="Yêu cầu cần có định dạng JSON."), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.keys() - {"model", "measurements"}:
        return jsonify(error="Yêu cầu cần chứa chỉ số đo và tên mô hình nếu có."), 400

    measurements = payload.get("measurements")
    if not isinstance(measurements, dict) or measurements.keys() - set(FEATURE_COLUMNS):
        return jsonify(error="Danh sách chỉ số đo có cấu trúc hoặc tên trường không hợp lệ."), 400

    for feature, value in measurements.items():
        if value is None:
            continue
        if type(value) not in (int, float) or not 0 <= value <= MAX_MODEL_INPUT:
            return jsonify(error=f"{feature} cần là số không âm trong phạm vi xử lý của mô hình."), 400
        if feature in WHOLE_NUMBER_FIELDS and value != int(value):
            return jsonify(error=f"{feature} cần là số nguyên."), 400

    model_name = payload.get("model")
    if model_name is not None and (
        not isinstance(model_name, str) or model_name not in load_manifest()["models"]
    ):
        return jsonify(error="Mô hình đã chọn không tồn tại."), 400

    observed = prepare_predictors(to_frame(measurements)).iloc[0]
    if observed.isna().all():
        return jsonify(error="Hãy nhập ít nhất một chỉ số đã đo trước khi chạy mô hình."), 400

    result = load_model(model_name).predict(measurements)[0]
    response = jsonify(**result, reference=compare_measurements(observed))
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(RequestEntityTooLarge)
def oversized_request(_error):
    return jsonify(error="Yêu cầu vượt quá kích thước cho phép."), 413


@app.errorhandler(500)
def server_error(_error):
    return jsonify(error="Máy chủ không xử lý được yêu cầu. Hãy kiểm tra môi trường và checkpoint."), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
