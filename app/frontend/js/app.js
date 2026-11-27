"use strict";

const form = document.querySelector("#measurement-form");
const modelSelect = document.querySelector("#model");
const submit = document.querySelector("#submit-prediction");
const submitLabel = document.querySelector("#submit-label");
const formError = document.querySelector("#form-error");
const reportButton = document.querySelector("#print-report");
const announcements = document.querySelector("#announcements");
const radar = document.querySelector("#radar");

const features = [
  { key: "Glucose", name: "Đường huyết", unit: "mg/dL", zeroMissing: true },
  { key: "BloodPressure", name: "Huyết áp tâm trương", axis: "Huyết áp", unit: "mmHg", zeroMissing: true },
  { key: "BMI", name: "BMI", unit: "kg/m²", zeroMissing: true },
  { key: "Age", name: "Tuổi", unit: "năm" },
  { key: "Insulin", name: "Insulin", unit: "µU/mL", zeroMissing: true },
  { key: "SkinThickness", name: "Nếp gấp da", unit: "mm", zeroMissing: true },
  { key: "Pregnancies", name: "Số lần mang thai", axis: "Lần mang thai", unit: "lần" },
  { key: "DiabetesPedigreeFunction", name: "Điểm tiền sử gia đình", axis: "Tiền sử GĐ" },
];
const byKey = new Map(features.map((feature) => [feature.key, feature]));
const sample = {
  Pregnancies: 6,
  Glucose: 148,
  BloodPressure: 72,
  SkinThickness: 35,
  BMI: 33.6,
  DiabetesPedigreeFunction: 0.627,
  Age: 50,
};
const percent = new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 1 });
const number = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 3 });
const rank = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 });

let modelsReady = false;
let activeRequest = null;
let shownReference = [];

function setBusy(busy) {
  submit.disabled = busy || !modelsReady;
  submitLabel.textContent = busy ? "Đang dự đoán…" : "Chạy dự đoán";
  form.setAttribute("aria-busy", String(busy));
}

function showError(message = "") {
  formError.textContent = message;
  formError.hidden = !message;
}

async function api(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload) {
    throw new Error(payload?.error || "Máy chủ chưa xử lý được yêu cầu. Hãy thử lại.");
  }
  return payload;
}

async function loadModels() {
  try {
    const { models, default_model: defaultModel } = await api("/api/models");
    modelSelect.replaceChildren(...models.map(({ id, label }) => new Option(
      label, id, id === defaultModel, id === defaultModel,
    )));
    modelSelect.disabled = false;
    modelsReady = true;
    setBusy(false);
  } catch {
    modelSelect.replaceChildren(new Option("Không tải được mô hình"));
    showError("Không kết nối được với API. Kiểm tra backend và tải lại trang.");
  }
}

function showReport(visible) {
  document.querySelector("#result-empty").hidden = visible;
  document.querySelector("#result-data").hidden = !visible;
  document.querySelector("#reference-empty").hidden = visible;
  document.querySelector("#reference-data").hidden = !visible;
  reportButton.hidden = !visible;
}

function clearReport() {
  if (activeRequest) {
    activeRequest.abort();
    activeRequest = null;
  }
  setBusy(false);
  if (modelsReady) showError();
  shownReference = [];
  showReport(false);
  document.querySelector("#result-model-label").textContent = "Điểm và ngưỡng quyết định từ checkpoint.";
  announcements.textContent = "";
}

function showResult(result) {
  const scoreText = percent.format(result.probability);
  const thresholdText = percent.format(result.threshold);
  document.querySelector("#result-model-label").textContent = result.display_name;
  document.querySelector("#score").textContent = scoreText;

  const badge = document.querySelector("#risk-badge");
  badge.textContent = result.high_risk ? "Đạt ngưỡng mô hình" : "Dưới ngưỡng mô hình";
  badge.classList.toggle("is-high", result.high_risk);

  const scale = document.querySelector("#risk-scale");
  scale.style.setProperty("--score-position", String(result.probability * 100) + "%");
  scale.style.setProperty("--threshold-position", String(result.threshold * 100) + "%");
  scale.setAttribute("aria-label", "Điểm " + scoreText + ", ngưỡng " + thresholdText);
  document.querySelector("#threshold-label").textContent = "Ngưỡng " + thresholdText;

  const missing = result.imputed_fields.map((key) => byKey.get(key)?.name ?? key);
  document.querySelector("#missing-summary").textContent = missing.length
    ? "Đã điền " + missing.length + " chỉ số thiếu (" + missing.join(", ") + ") bằng KNNImputer (k=5), dựa trên các mẫu gần nhất trong tập huấn luyện."
    : "Cả 8 chỉ số đều có số liệu đầu vào.";

  shownReference = result.reference;
  renderTable(shownReference);
  showReport(true);
  drawRadar();
  announcements.textContent = "Đã có kết quả. Điểm mô hình " + scoreText + ". " + badge.textContent + ".";
  if (window.matchMedia("(max-width: 920px)").matches) {
    document.querySelector("#result-heading").focus();
  }
}

function renderTable(reference) {
  const entries = new Map(reference.map((entry) => [entry.feature, entry]));
  const rows = features.filter(({ key }) => entries.has(key)).map(({ key, name, unit }) => {
    const entry = entries.get(key);
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = name;

    const value = document.createElement("td");
    value.textContent = number.format(entry.value) + (unit ? " " + unit : "");
    const percentile = document.createElement("td");
    percentile.textContent = "P" + rank.format(entry.percentile);
    const range = document.createElement("td");
    range.textContent = number.format(entry.lower) + "–" + number.format(entry.upper);
    if (entry.status !== "within") {
      percentile.classList.add("percentile-outside");
      range.textContent += entry.status === "below" ? " · Dưới" : " · Trên";
    }
    row.append(label, value, percentile, range);
    return row;
  });
  document.querySelector("#reference-rows").replaceChildren(...rows);
}

function drawRadar() {
  if (!shownReference.length) return;

  const width = radar.clientWidth;
  const height = radar.clientHeight;
  if (!width || !height) return;

  const pixelRatio = window.devicePixelRatio || 1;
  radar.width = Math.round(width * pixelRatio);
  radar.height = Math.round(height * pixelRatio);
  const context = radar.getContext("2d");
  if (!context) return;
  context.scale(pixelRatio, pixelRatio);

  const radius = Math.min(width / 2 - 72, height / 2 - 30);
  const center = { x: width / 2, y: height / 2 };
  const values = new Map(shownReference.map((entry) => [entry.feature, entry]));

  function point(index, percentOfRadius) {
    const angle = index * 2 * Math.PI / features.length - Math.PI / 2;
    return {
      x: center.x + Math.cos(angle) * radius * percentOfRadius,
      y: center.y + Math.sin(angle) * radius * percentOfRadius,
    };
  }

  function path(points, close = false) {
    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    for (const next of points.slice(1)) context.lineTo(next.x, next.y);
    if (close) context.closePath();
  }

  context.strokeStyle = "#d2e2e3";
  context.lineWidth = 1;
  for (const ring of [0.25, 0.5, 0.75, 1]) {
    path(features.map((_feature, index) => point(index, ring)), true);
    context.stroke();
  }
  for (const [index] of features.entries()) {
    path([center, point(index, 1)]);
    context.stroke();
  }

  context.font = "600 11px Segoe UI, system-ui, sans-serif";
  context.fillStyle = "#35545b";
  context.textBaseline = "middle";

  for (const [index, feature] of features.entries()) {
    const endpoint = point(index, 1 + 9 / radius);
    const dx = endpoint.x - center.x;
    context.textAlign = Math.abs(dx) < 3 ? "center" : dx < 0 ? "right" : "left";
    const y = endpoint.y + (index === 0 ? -3 : index === 4 ? 3 : 0);
    context.fillText(feature.axis ?? feature.name, endpoint.x, y);
  }

  const points = features.map(({ key }, index) => {
    const entry = values.get(key);
    if (!entry) return null;
    const position = Math.min(100, Math.max(0, entry.percentile)) / 100;
    return { ...point(index, position), status: entry.status };
  });

  context.strokeStyle = "#0a7a6d";
  context.lineWidth = 2;
  if (points.every(Boolean)) {
    path(points, true);
    context.fillStyle = "#22988930";
    context.fill();
    context.stroke();
  } else {
    for (const [index, current] of points.entries()) {
      const next = points[(index + 1) % points.length];
      if (!current || !next) continue;
      path([current, next]);
      context.stroke();
    }
  }

  for (const item of points.filter(Boolean)) {
    context.beginPath();
    context.arc(item.x, item.y, 4, 0, 2 * Math.PI);
    context.fillStyle = item.status === "within" ? "#087469" : "#a44c22";
    context.fill();
    context.strokeStyle = "#fff";
    context.lineWidth = 1.4;
    context.stroke();
  }
}

form.addEventListener("input", clearReport);
form.addEventListener("reset", clearReport);
document.querySelector("#load-sample").addEventListener("click", () => {
  clearReport();
  for (const { key } of features) document.getElementById(key).value = sample[key] ?? "";
});
reportButton.addEventListener("click", () => window.print());

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!modelsReady || activeRequest) return;

  clearReport();
  const measurements = {};
  for (const { key } of features) {
    const value = document.getElementById(key).valueAsNumber;
    if (Number.isFinite(value)) measurements[key] = value;
  }
  const hasObservation = features.some(({ key, zeroMissing }) =>
    key in measurements && (!zeroMissing || measurements[key] !== 0),
  );
  if (!hasObservation) {
    showError("Nhập ít nhất một chỉ số hợp lệ để chạy dự đoán.");
    return;
  }

  const request = new AbortController();
  activeRequest = request;
  setBusy(true);
  announcements.textContent = "Đang chạy dự đoán.";

  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelSelect.value, measurements }),
      signal: request.signal,
    });
    if (activeRequest === request) showResult(result);
  } catch (error) {
    if (activeRequest === request) {
      announcements.textContent = "";
      showError(error.message);
    }
  } finally {
    if (activeRequest === request) {
      activeRequest = null;
      setBusy(false);
    }
  }
});

new ResizeObserver(drawRadar).observe(radar);
window.addEventListener("beforeprint", drawRadar);
loadModels();
