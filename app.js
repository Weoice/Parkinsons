(() => {
  const canvas = document.getElementById("spiral-canvas");
  const clearBtn = document.getElementById("clear-btn");
  const statusText = document.getElementById("status-text");
  const resultIdle = document.getElementById("result-idle");
  const resultBody = document.getElementById("result-body");
  const resultError = document.getElementById("result-error");
  const resultPct = document.getElementById("result-pct");
  const resultSummary = document.getElementById("result-summary");
  const resultBar = document.getElementById("result-bar");
  const readoutBody = document.getElementById("readout-body");
  const ctx = canvas.getContext("2d");

  const SIZE = canvas.width;
  const CENTER = SIZE / 2;
  const MAX_RADIUS = SIZE * 0.42;
  const TURNS = 4.5;

  let points = [];
  let strokeId = 0;
  let drawing = false;
  let lastX = null;
  let lastY = null;
  let predictTimer = null;

  function drawGuide() {
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.save();
    ctx.strokeStyle = "rgba(91, 88, 79, 0.3)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const thetaMax = TURNS * 2 * Math.PI;
    const steps = 400;
    for (let i = 0; i <= steps; i++) {
      const theta = (i / steps) * thetaMax;
      const r = (MAX_RADIUS / thetaMax) * theta;
      const x = CENTER + r * Math.cos(theta);
      const y = CENTER + r * Math.sin(theta);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function canvasPoint(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = SIZE / rect.width;
    const scaleY = SIZE / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
      t: e.timeStamp,
      pressure: e.pressure || 0.5,
      stroke: strokeId,
    };
  }

  function addPoint(e) {
    const p = canvasPoint(e);
    points.push(p);
    return p;
  }

  function formatFeatureValue(value) {
    const abs = Math.abs(value);
    if (abs !== 0 && abs < 0.001) return value.toExponential(2);
    return value.toFixed(4);
  }

  function resetReadout() {
    readoutBody.innerHTML =
      '<tr class="readout-empty"><td colspan="2">Draw a spiral to populate this table.</td></tr>';
  }

  function renderReadout(features) {
    readoutBody.innerHTML = "";
    for (const [name, value] of Object.entries(features)) {
      const row = document.createElement("tr");
      const label = document.createElement("td");
      label.textContent = name.replace(/_/g, " ");
      const val = document.createElement("td");
      val.textContent = formatFeatureValue(value);
      row.append(label, val);
      readoutBody.appendChild(row);
    }
  }

  function showIdle() {
    resultIdle.classList.remove("hidden");
    resultBody.classList.add("hidden");
    resultError.classList.add("hidden");
    resetReadout();
  }

  function showError(message) {
    resultIdle.classList.add("hidden");
    resultBody.classList.add("hidden");
    resultError.classList.remove("hidden");
    resultError.textContent = message;
    resetReadout();
  }

  function showResult(data) {
    resultIdle.classList.add("hidden");
    resultError.classList.add("hidden");
    resultBody.classList.remove("hidden");
    const pct = Math.round(data.probability * 100);
    resultPct.textContent = `${pct}%`;
    resultBar.style.width = `${pct}%`;
    const label = data.similarity === "parkinson" ? "Parkinson's" : "control";
    resultSummary.innerHTML =
      `Closer to <strong>${label}</strong> examples in the training set ` +
      `(decision at ${Math.round(data.threshold * 100)}%).`;
    if (data.features) renderReadout(data.features);
  }

  async function predict() {
    if (!points.length) {
      showIdle();
      return;
    }
    statusText.textContent = "Scoring...";
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points }),
      });
      const data = await res.json();
      if (!res.ok) {
        showError(data.error || "Prediction failed.");
        statusText.textContent = `${points.length} points captured`;
        return;
      }
      showResult(data);
      statusText.textContent = `${points.length} points captured`;
    } catch (_) {
      showError("Could not reach the prediction service.");
      statusText.textContent = `${points.length} points captured`;
    }
  }

  function schedulePredict() {
    clearTimeout(predictTimer);
    predictTimer = setTimeout(predict, 50);
  }

  canvas.addEventListener("pointerdown", (e) => {
    drawing = true;
    strokeId += 1;
    canvas.setPointerCapture(e.pointerId);
    const p = addPoint(e);
    lastX = p.x;
    lastY = p.y;
    statusText.textContent = "Drawing...";
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = addPoint(e);
    ctx.strokeStyle = "#2b4c5e";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    lastX = p.x;
    lastY = p.y;
  });

  function endStroke(e) {
    if (!drawing) return;
    drawing = false;
    addPoint(e);
    statusText.textContent = `${points.length} points captured`;
    schedulePredict();
  }

  canvas.addEventListener("pointerup", endStroke);
  canvas.addEventListener("pointercancel", endStroke);

  clearBtn.addEventListener("click", () => {
    points = [];
    strokeId = 0;
    drawing = false;
    clearTimeout(predictTimer);
    drawGuide();
    statusText.textContent = "Draw a spiral above";
    showIdle();
  });

  drawGuide();
  showIdle();
})();
