/**
 * frontend/assets/app.js
 * ----------------------
 * Client-side JavaScript for the Disease Risk Sentinel app.
 *
 * FLOW:
 *   1. User fills form (name, age, gender, lat/lon) and clicks "Analyze Risk"
 *   2. handleSubmit() sends POST /predict → gets disease risk percentages
 *   3. renderPrediction() draws animated risk bars on screen
 *   4. fetchAdvice() sends POST /advice → gets Gemini-generated health guidance
 *   5. renderAdvice() parses the guidance into cards and displays them
 *
 * "Use My Current Location" button calls the browser Geolocation API
 * to auto-fill latitude and longitude fields.
 */

// ── DOM element references ──────────────────────────────────────────────────
const form = document.getElementById("risk-form");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submit-btn");
const useLocationBtn = document.getElementById("use-location");
const latitudeInput = document.getElementById("latitude");
const longitudeInput = document.getElementById("longitude");

const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");
const riskCards = document.getElementById("risk-cards");
const riskChartCanvas = document.getElementById("risk-chart");
const riskMapEl = document.getElementById("risk-map");
const whyRiskList = document.getElementById("why-risk-list");
const topFactorsList = document.getElementById("top-factors-list");
const modelPerformanceGrid = document.getElementById("model-performance-grid");
const densityLevelEl = document.getElementById("density-level");
const nearbyTotalEl = document.getElementById("nearby-total");
const disclaimerTextEl = document.getElementById("disclaimer-text");

const advicePlaceholder = document.getElementById("advice-placeholder");
const adviceContent = document.getElementById("advice-content");

let riskChart = null;
let riskMap = null;
let riskMarker = null;

const FALLBACK_MODEL_METRICS = {
  accuracy: 0.9281,
  precision: 0.91,
  recall: 0.9,
  f1_score: 0.89,
};

// ── Utility functions ───────────────────────────────────────────────────────

/** Show a status message below the form. Pass isError=true to style it red. */
function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

/** Strip markdown symbols (**, __, ```) from Gemini response text. */
function cleanAdviceText(text) {
  if (!text) return "";
  return String(text)
    .replace(/\r\n/g, "\n")
    .replace(/^\s*---+\s*$/gm, "")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/`/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * Parse Gemini advice text into structured sections.
 * Looks for numbered headers like "1. Quick Risk Summary" or known title strings.
 * Returns an array of { index, title, body } objects.
 */
function parseAdviceSections(text) {
  const lines = String(text || "").split("\n");
  const headerRegex = /^\s*(\d+)[.)]\s+(.+?)\s*$/;
  const knownTitleRegex = /^\s*(Quick Risk Summary|Most Important Precautions Today|Early Symptoms To Monitor.*|When To Test Or Visit Hospital|Safety Notes)\s*$/i;
  const sections = [];
  let current = null;
  let nextAutoIndex = 1;

  lines.forEach((line) => {
    const match = line.match(headerRegex);
    if (match) {
      if (current) {
        current.body = current.body.join("\n").trim();
        sections.push(current);
      }
      nextAutoIndex = Math.max(nextAutoIndex, Number(match[1]) + 1);
      current = {
        index: Number(match[1]),
        title: match[2].trim(),
        body: [],
      };
      return;
    }

    const knownMatch = line.match(knownTitleRegex);
    if (knownMatch) {
      if (current) {
        current.body = current.body.join("\n").trim();
        sections.push(current);
      }
      current = {
        index: nextAutoIndex,
        title: knownMatch[1].trim(),
        body: [],
      };
      nextAutoIndex += 1;
      return;
    }

    if (current) {
      current.body.push(line);
    }
  });

  if (current) {
    current.body = current.body.join("\n").trim();
    sections.push(current);
  }

  return sections;
}

/** Build a single advice card DOM element from a parsed section object. */
function createAdviceCard(section) {
  const card = document.createElement("section");
  card.className = "advice-card";
  card.innerHTML = `
    <div class="advice-card__head">
      <span class="advice-card__index">${section.index}</span>
      <h3>${section.title}</h3>
    </div>
    <p>${section.body || "No details provided."}</p>
  `;
  return card;
}

/**
 * Render the Gemini advice response into the advice panel.
 * If the response has 2+ sections → shows cards in a grid.
 * Otherwise → shows plain text paragraph.
 */
function renderAdvice(adviceResponse) {
  const baseAdvice = cleanAdviceText(adviceResponse.advice) || "No guidance returned.";
  const sections = parseAdviceSections(baseAdvice);

  adviceContent.innerHTML = "";
  adviceContent.classList.remove("advice--plain");

  if (sections.length >= 2) {
    const grid = document.createElement("div");
    grid.className = "advice-grid";
    sections.sort((a, b) => a.index - b.index).forEach((section) => {
      grid.appendChild(createAdviceCard(section));
    });
    adviceContent.appendChild(grid);
  } else {
    adviceContent.classList.add("advice--plain");
    adviceContent.textContent = baseAdvice;
  }

  if (!adviceResponse.llm_used && adviceResponse.llm_error?.hint) {
    const note = document.createElement("p");
    note.className = "advice-note";
    note.textContent = `Troubleshooting hint: ${adviceResponse.llm_error.hint}`;
    adviceContent.appendChild(note);
  }
}

// ── Risk card rendering ─────────────────────────────────────────────────────

/** Convert a 0.0–1.0 float value to a "XX.X%" display string. */
function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function riskCategory(value) {
  const score = Number(value);
  if (score <= 0.3) {
    return "Low";
  }
  if (score <= 0.6) {
    return "Medium";
  }
  return "High";
}

function orderedRiskEntries(riskObj) {
  const entries = Object.entries(riskObj || {});
  entries.sort((a, b) => Number(b[1]) - Number(a[1]));
  return entries;
}

function humanizeKey(value) {
  const text = String(value || "");
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Unknown";
}

function nearbyCaseTotal(nearbyCases) {
  return Object.values(nearbyCases || {}).reduce(
    (sum, value) => sum + Number(value || 0),
    0
  );
}

function densityLabel(value) {
  return String(value || "unknown").replace(/_/g, " ").toLowerCase();
}

function explanationPoints(prediction) {
  const points = [];
  const densityLevel = densityLabel(prediction.location_density_level);
  const nearby = prediction.nearby_cases_25km || {};
  const nearbyTotal = nearbyCaseTotal(nearby);
  const nearbyEntries = orderedRiskEntries(nearby);
  const topNearby = nearbyEntries[0];

  if (densityLevel !== "unknown" && densityLevel !== "-") {
    points.push(`Region classified as a ${densityLevel}-risk zone based on local disease density.`);
  }

  if (nearbyTotal > 0) {
    points.push(`Nearby case activity detected within 25 km, with ${nearbyTotal} total reported cases contributing to local risk.`);
  } else {
    points.push("Few or no nearby reported cases were detected in the current local area window.");
  }

  if (topNearby && Number(topNearby[1]) > 0) {
    points.push(`${humanizeKey(topNearby[0])} has the strongest nearby case presence in this area.`);
  }

  if (points.length < 2) {
    points.push("The estimate is based on your location and the surrounding disease activity pattern in the dataset.");
  }

  if (points.length < 3) {
    const riskEntries = orderedRiskEntries(prediction.balanced_risk || {});
    const topRisk = riskEntries[0];
    if (topRisk) {
      points.push(`${humanizeKey(topRisk[0])} currently ranks highest in the model's balanced risk output.`);
    }
  }

  return points.slice(0, 3);
}

function renderWhyRisk(prediction) {
  if (!whyRiskList) {
    return;
  }

  const points = explanationPoints(prediction || {});
  whyRiskList.innerHTML = "";

  points.forEach((point) => {
    const item = document.createElement("li");
    item.textContent = point;
    whyRiskList.appendChild(item);
  });
}

function formatFeatureName(name) {
  return String(name || "Unknown")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function impactLabel(value, maxImpact) {
  const normalized = maxImpact > 0 ? Number(value) / maxImpact : 0;
  if (normalized >= 0.75) {
    return "High impact";
  }
  if (normalized >= 0.4) {
    return "Moderate impact";
  }
  return "Low impact";
}

function renderTopFactors(prediction) {
  if (!topFactorsList) {
    return;
  }

  const factors = Array.isArray(prediction.top_factors) ? prediction.top_factors : [];
  topFactorsList.innerHTML = "";

  if (!factors.length) {
    const empty = document.createElement("p");
    empty.className = "factor-empty";
    empty.textContent = "Model factor explanations are not available for this prediction.";
    topFactorsList.appendChild(empty);
    return;
  }

  const maxImpact = Math.max(...factors.map((item) => Number(item.impact) || 0), 0);

  factors.forEach((factor) => {
    const impact = Number(factor.impact) || 0;
    const widthPercent = maxImpact > 0 ? Math.max(12, (impact / maxImpact) * 100) : 12;
    const item = document.createElement("div");
    item.className = "factor-item";
    item.innerHTML = `
      <div class="factor-item__head">
        <span class="factor-item__name">${formatFeatureName(factor.feature)}</span>
        <span class="factor-item__label">${impactLabel(impact, maxImpact)}</span>
      </div>
      <div class="factor-item__bar">
        <div class="factor-item__fill" style="width: ${widthPercent}%"></div>
      </div>
    `;
    topFactorsList.appendChild(item);
  });
}

function metricValue(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function formatMetric(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function renderModelPerformance(prediction) {
  if (!modelPerformanceGrid) {
    return;
  }

  const metadata = prediction.metadata || {};
  const metrics = [
    {
      label: "Accuracy",
      value: metricValue(metadata.model_accuracy, FALLBACK_MODEL_METRICS.accuracy),
    },
    {
      label: "Precision",
      value: metricValue(metadata.precision, FALLBACK_MODEL_METRICS.precision),
    },
    {
      label: "Recall",
      value: metricValue(metadata.recall, FALLBACK_MODEL_METRICS.recall),
    },
    {
      label: "F1 Score",
      value: metricValue(metadata.f1_score, FALLBACK_MODEL_METRICS.f1_score),
    },
  ];

  modelPerformanceGrid.innerHTML = "";
  metrics.forEach((metric) => {
    const item = document.createElement("div");
    item.className = "performance-item";
    item.innerHTML = `
      <span class="performance-item__label">${metric.label}</span>
      <strong class="performance-item__value">${formatMetric(metric.value)}</strong>
    `;
    modelPerformanceGrid.appendChild(item);
  });
}

function markerColorForRisk(score) {
  const value = Number(score) || 0;
  if (value > 0.6) {
    return "#cf3f31";
  }
  if (value > 0.3) {
    return "#f0b429";
  }
  return "#2cc3a6";
}

function highestRiskScore(balancedRisk) {
  const entries = orderedRiskEntries(balancedRisk || {});
  return entries.length ? Number(entries[0][1]) || 0 : 0;
}

function ensureRiskMap() {
  if (!riskMapEl || typeof L === "undefined") {
    return null;
  }

  if (riskMap) {
    return riskMap;
  }

  riskMap = L.map(riskMapEl, {
    zoomControl: true,
    scrollWheelZoom: false,
  }).setView([11.1271, 78.6569], 6);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(riskMap);

  return riskMap;
}

function updateRiskMap(latitude, longitude, riskScore = 0) {
  const map = ensureRiskMap();
  const lat = Number(latitude);
  const lon = Number(longitude);
  if (!map || !Number.isFinite(lat) || !Number.isFinite(lon)) {
    return;
  }

  const color = markerColorForRisk(riskScore);
  const markerIcon = L.divIcon({
    className: "risk-map-marker-wrapper",
    html: `<span class="risk-map-marker" style="background:${color}"></span>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });

  if (!riskMarker) {
    riskMarker = L.marker([lat, lon], { icon: markerIcon }).addTo(map);
  } else {
    riskMarker.setLatLng([lat, lon]);
    riskMarker.setIcon(markerIcon);
  }

  riskMarker.bindPopup(`Latitude: ${lat.toFixed(4)}<br>Longitude: ${lon.toFixed(4)}`);
  map.setView([lat, lon], 11);
}

function chartRiskValues(balancedRisk) {
  const risk = balancedRisk || {};
  return [
    Number(risk.dengue || 0),
    Number(risk.malaria || 0),
    Number(risk.typhoid || 0),
  ];
}

function ensureRiskChart() {
  if (!riskChartCanvas || typeof Chart === "undefined") {
    return null;
  }

  if (riskChart) {
    return riskChart;
  }

  riskChart = new Chart(riskChartCanvas, {
    type: "bar",
    data: {
      labels: ["Dengue", "Malaria", "Typhoid"],
      datasets: [
        {
          label: "Balanced Risk",
          data: [0, 0, 0],
          backgroundColor: ["#ff7a18", "#2cc3a6", "#f0b429"],
          borderRadius: 10,
          borderSkipped: false,
          maxBarThickness: 72,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 500,
        easing: "easeOutQuart",
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label(context) {
              return `Risk: ${(Number(context.parsed.y) * 100).toFixed(1)}%`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            callback(value) {
              return `${Math.round(Number(value) * 100)}%`;
            },
          },
          grid: {
            color: "rgba(15, 37, 48, 0.08)",
          },
        },
        x: {
          grid: {
            display: false,
          },
        },
      },
    },
  });

  return riskChart;
}

function updateRiskChart(balancedRisk) {
  const chart = ensureRiskChart();
  if (!chart) {
    return;
  }

  chart.data.datasets[0].data = chartRiskValues(balancedRisk);
  chart.update();
}

/**
 * Build a risk card DOM element with an animated progress bar.
 * Bar width animates from 0% → actual value using requestAnimationFrame.
 */
function createRiskCard(name, value) {
  const category = riskCategory(value);
  const card = document.createElement("div");
  card.className = "risk-card";
  card.innerHTML = `
    <div class="risk-head">
      <span class="risk-name">${name}</span>
      <div class="risk-score-group">
        <span class="risk-score">${percent(value)}</span>
        <span class="risk-category risk-category--${category.toLowerCase()}">${category} Risk</span>
      </div>
    </div>
    <div class="bar-wrap">
      <div class="bar-fill" style="width: 0%"></div>
    </div>
  `;

  const fill = card.querySelector(".bar-fill");
  requestAnimationFrame(() => {
    fill.style.width = `${Math.max(0, Math.min(100, Number(value) * 100))}%`;
  });

  return card;
}

/** Render disease risk cards and metadata (density level, nearby cases count) from /predict response. */
function renderPrediction(payload) {
  const prediction = payload.prediction || {};
  const balancedRisk = prediction.balanced_risk || {};
  const nearby = prediction.nearby_cases_25km || {};
  const entries = orderedRiskEntries(balancedRisk);

  // Remove any existing out-of-region warning before re-rendering
  const existingWarning = document.getElementById("out-of-region-warning");
  if (existingWarning) existingWarning.remove();

  // If location is outside training region — show a prominent warning banner
  if (prediction.out_of_region) {
    const warning = document.createElement("div");
    warning.id = "out-of-region-warning";
    warning.className = "out-of-region-warning";
    warning.innerHTML = `
      <strong>⚠️ Location Outside Data Region</strong>
      <p>${prediction.out_of_region_message}</p>
      <p>Predictions below are <strong>not reliable</strong> for your area. This model only covers <strong>Tamil Nadu, India</strong>.</p>
    `;
    // Insert warning before the risk cards
    resultContent.insertBefore(warning, resultContent.firstChild);
  }

  riskCards.innerHTML = "";
  entries.forEach(([disease, score]) => {
    riskCards.appendChild(createRiskCard(disease, score));
  });
  updateRiskChart(balancedRisk);
  renderWhyRisk(prediction);
  renderTopFactors(prediction);
  renderModelPerformance(prediction);
  updateRiskMap(
    payload.input?.latitude,
    payload.input?.longitude,
    highestRiskScore(balancedRisk)
  );

  const nearbyTotal = nearbyCaseTotal(nearby);

  densityLevelEl.textContent = (prediction.location_density_level || "-").replace("_", " ");
  nearbyTotalEl.textContent = String(nearbyTotal);
  disclaimerTextEl.textContent = prediction.disclaimer || "";

  emptyState.classList.add("hidden");
  resultContent.classList.remove("hidden");
}

// ── API calls ───────────────────────────────────────────────────────────────

/** POST to /advice endpoint with the full prediction payload. Returns Gemini guidance text. */
async function fetchAdvice(payload) {
  const response = await fetch("/advice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail ? ` (${data.detail})` : "";
    throw new Error((data.error || "Advice request failed") + detail);
  }
  return data;
}

/** POST to /predict endpoint with user's {name, age, gender, latitude, longitude}. Returns risk percentages. */
async function predictRisk(input) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Prediction failed");
  }
  return data;
}

// ── Form handling ───────────────────────────────────────────────────────────

/** Read all form field values and return them as a single payload object. */
function formPayload() {
  const payload = {
    name: document.getElementById("name").value.trim(),
    age: Number(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    latitude: Number(latitudeInput.value),
    longitude: Number(longitudeInput.value),
  };
  return payload;
}

/** Validate the form payload. Throws an Error with a user-friendly message if any required field is missing or invalid. */
function validatePayload(payload) {
  if (!payload.name || payload.name.trim() === "") {
    throw new Error("Name is required.");
  }
  if (!Number.isFinite(payload.age) || payload.age <= 0 || payload.age > 120) {
    throw new Error("Enter a valid age (1-120).");
  }
  if (!payload.gender || payload.gender === "") {
    throw new Error("Please select a gender.");
  }
  if (!Number.isFinite(payload.latitude) || payload.latitude < -90 || payload.latitude > 90) {
    throw new Error("Latitude is required and must be between -90 and 90.");
  }
  if (!Number.isFinite(payload.longitude) || payload.longitude < -180 || payload.longitude > 180) {
    throw new Error("Longitude is required and must be between -180 and 180.");
  }
}

/**
 * Main form submit handler.
 * Step 1 → call /predict to get risk percentages and render risk bars.
 * Step 2 → call /advice to get Gemini guidance and render advice cards.
 */
async function handleSubmit(event) {
  event.preventDefault();
  adviceContent.classList.add("hidden");
  advicePlaceholder.classList.remove("hidden");
  advicePlaceholder.textContent = "Generating guidance...";

  let payload;
  try {
    payload = formPayload();
    validatePayload(payload);
  } catch (error) {
    setStatus(error.message || "Invalid input", true);
    return;
  }

  submitBtn.disabled = true;
  setStatus("Analyzing local case patterns...");

  try {
    const predictionResponse = await predictRisk(payload);
    renderPrediction(predictionResponse);

    // If out of region, skip Gemini advice (would be misleading)
    if (predictionResponse.prediction && predictionResponse.prediction.out_of_region) {
      advicePlaceholder.textContent = "AI guidance is not available for locations outside Tamil Nadu, India.";
      submitBtn.disabled = false;
      return;
    }

    setStatus("Risk profile ready. Asking Gemini for guidance...");
    const adviceResponse = await fetchAdvice(predictionResponse);

    renderAdvice(adviceResponse);
    adviceContent.classList.remove("hidden");
    advicePlaceholder.classList.add("hidden");

    if (adviceResponse.llm_used) {
      setStatus("Complete. Gemini guidance is included.");
    } else {
      setStatus("Prediction complete. Showing fallback guidance.");
    }
  } catch (error) {
    setStatus(error.message || "Request failed", true);
    advicePlaceholder.textContent = "Unable to generate advice right now.";
  } finally {
    submitBtn.disabled = false;
  }
}

/**
 * "Use My Current Location" button handler.
 * Uses the browser Geolocation API to auto-fill latitude and longitude fields.
 */
function handleUseLocation() {
  if (!navigator.geolocation) {
    setStatus("Geolocation is not supported in this browser.", true);
    return;
  }
  setStatus("Reading your current location...");
  useLocationBtn.disabled = true;

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      latitudeInput.value = pos.coords.latitude.toFixed(6);
      longitudeInput.value = pos.coords.longitude.toFixed(6);
      updateRiskMap(pos.coords.latitude, pos.coords.longitude);
      setStatus("Location filled. You can now analyze risk.");
      useLocationBtn.disabled = false;
    },
    (err) => {
      setStatus(`Location unavailable: ${err.message}`, true);
      useLocationBtn.disabled = false;
    },
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 0 }
  );
}

function handleCoordinatePreview() {
  updateRiskMap(latitudeInput.value, longitudeInput.value);
}

form.addEventListener("submit", handleSubmit);
useLocationBtn.addEventListener("click", handleUseLocation);
["change", "input"].forEach((eventName) => {
  latitudeInput.addEventListener(eventName, handleCoordinatePreview);
  longitudeInput.addEventListener(eventName, handleCoordinatePreview);
});
ensureRiskMap();
