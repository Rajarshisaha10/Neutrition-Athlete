const API_BASE = window.ATHLETEEDGE_API_URL || "/api";

const riskSteps = [
  { key: "sport_type", title: "Sport type", category: "Athlete context", type: "select", options: ["Soccer", "Basketball", "Track", "Football", "Kabaddi", "Badminton"], value: "Soccer", help: "Choose the sport or the closest match to the athlete's main training.", guide: "If the athlete plays more than one sport, choose the one they trained hardest for this week." },
  { key: "gender", title: "Gender", category: "Athlete context", type: "select", options: ["Female", "Male"], value: "Female", help: "Select the athlete's gender as recorded by the team or coach.", guide: "Use the same value you would write in a training register." },
  { key: "age", title: "Age", category: "Athlete context", type: "number", min: 10, max: 70, step: 1, value: 24, help: "Enter the athlete's age in completed years.", guide: "If exact age is unknown, use the age from school, Aadhaar, team record, or the closest known year." },
  { key: "bmi", title: "Body size score", category: "Athlete context", type: "number", min: 12, max: 45, step: 0.1, value: 22.8, help: "BMI is a rough body-size number. If you know height and weight, BMI = weight kg divided by height in metres twice.", guide: "No calculator? Use 22 for lean build, 25 for average build, 28 for heavier build, and 18-20 for very thin build." },
  { key: "heart_rate", title: "Heart rate", category: "Body load", type: "number", min: 40, max: 220, step: 1, value: 156, help: "Heart rate means beats per minute after training or during hard effort.", guide: "Without a device, count pulse for 15 seconds and multiply by 4. If you cannot count, use 90 for easy, 130 for moderate, 160 for hard, 180 for very hard." },
  { key: "body_temperature", title: "Body temperature", category: "Body load", type: "number", min: 35, max: 41, step: 0.1, value: 37.4, help: "Body temperature helps detect heat strain after practice.", guide: "No thermometer? Use 36.8 if normal, 37.5 if warm and sweating, 38.0 if unusually hot or feverish." },
  { key: "hydration_level", title: "Hydration level", category: "Recovery inputs", type: "number", min: 0, max: 100, step: 1, value: 58, help: "Hydration level is how well-watered the athlete seems today, from 0 to 100.", guide: "Use urine color and thirst: 85 clear/pale, 65 light yellow, 45 dark yellow/thirsty, 25 dizzy or very dry mouth." },
  { key: "sleep_quality", title: "Sleep quality", category: "Recovery inputs", type: "select", options: ["1 (Almost no sleep)", "2", "3 (Okay sleep)", "4", "5 (Deep full sleep)"], value: "3 (Okay sleep)", help: "Rate last night's sleep 1 to 5.", guide: "1 is barely slept, 5 is deeply rested." },
  { key: "recovery_score", title: "Recovery score", category: "Recovery inputs", type: "select", options: ["1 (Very poor)", "2", "3 (Average)", "4", "5 (Excellent)"], value: "3 (Average)", help: "Rate how ready the body feels 1 to 5.", guide: "1 is heavy/pain, 5 is fresh/energetic." },
  { key: "stress_level", title: "Stress level", category: "Readiness inputs", type: "select", options: ["1 (Calm)", "2", "3 (Normal pressure)", "4", "5 (Very tense)"], value: "3 (Normal pressure)", help: "Rate current mental/physical stress 1 to 5.", guide: "1 is calm, 5 is overloaded or worried." },
  { key: "training_intensity", title: "Training intensity", category: "Workload inputs", type: "select", options: ["1 (Very easy)", "2", "3 (Moderate)", "4", "5 (Very hard)"], value: "3 (Moderate)", help: "How hard was the session 1 to 5?", guide: "1 is light, 5 is match-level intensity." },
  { key: "training_duration", title: "Training duration", category: "Workload inputs", type: "number", min: 0, max: 300, step: 1, value: 105, help: "Duration is total active training time in minutes.", guide: "Count warm-up, drills, running, gym, and match play. Do not count long sitting breaks." },
  { key: "fatigue_index", title: "Fatigue index", category: "Readiness inputs", type: "select", options: ["1 (Fresh)", "2", "3 (Heavy legs)", "4", "5 (Exhausted)"], value: "3 (Heavy legs)", help: "Rate how tired you feel 1 to 5.", guide: "1 is completely fresh, 5 is exhausted." },
];

const diagnosisSteps = [
  { key: "competition_level", title: "Competition level", category: "Athlete context", type: "select", options: ["Village", "School", "District", "State", "National"], value: "District", help: "Choose the level where the athlete usually competes.", guide: "If unsure, choose the level of the last official match or meet." },
  { key: "pain_location", title: "Pain location", category: "Pain map", type: "select", options: ["head", "neck", "shoulder", "elbow", "wrist", "back", "hip", "groin", "thigh", "hamstring", "knee", "shin", "calf", "ankle", "foot"], value: "knee", help: "Select where the main pain is felt.", guide: "Pick the single most painful area. If pain spreads, choose where it started." },
  { key: "onset", title: "Onset", category: "Injury story", type: "select", options: ["sudden", "gradual"], value: "sudden", help: "Sudden means it started in one moment. Gradual means it built up over days or weeks.", guide: "A twist, fall, or pop is usually sudden. Pain from repeated running or practice is usually gradual." },
  { key: "pain_severity", title: "Pain severity", category: "Symptoms", type: "number", min: 0, max: 10, step: 1, value: 8, help: "Rate pain from 0 to 10.", guide: "0 is no pain. 5 affects play. 8 means severe pain. 10 means unbearable." },
  { key: "swelling", title: "Swelling", category: "Symptoms", type: "boolean", value: true, help: "Swelling means the area looks puffy or larger than the other side.", guide: "Compare both sides of the body. If one side looks raised or tight, choose Yes." },
  { key: "bruising", title: "Bruising", category: "Symptoms", type: "boolean", value: false, help: "Bruising means blue, purple, red, or dark marks after injury.", guide: "Look around the painful area and below it, especially after a few hours." },
  { key: "instability", title: "Instability", category: "Symptoms", type: "boolean", value: true, help: "Instability means the joint feels like it may give way.", guide: "If the athlete says the knee or ankle feels loose, shaking, or unsafe, choose Yes." },
  { key: "locking", title: "Locking", category: "Symptoms", type: "boolean", value: false, help: "Locking means the joint gets stuck or cannot fully bend/straighten.", guide: "Ask if the joint catches or blocks movement during walking or squatting." },
  { key: "clicking", title: "Clicking", category: "Symptoms", type: "boolean", value: false, help: "Clicking is repeated clicking or catching with pain.", guide: "Only choose Yes if clicking is linked with pain or movement trouble." },
  { key: "popping_sound", title: "Popping sound", category: "Symptoms", type: "boolean", value: true, help: "A pop means the athlete heard or felt a pop at injury time.", guide: "Ask what they felt in the exact moment the pain began." },
  { key: "reduced_range_of_motion", title: "Reduced movement", category: "Symptoms", type: "boolean", value: true, help: "Reduced movement means they cannot move the area normally.", guide: "Compare to the other side. If bending, lifting, or straightening is limited, choose Yes." },
  { key: "weight_bearing_ability", title: "Weight bearing", category: "Symptoms", type: "select", options: ["normal", "limited", "difficult", "unable"], value: "difficult", help: "This means how well the athlete can stand or walk on the injured side.", guide: "Normal means walking normally. Unable means they cannot put weight on it." },
  { key: "twisting", title: "Twisting", category: "Mechanism", type: "boolean", value: true, help: "Did the body part twist during injury?", guide: "Common when the foot is planted and the body turns." },
  { key: "overuse", title: "Overuse", category: "Mechanism", type: "boolean", value: false, help: "Overuse means pain built from repeated training.", guide: "Choose Yes for repeated running, jumping, throwing, or practice without one clear accident." },
  { key: "direct_impact", title: "Direct impact", category: "Mechanism", type: "boolean", value: false, help: "Direct impact means hit by another player, ball, ground, or object.", guide: "Choose Yes if there was a clear blow to the painful area." },
  { key: "landing", title: "Landing", category: "Mechanism", type: "boolean", value: false, help: "Did pain start while landing from a jump?", guide: "Choose Yes if the athlete landed awkwardly or on another foot." },
  { key: "sprinting", title: "Sprinting", category: "Mechanism", type: "boolean", value: false, help: "Did pain start during fast running?", guide: "Sudden back-thigh or calf pain during sprinting is important." },
  { key: "running", title: "Running", category: "Mechanism", type: "boolean", value: false, help: "Was running a main part of the pain story?", guide: "Choose Yes for long runs, repeated running drills, or pain that worsens while running." },
  { key: "throwing", title: "Throwing", category: "Mechanism", type: "boolean", value: false, help: "Was throwing or overhead arm action involved?", guide: "Important for shoulder and elbow pain." },
  { key: "jumping", title: "Jumping", category: "Mechanism", type: "boolean", value: false, help: "Was jumping a major part of the session or injury?", guide: "Choose Yes for repeated jumps, rebounds, spikes, or jump landings." },
  { key: "cutting", title: "Change of direction", category: "Mechanism", type: "boolean", value: true, help: "Cutting means sudden change of direction.", guide: "Important in football, kabaddi, basketball, and badminton." },
  { key: "fall", title: "Fall", category: "Mechanism", type: "boolean", value: false, help: "Did the athlete fall before pain started?", guide: "Choose Yes for fall to ground, mat, court, or field." },
];

const riskValues = Object.fromEntries(riskSteps.map((step) => [step.key, step.value]));
const diagnosisValues = Object.fromEntries(diagnosisSteps.map((step) => [step.key, step.value]));

const $ = (selector) => document.querySelector(selector);
const form = $("#risk-form");
const heroTitle = $("#hero-title");
const heroCopy = $("#hero-copy");
const riskModeButton = $("#risk-mode");
const diagnosisModeButton = $("#diagnosis-mode");
const statusText = $("#status-text");
const riskProbability = $("#risk-probability");
const riskLevel = $("#risk-level");
const confidence = $("#confidence");
const threshold = $("#threshold");
const factors = $("#risk-factors");
const modelState = $("#model-state");
const bodyMapSport = $("#body-map-sport");
const bodyMapPoints = $("#body-map-points");
const bodyAreaList = $("#body-area-list");
const diagnosisPanel = $("#diagnosis-panel");
const diagnosisPredictions = $("#diagnosis-predictions");
const diagnosisReasoning = $("#diagnosis-reasoning");
const confirmForm = $("#confirm-form");
const feedbackForm = $("#feedback-form");
const confirmedDiagnosis = $("#confirmed-diagnosis");
const clinicianType = $("#clinician-type");
const feedbackHelpful = $("#feedback-helpful");
const feedbackRating = $("#feedback-rating");
const loadSample = $("#load-sample");
const startWizard = $("#start-wizard");
const startPanel = $("#start-panel");
const wizardPanel = $("#wizard-panel");
const reviewPanel = $("#review-panel");
const reviewTitle = $("#review-title");
const reviewCopy = $("#review-copy");
const wizardLabel = $("#wizard-label");
const wizardTitle = $("#wizard-title");
const stepCount = $("#step-count");
const stepBar = $("#step-bar");
const metricCategory = $("#metric-category");
const metricTitle = $("#metric-title");
const metricHelp = $("#metric-help");
const metricGuide = $("#metric-guide");
const metricLabelText = $("#metric-label-text");
const metricControl = $("#metric-control");
const reviewList = $("#review-list");
const prevStep = $("#prev-step");
const nextStep = $("#next-step");
const submitScan = $("#submit-scan");

let mode = "risk";
let currentStep = -1;
let hasPrediction = false;
let latestDiagnosisAssessmentId = null;

const sportBodyAreas = {
  Soccer: [{ label: "Knee", x: 92, y: 226 }, { label: "Ankle", x: 76, y: 312 }, { label: "Hamstring", x: 126, y: 176 }],
  Basketball: [{ label: "Ankle", x: 76, y: 312 }, { label: "Knee", x: 136, y: 226 }, { label: "Shoulder", x: 68, y: 92 }],
  Track: [{ label: "Hamstring", x: 94, y: 176 }, { label: "Calf", x: 140, y: 266 }, { label: "Ankle", x: 144, y: 312 }],
  Football: [{ label: "Shoulder", x: 68, y: 92 }, { label: "Knee", x: 136, y: 226 }, { label: "Back", x: 110, y: 124 }],
  Kabaddi: [{ label: "Knee", x: 92, y: 226 }, { label: "Shoulder", x: 152, y: 92 }, { label: "Back", x: 110, y: 124 }],
  Badminton: [{ label: "Shoulder", x: 152, y: 92 }, { label: "Wrist", x: 150, y: 158 }, { label: "Knee", x: 92, y: 226 }],
};

function activeSteps() {
  return mode === "risk" ? riskSteps : diagnosisSteps;
}

function activeValues() {
  return mode === "risk" ? riskValues : diagnosisValues;
}

function selectedSport() {
  return riskValues.sport_type || "Soccer";
}

function sentenceCase(text) {
  const cleaned = String(text || "").replace(/\s+/g, " ").trim();
  return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : "";
}

function setMode(nextMode) {
  mode = nextMode;
  currentStep = -1;
  hasPrediction = false;
  latestDiagnosisAssessmentId = null;
  riskModeButton.classList.toggle("active", mode === "risk");
  diagnosisModeButton.classList.toggle("active", mode === "diagnosis");
  diagnosisPanel.hidden = mode !== "diagnosis";
  heroTitle.textContent = mode === "risk" ? "Risk signal." : "Injury support.";
  heroCopy.textContent = mode === "risk"
    ? "Enter simple athlete details one step at a time. Each screen explains how to estimate the value without sensors, so coaches and village athletes can use the model with everyday observations."
    : "Answer symptom and mechanism questions one step at a time. The engine estimates likely injuries, explains its reasoning, and stores confirmed diagnoses for future learning.";
  riskProbability.textContent = "--";
  riskLevel.textContent = "Awaiting signal";
  confidence.textContent = "--";
  threshold.textContent = mode === "risk" ? "--" : "N/A";
  factors.innerHTML = `<li>${mode === "risk" ? "Complete the guided inputs to generate model reasoning." : "Complete injury support to generate reasoning."}</li>`;
  renderDiagnosisEmpty();
  renderStep();
  updateBodyMap();
}

function updateBodyMap() {
  const sport = selectedSport();
  const areas = sportBodyAreas[sport] || sportBodyAreas.Soccer;
  bodyMapSport.textContent = sport;
  bodyMapPoints.innerHTML = "";
  bodyAreaList.innerHTML = "";
  for (const [index, area] of areas.entries()) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "body-point-group");
    group.innerHTML = `<circle class="body-pulse" cx="${area.x}" cy="${area.y}" r="${18 + index * 2}" /><circle class="body-point" cx="${area.x}" cy="${area.y}" r="7" /><text x="${area.x + 12}" y="${area.y - 10}">${area.label}</text>`;
    bodyMapPoints.appendChild(group);
    const item = document.createElement("div");
    item.className = "body-area-chip";
    item.textContent = sentenceCase(area.label);
    bodyAreaList.appendChild(item);
  }
}

function createControl(step) {
  metricControl.innerHTML = "";
  let control;
  if (step.type === "select") {
    control = document.createElement("select");
    for (const option of step.options) {
      const item = document.createElement("option");
      item.value = option;
      item.textContent = sentenceCase(option);
      control.appendChild(item);
    }
  } else if (step.type === "boolean") {
    control = document.createElement("select");
    for (const [label, value] of [["Yes", "true"], ["No", "false"]]) {
      const item = document.createElement("option");
      item.value = value;
      item.textContent = label;
      control.appendChild(item);
    }
  } else {
    control = document.createElement("input");
    control.type = "number";
    control.min = step.min;
    control.max = step.max;
    control.step = step.step;
  }
  const values = activeValues();
  control.name = step.key;
  control.value = step.type === "boolean" ? String(Boolean(values[step.key])) : values[step.key];
  control.addEventListener("input", () => {
    const numberValue = Number(control.value);
    values[step.key] = step.type === "boolean"
      ? control.value === "true"
      : step.type === "number" && Number.isFinite(numberValue)
        ? numberValue
        : control.value;
    if (step.key === "training_intensity" || step.key === "training_duration") {
      const load = Math.round(Number(values.training_intensity || 0) * Number(values.training_duration || 0));
      if (!Number.isNaN(load)) values.training_load = load;
    }
    if (step.key === "sport_type") updateBodyMap();
  });
  metricControl.appendChild(control);
  setTimeout(() => control.focus(), 30);
}

function renderStep() {
  const steps = activeSteps();
  const total = steps.length;
  const isReview = currentStep >= total;
  const isStart = currentStep < 0;
  startPanel.hidden = !isStart;
  wizardPanel.hidden = isStart || isReview;
  reviewPanel.hidden = !isReview;
  prevStep.hidden = isStart;
  nextStep.hidden = isStart || isReview;
  submitScan.hidden = !isReview;
  submitScan.textContent = mode === "risk" ? "Run risk scan" : "Run injury support";

  if (isStart) {
    wizardLabel.textContent = mode === "risk" ? "Athlete metrics uplink" : "Injury questionnaire";
    wizardTitle.textContent = mode === "risk" ? "Start readiness profile." : "Start injury support.";
    startPanel.querySelector("p").textContent = mode === "risk"
      ? "This guided form asks for sport, age, body size, training load, sleep, hydration, stress, recovery, and fatigue. Use honest estimates from the last training day or the last 24 hours."
      : "This guided form asks where pain is, what happened, and what symptoms are present. Use simple observations and athlete answers.";
    stepCount.textContent = "Ready";
    stepBar.style.width = "0%";
    statusText.textContent = "API: waiting for input";
    return;
  }

  if (isReview) {
    wizardLabel.textContent = mode === "risk" ? "Athlete metrics uplink" : "Injury questionnaire";
    wizardTitle.textContent = hasPrediction ? (mode === "risk" ? "Risk signal results." : "Injury support results.") : (mode === "risk" ? "Transmit readiness profile." : "Transmit injury profile.");
    reviewTitle.textContent = hasPrediction ? "Results." : "Ready to scan.";
    reviewCopy.textContent = hasPrediction
      ? "Signal generated. Review the submitted profile below, then adjust any value and scan again if needed."
      : "Check the values below. If something looks wrong, go back and adjust it before sending the profile.";
    stepCount.textContent = `${total}/${total}`;
    stepBar.style.width = "100%";
    statusText.textContent = hasPrediction ? "API: signal received" : "API: profile ready";
    renderReview();
    return;
  }

  const step = steps[currentStep];
  wizardLabel.textContent = step.category;
  wizardTitle.textContent = step.title;
  stepCount.textContent = `${currentStep + 1}/${total}`;
  stepBar.style.width = `${((currentStep + 1) / total) * 100}%`;
  metricCategory.textContent = step.category;
  metricTitle.textContent = step.title;
  metricHelp.textContent = step.help;
  metricGuide.textContent = step.guide;
  metricLabelText.textContent = step.title;
  createControl(step);
  statusText.textContent = "API: collecting profile";
}

function renderReview() {
  reviewList.innerHTML = "";
  const values = activeValues();
  activeSteps().forEach((step, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "review-row";
    row.innerHTML = `<span>${step.title}</span><strong>${sentenceCase(values[step.key])}</strong>`;
    row.addEventListener("click", () => {
      currentStep = index;
      renderStep();
    });
    reviewList.appendChild(row);
  });
}

function parse1to5(value, maxVal) {
  if (typeof value === 'string') {
    const match = value.match(/^(\d)/);
    if (match) {
      const num = parseInt(match[1], 10);
      return (num / 5) * maxVal;
    }
  }
  return Number(value) || 0;
}

function riskPayload() {
  const payload = { ...riskValues };
  payload.sleep_quality = parse1to5(payload.sleep_quality, 10);
  payload.recovery_score = parse1to5(payload.recovery_score, 100);
  payload.stress_level = parse1to5(payload.stress_level, 1.0);
  payload.training_intensity = parse1to5(payload.training_intensity, 10);
  payload.fatigue_index = parse1to5(payload.fatigue_index, 100);
  payload.training_load = payload.training_intensity * Number(payload.training_duration || 0);
  return payload;
}

function diagnosisPayload(lastRiskResult = null) {
  return {
    athlete_profile: {
      age: riskValues.age,
      gender: riskValues.gender,
      sport: riskValues.sport_type,
      competition_level: diagnosisValues.competition_level,
    },
    pain_location: diagnosisValues.pain_location,
    onset: diagnosisValues.onset,
    symptoms: {
      pain_severity: diagnosisValues.pain_severity,
      swelling: diagnosisValues.swelling,
      bruising: diagnosisValues.bruising,
      instability: diagnosisValues.instability,
      locking: diagnosisValues.locking,
      clicking: diagnosisValues.clicking,
      popping_sound: diagnosisValues.popping_sound,
      reduced_range_of_motion: diagnosisValues.reduced_range_of_motion,
      weight_bearing_ability: diagnosisValues.weight_bearing_ability,
    },
    mechanism: {
      twisting: diagnosisValues.twisting,
      overuse: diagnosisValues.overuse,
      direct_impact: diagnosisValues.direct_impact,
      landing: diagnosisValues.landing,
      sprinting: diagnosisValues.sprinting,
      running: diagnosisValues.running,
      throwing: diagnosisValues.throwing,
      jumping: diagnosisValues.jumping,
      cutting: diagnosisValues.cutting,
      fall: diagnosisValues.fall,
    },
    risk_model_output: lastRiskResult ? {
      injury_risk_probability: lastRiskResult.injury_risk_probability,
      fatigue_score: riskValues.fatigue_index,
      workload_spike: riskValues.training_load,
      sleep_quality: riskValues.sleep_quality,
      recovery_index: riskValues.recovery_score,
      raw: lastRiskResult,
    } : null,
    athlete_metrics: riskPayload(),
  };
}

function setLoading(isLoading) {
  submitScan.disabled = isLoading;
  submitScan.textContent = isLoading ? "Scanning..." : mode === "risk" ? "Run risk scan" : "Run injury support";
}

function renderRiskResult(result) {
  const probability = Number(result.injury_risk_probability || 0);
  riskProbability.textContent = `${Math.round(probability * 100)}%`;
  riskLevel.textContent = result.risk_level || "Risk signal";
  confidence.textContent = result.confidence == null ? "--" : `${Math.round(Number(result.confidence) * 100)}%`;
  threshold.textContent = result.decision_threshold == null ? "--" : Number(result.decision_threshold).toFixed(2);
  factors.innerHTML = "";
  const topFactors = result.top_risk_factors?.length ? result.top_risk_factors : ["No dominant risk factor returned"];
  for (const factor of topFactors) {
    const li = document.createElement("li");
    li.textContent = sentenceCase(factor);
    factors.appendChild(li);
  }
  updateBodyMap();
}

function renderDiagnosisEmpty() {
  diagnosisPredictions.innerHTML = "<p>Switch to Injury Support and complete the questionnaire to estimate likely injuries.</p>";
  diagnosisReasoning.innerHTML = "<li>Awaiting questionnaire.</li>";
  confirmForm.hidden = true;
  feedbackForm.hidden = true;
}

function renderDiagnosisResult(result) {
  latestDiagnosisAssessmentId = result.assessment_id;
  riskProbability.textContent = `${Math.round(Number(result.confidence || 0) * 100)}%`;
  riskLevel.textContent = result.insufficient_confidence ? "Low confidence" : "Injury support";
  confidence.textContent = `${Math.round(Number(result.confidence || 0) * 100)}%`;
  threshold.textContent = "0.55";
  factors.innerHTML = "";
  const top = result.top_predictions || [];
  diagnosisPredictions.innerHTML = "";
  confirmedDiagnosis.innerHTML = "";
  top.forEach((prediction, index) => {
    const card = document.createElement("div");
    card.className = "diagnosis-card";
    card.innerHTML = `<span>${index + 1}</span><strong>${prediction.injury}</strong><em>${Math.round(prediction.probability * 100)}%</em>`;
    diagnosisPredictions.appendChild(card);
    const item = document.createElement("option");
    item.value = prediction.injury;
    item.textContent = prediction.injury;
    confirmedDiagnosis.appendChild(item);
  });
  if (result.message) {
    const note = document.createElement("p");
    note.className = "confidence-note";
    note.textContent = result.message;
    diagnosisPredictions.appendChild(note);
  }
  diagnosisReasoning.innerHTML = "";
  (result.reasoning?.length ? result.reasoning : ["No strong reasoning trace returned."]).forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = sentenceCase(reason);
    diagnosisReasoning.appendChild(li);
    const factor = document.createElement("li");
    factor.textContent = sentenceCase(reason);
    factors.appendChild(factor);
  });
  confirmForm.hidden = false;
  feedbackForm.hidden = false;
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    modelState.textContent = data.status === "ok" ? data.selected_model || "Ready" : "Not ready";
  } catch {
    modelState.textContent = "Offline";
  }
}

riskModeButton.addEventListener("click", () => setMode("risk"));
diagnosisModeButton.addEventListener("click", () => setMode("diagnosis"));
startWizard.addEventListener("click", () => { currentStep = 0; renderStep(); });
prevStep.addEventListener("click", () => { currentStep -= 1; renderStep(); });
nextStep.addEventListener("click", () => { currentStep += 1; renderStep(); });

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  statusText.textContent = "API: transmitting profile";
  try {
    if (mode === "risk") {
      const response = await fetch(`${API_BASE}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(riskPayload()) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed");
      renderRiskResult(data);
      localStorage.setItem('athleteedge_risk_context', JSON.stringify({risk: data}));
    } else {
      const riskResponse = await fetch(`${API_BASE}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(riskPayload()) });
      const riskData = riskResponse.ok ? await riskResponse.json() : null;
      const response = await fetch(`${API_BASE}/diagnose`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(diagnosisPayload(riskData)) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Diagnosis failed");
      renderRiskResult(riskData || {});
      renderDiagnosisResult(data);
      localStorage.setItem('athleteedge_risk_context', JSON.stringify({risk: riskData, diagnosis: data}));
    }
    hasPrediction = true;
    currentStep = activeSteps().length;
    renderStep();
    statusText.textContent = "API: signal received";
  } catch (error) {
    statusText.textContent = `API: ${error.message}`;
  } finally {
    setLoading(false);
  }
});

loadSample.addEventListener("click", () => {
  const values = activeValues();
  for (const step of activeSteps()) values[step.key] = step.value;
  currentStep = activeSteps().length;
  renderStep();
  updateBodyMap();
  statusText.textContent = "API: sample profile loaded";
});

confirmForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!latestDiagnosisAssessmentId) return;
  const response = await fetch(`${API_BASE}/confirm-diagnosis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assessment_id: latestDiagnosisAssessmentId, confirmed_diagnosis: confirmedDiagnosis.value, clinician_type: clinicianType.value }),
  });
  statusText.textContent = response.ok ? "API: confirmation stored for learning" : "API: confirmation failed";
});

feedbackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!latestDiagnosisAssessmentId) return;
  const response = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assessment_id: latestDiagnosisAssessmentId, helpful: feedbackHelpful.value === "true", rating: Number(feedbackRating.value) }),
  });
  statusText.textContent = response.ok ? "API: feedback stored" : "API: feedback failed";
});

setMode("risk");
checkHealth();
