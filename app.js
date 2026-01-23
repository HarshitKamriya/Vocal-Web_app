let videoEl, canvasEl, ctx;
let model = null;
let isRunning = false;

let voiceOn = true;
let lastSpoken = {}; // { key: { timestamp } }
const SPEAK_COOLDOWN_MS = 4000; // per object+position
const GLOBAL_SPEAK_INTERVAL_MS = 1500; // min gap between phrases
let lastSpeakTime = 0;

let currentTarget = "";
let confidenceThreshold = 0.5;

function getStatusLabel() {
  return document.getElementById("status-label");
}

window.addEventListener("DOMContentLoaded", () => {
  videoEl = document.getElementById("video");
  canvasEl = document.getElementById("overlay");
  ctx = canvasEl.getContext("2d");

  document.getElementById("start-btn").addEventListener("click", () => {
    startApp().catch((err) => {
      console.error(err);
      getStatusLabel().textContent = "Error: could not start guidance.";
    });
  });

  document
    .getElementById("toggle-voice-btn")
    .addEventListener("click", toggleVoice);

  document
    .getElementById("confidence-slider")
    .addEventListener("input", onConfidenceChange);

  document.getElementById("target-select").addEventListener("change", (e) => {
    currentTarget = e.target.value;
  });

  // Try to auto-start (may be blocked on some browsers until user gesture)
  startApp().catch(() => {
    getStatusLabel().textContent = "Tap 'Start Guidance' to allow camera.";
  });
});

async function initCamera() {
  const constraintsBase = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraintsBase);
  } catch (err) {
    // Fallback to any camera
    console.warn("Rear camera failed, falling back to any camera", err);
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
  }

  videoEl.srcObject = stream;

  return new Promise((resolve, reject) => {
    videoEl.onloadedmetadata = () => {
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
      resolve();
    };
    videoEl.onerror = (e) => reject(e);
  });
}

async function loadModel() {
  if (model) return model;
  getStatusLabel().textContent = "Loading object detection model…";
  model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
  return model;
}

async function startApp() {
  if (isRunning) return;
  isRunning = true;

  try {
    await initCamera();
    await loadModel();
    getStatusLabel().textContent = "Guidance running…";
    const startBtn = document.getElementById("start-btn");
    startBtn.textContent = "Running";
    startBtn.disabled = true;
    detectionLoop();
  } catch (err) {
    console.error(err);
    getStatusLabel().textContent = "Error starting camera or model.";
    isRunning = false;
    throw err;
  }
}

let lastDetectionTime = 0;
const DETECTION_INTERVAL_MS = 150; // tune this for performance

async function detectionLoop(timestamp) {
  if (!isRunning || !model) return;

  requestAnimationFrame(detectionLoop);

  if (!timestamp) return;
  if (timestamp - lastDetectionTime < DETECTION_INTERVAL_MS) {
    return;
  }
  lastDetectionTime = timestamp;

  if (videoEl.readyState < 2) return;

  try {
    const predictions = await model.detect(videoEl);
    renderDetections(predictions);
    handleVoiceGuidance(predictions);
  } catch (err) {
    console.error("Detection error", err);
  }
}

function renderDetections(predictions) {
  const w = canvasEl.width;
  const h = canvasEl.height;

  ctx.clearRect(0, 0, w, h);

  // Semi-transparent overlay
  ctx.fillStyle = "rgba(15, 23, 42, 0.25)";
  ctx.fillRect(0, 0, w, h);

  predictions.forEach((pred) => {
    if (pred.score < confidenceThreshold) return;

    const [x, y, width, height] = pred.bbox;
    const label = `${pred.class} ${(pred.score * 100).toFixed(0)}%`;

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, width, height);

    ctx.font = "14px system-ui";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
    ctx.fillRect(x, y - 22, textWidth + 10, 20);

    ctx.fillStyle = "#e5e7eb";
    ctx.fillText(label, x + 4, y - 8);
  });
}

function getSpatialDescriptor(pred) {
  const [x, y, width, height] = pred.bbox;
  const centerX = x + width / 2;
  const frameWidth = canvasEl.width;
  const frameHeight = canvasEl.height;

  const normalizedCenterX = centerX / frameWidth;
  let horizontal;
  if (normalizedCenterX < 0.33) horizontal = "left";
  else if (normalizedCenterX > 0.66) horizontal = "right";
  else horizontal = "center";

  const boxArea = width * height;
  const frameArea = frameWidth * frameHeight;
  const areaRatio = boxArea / frameArea;

  let distance;
  if (areaRatio > 0.25) distance = "very near";
  else if (areaRatio > 0.1) distance = "near";
  else distance = "far";

  return { horizontal, distance };
}

function buildPhrase(pred, spatial) {
  const baseLabel = pred.class.toLowerCase();

  const horizontalText =
    spatial.horizontal === "center"
      ? "in front"
      : spatial.horizontal === "left"
      ? "on the left"
      : "on the right";

  let distanceText;
  if (spatial.distance === "very near") distanceText = "very close";
  else if (spatial.distance === "near") distanceText = "near";
  else distanceText = "far";

  return `${capitalize(baseLabel)} ${horizontalText}, ${distanceText}.`;
}

function buildTargetPhrase(pred, spatial) {
  const base = buildPhrase(pred, spatial);
  return `Target ${base}`;
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function handleVoiceGuidance(predictions) {
  if (!voiceOn || !("speechSynthesis" in window)) return;

  const now = performance.now();
  if (now - lastSpeakTime < GLOBAL_SPEAK_INTERVAL_MS) {
    return;
  }

  let filtered = predictions.filter((p) => p.score >= confidenceThreshold);
  if (currentTarget) {
    filtered = filtered.filter((p) => p.class === currentTarget);
  }
  if (filtered.length === 0) return;

  const best = filtered.sort((a, b) => b.score - a.score)[0];
  const spatial = getSpatialDescriptor(best);

  const key = `${best.class}-${spatial.horizontal}-${spatial.distance}`;
  const last = lastSpoken[key];

  if (last && now - last.timestamp < SPEAK_COOLDOWN_MS) {
    return;
  }

  const phrase = currentTarget
    ? buildTargetPhrase(best, spatial)
    : buildPhrase(best, spatial);

  speak(phrase);
  lastSpoken[key] = { timestamp: now };
  lastSpeakTime = now;

  if (currentTarget && best.class === currentTarget && "vibrate" in navigator) {
    navigator.vibrate([80, 40, 80]);
  }

  getStatusLabel().textContent = phrase;
}

function speak(text) {
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.9;
  utterance.pitch = 0.9;
  utterance.volume = 1.0;
  window.speechSynthesis.speak(utterance);
}

function toggleVoice() {
  voiceOn = !voiceOn;
  const btn = document.getElementById("toggle-voice-btn");
  btn.setAttribute("aria-pressed", String(voiceOn));
  btn.textContent = voiceOn ? "Voice: On" : "Voice: Off";

  if (!voiceOn) {
    window.speechSynthesis.cancel();
    getStatusLabel().textContent = "Voice guidance muted.";
  } else {
    getStatusLabel().textContent = "Voice guidance enabled.";
  }
}

function onConfidenceChange(e) {
  confidenceThreshold = parseFloat(e.target.value);
  document.getElementById("confidence-value").textContent =
    confidenceThreshold.toFixed(2);
}

