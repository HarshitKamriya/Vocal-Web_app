let videoEl, canvasEl, ctx;

// YOLOv8 via ONNX Runtime Web
let yoloSession = null;
const YOLO_MODEL_URL = "models/yolov8n.onnx"; // Place YOLOv8n.onnx here
const YOLO_INPUT_SIZE = 640;

// COCO80 class list used by YOLOv8
const COCO_CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];
let isRunning = false;

let voiceOn = true;
let lastSpoken = {}; // { key: { timestamp } }
const SPEAK_COOLDOWN_MS = 4000; // per object+position
const GLOBAL_SPEAK_INTERVAL_MS = 1500; // min gap between phrases
let lastSpeakTime = 0;

let currentTarget = "";
let confidenceThreshold = 0.5;

// -------- Monocular step estimation (heuristic) --------
// Approximate real-world heights (meters) for some common classes.
// Used with a simple pinhole model to estimate distance from bbox height.
const CLASS_HEIGHT_M = {
  person: 1.7,
  chair: 0.9,
  "dining table": 0.75,
  couch: 0.9,
  bed: 0.9,
  car: 1.4,
  bus: 3.0,
  truck: 3.0,
  bicycle: 1.1,
  motorcycle: 1.2,
  "cell phone": 0.15,
  laptop: 0.02,
  bottle: 0.25,
  door: 2.0,
  "stop sign": 2.1,
};

// Focal length in pixels (rough heuristic for typical phone rear camera).
// You can later calibrate this per device for better absolute accuracy.
const FOCAL_LENGTH_PX = 900;

// Average user step / stride length in meters (can be user-calibrated later).
const USER_STRIDE_M = 0.7;

// Only trust distance when bbox is reasonably tall in the frame.
const MIN_BOX_HEIGHT_PX = 20;
// Scale factor for converting relative box height to approximate steps.
const STEP_SCALE_FACTOR = 1.8;

// Speak only when estimated steps change by at least this much.
const STEP_CHANGE_THRESHOLD = 1;

// If target is this close in steps (or closer), announce "target found" and stop.
const TARGET_FOUND_MAX_STEPS = 2;

// Track last estimated integer steps per class+side to decide if it's getting closer/farther.
// key -> { steps: number, timestamp: number }
let lastStepsMap = {};

// Speech recognition (voice input for target)
let recognition = null;
let isListeningForTarget = false;
let waitingForInitialTarget = false; // Flag to track if we're waiting for initial target
let detectionStarted = false; // Track if detection loop has started
let startAppInFlight = null; // prevents double-start when buttons are tapped quickly

// -------- Double-tap state machine --------
const GUIDANCE_STATE = { STOPPED: "STOPPED", RUNNING: "RUNNING", ASK_TARGET: "ASK_TARGET", CURRENCY_DETECTION: "CURRENCY_DETECTION" };
// RUNNING = Guidance Mode (voice commands active). CURRENCY_DETECTION = sub-state while detecting currency.
let guidanceState = GUIDANCE_STATE.STOPPED;
const DOUBLE_TAP_THRESHOLD_MS = 300;
let lastTapTime = 0;
let lastRunDoubleTapWasStop = false; // from RUNNING: next double-tap stops (false) or asks target (true)
let askTargetMode = null; // null | "initial" | "while_running" (for ASK_TARGET cancel behavior)

// -------- Currency detection (voice-only, Guidance Mode only) --------
let commandRecognition = null;
let commandRecognitionStarted = false;
const CURRENCY_COMMAND_PHRASES = ["detect currency", "currency detection", "identify currency", "what currency"];
const CURRENCY_COMMAND_COOLDOWN_MS = 5000;
let lastCurrencyCommandTime = 0;

function getStatusLabel() {
  return document.getElementById("status-label");
}

// Auto-start guidance when entering detect page
function autoStartGuidance() {
  // Speak welcome message
  const welcomeMsg = "Welcome to guidance mode. " +
    "Tap the green button to start detection, or hold it for currency detection. " +
    "Hold the red button for emergency mode.";

  getStatusLabel().textContent = "Tap green button to start, or hold for currency";

  // Speak after a short delay to ensure speech synthesis is ready
  setTimeout(() => {
    speak(welcomeMsg);
  }, 500);

  // Auto-start the camera and model after welcome
  setTimeout(() => {
    if (!isRunning) {
      getStatusLabel().textContent = "Starting camera...";
      startApp().catch((err) => {
        console.warn("Auto-start failed:", err);
        getStatusLabel().textContent = "Tap green button to start guidance";
      });
    }
  }, 4000);
}

window.addEventListener("DOMContentLoaded", () => {
  videoEl = document.getElementById("video");
  canvasEl = document.getElementById("overlay");
  ctx = canvasEl.getContext("2d");

  document.getElementById("start-btn").addEventListener("click", () => {
    // Let startApp handle its own errors and update status label
    startApp();
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

  // Currency Detection Button - Hold to detect, Tap to start guidance
  initCurrencyButton();

  // Emergency Mode Button - Long Press to Activate
  initEmergencyButton();

  // Cancel Emergency Button
  const cancelEmergencyBtn = document.getElementById("cancel-emergency-btn");
  if (cancelEmergencyBtn) {
    cancelEmergencyBtn.addEventListener("click", cancelEmergency);
  }

  initSpeechRecognition();
  initCommandRecognition();
  initDoubleTapHandler();

  // Auto-start guidance when entering detect page
  autoStartGuidance();
});

async function initCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const msg = "Camera not supported in this browser. Try Chrome or Edge.";
    console.error(msg);
    getStatusLabel().textContent = msg;
    throw new Error(msg);
  }

  const constraintsBase = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };

  let stream = null;
  try {
    getStatusLabel().textContent = "Starting camera…";
    stream = await navigator.mediaDevices.getUserMedia(constraintsBase);
  } catch (err) {
    // Fallback to any camera
    console.warn("Rear camera failed, falling back to any camera", err);
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    } catch (err2) {
      console.error("Camera error (fallback also failed)", err2);
      const reason =
        err2.name === "NotAllowedError"
          ? "Permission denied. Check browser camera permissions."
          : err2.name === "NotFoundError"
            ? "No camera device found."
            : err2.name === "NotReadableError"
              ? "Camera is busy or blocked by another app."
              : err2.name || "Unknown error";
      getStatusLabel().textContent = "Camera error: " + reason;
      throw err2;
    }
  }

  // Extra safety for mobile: ensure correct attributes before attaching stream
  if (videoEl) {
    videoEl.setAttribute("playsinline", "true");
    videoEl.muted = true;
  }

  videoEl.srcObject = stream;

  // Try to start playback immediately (some mobile browsers need this)
  try {
    const earlyPlay = videoEl.play();
    if (earlyPlay && typeof earlyPlay.then === "function") {
      earlyPlay.catch((err) => {
        console.warn("Early video play() failed", err);
      });
    }
  } catch (e) {
    console.warn("Error calling video.play() early", e);
  }

  return new Promise((resolve, reject) => {
    videoEl.onloadedmetadata = () => {
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
      // Explicitly start video playback (needed on some browsers)
      const playPromise = videoEl.play();
      if (playPromise && typeof playPromise.then === "function") {
        playPromise
          .then(() => resolve())
          .catch((err) => {
            console.warn("Video play() failed after metadata", err);
            // Even if autoplay is blocked, resolve so the rest of the app can proceed,
            // but update status label so the user knows to tap the video.
            getStatusLabel().textContent =
              "Camera stream is ready, but video is blocked. Tap the video to start preview.";
            resolve();
          });
      } else {
        resolve();
      }
    };
    videoEl.onerror = (e) => reject(e);
  });
}

async function loadYoloModel() {
  if (yoloSession) return yoloSession;
  if (!window.ort) {
    const msg = "ONNX Runtime Web not loaded. Check script tag.";
    console.error(msg);
    getStatusLabel().textContent = msg;
    throw new Error(msg);
  }

  getStatusLabel().textContent = "Loading YOLOv8 model…";

  try {
    yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, {
      executionProviders: ["wasm"],
    });
    console.log("YOLOv8 model loaded successfully");
    return yoloSession;
  } catch (err) {
    console.error("Failed to load YOLO model:", err);
    const errorMsg = err.message || String(err);

    if (errorMsg.includes("404") || errorMsg.includes("Failed to fetch") || errorMsg.includes("not found")) {
      const msg = `YOLO model file not found at ${YOLO_MODEL_URL}. Please download yolov8n.onnx and place it in client/models/ folder. Falling back to COCO-SSD...`;
      getStatusLabel().textContent = msg;
      console.warn(msg);
      // Fallback to COCO-SSD
      return await loadCocoSsdFallback();
    } else {
      const msg = `YOLO model error: ${errorMsg}. Falling back to COCO-SSD...`;
      getStatusLabel().textContent = msg;
      console.warn(msg);
      return await loadCocoSsdFallback();
    }
  }
}

// Fallback to COCO-SSD if YOLO fails
async function loadCocoSsdFallback() {
  getStatusLabel().textContent = "Loading COCO-SSD fallback model…";

  // Load TensorFlow.js and COCO-SSD dynamically
  if (!window.tf) {
    await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js");
  }
  if (!window.cocoSsd) {
    await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js");
  }

  if (!window.cocoSsd) {
    throw new Error("Failed to load COCO-SSD fallback. Check your internet connection.");
  }

  const model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
  // Wrap COCO-SSD model to work with our detection loop
  yoloSession = {
    isCocoSsd: true,
    cocoSsdModel: model,
    inputNames: ["image"],
    outputNames: ["detections"]
  };
  return yoloSession;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

async function startApp(options = {}) {
  // Prevent duplicate starts (common on mobile with double-taps)
  if (startAppInFlight) return startAppInFlight;
  if (isRunning && videoEl?.srcObject && yoloSession) return;

  startAppInFlight = (async () => {
    isRunning = true;
    const startBtn = document.getElementById("start-btn");

    try {
      // Start camera first so the user immediately sees video,
      // then load the model (which may take a few seconds the first time).
      await initCamera();

      getStatusLabel().textContent =
        "Camera ready. Loading detection model… This may take a few seconds.";

      await loadYoloModel();

      // If we were launched via "Set Target by Voice", don't auto-prompt here.
      if (options.skipAutoTargetPrompt) {
        getStatusLabel().textContent =
          "Ready. Tap “Set Target by Voice” and say the object name.";
        if (startBtn) {
          startBtn.textContent = "Ready";
          startBtn.disabled = false;
        }
        return;
      }

      // Default flow: automatically prompt for target via voice
      getStatusLabel().textContent = "Ready. Asking for target object…";
      startBtn.textContent = "Waiting for target…";
      startBtn.disabled = true;

      // Wait a moment for camera to stabilize, then ask for target
      setTimeout(() => {
        askForTargetAutomatically();
      }, 1000);

    } catch (err) {
      console.error("Start app error:", err);
      const reason =
        err && err.name
          ? `${err.name}: ${err.message || ""}`
          : err && err.message
            ? err.message
            : "Unknown error";
      getStatusLabel().textContent = "Could not start guidance: " + reason;
      isRunning = false;
      // Re-enable button so user can try again
      if (startBtn) {
        startBtn.textContent = "Start Guidance";
        startBtn.disabled = false;
      }
      throw err;
    } finally {
      startAppInFlight = null;
    }
  })();

  return startAppInFlight;
}

// Automatically ask user for target object via voice
function askForTargetAutomatically() {
  if (!recognition) {
    // If voice recognition not available, start detection without target
    console.warn("Voice recognition not available, starting without target");
    getStatusLabel().textContent = "Voice recognition not available. Starting detection for all objects.";
    speak("Voice recognition not available. Starting detection for all objects. You can select a target from the dropdown.");
    setTimeout(() => {
      startDetection();
    }, 2000);
    return;
  }

  waitingForInitialTarget = true;
  guidanceState = GUIDANCE_STATE.ASK_TARGET;
  askTargetMode = "initial";

  // Speak the prompt
  const prompt = "What object would you like me to find? Say the name of an object";
  speak(prompt);

  // Wait for speech to finish, then start listening
  setTimeout(() => {
    try {
      getStatusLabel().textContent = "🎤 Listening for target object…";
      recognition.start();
    } catch (e) {
      console.error("Failed to start recognition:", e);
      // If recognition fails, start detection without target
      waitingForInitialTarget = false;
      getStatusLabel().textContent = "Voice recognition failed. Starting detection for all objects.";
      speak("Voice recognition failed. Starting detection for all objects.");
      setTimeout(() => {
        startDetection();
      }, 2000);
    }
  }, 3500); // Wait 3.5 seconds for speech to finish
}

// Ask for a new target while guidance is already running (camera + detection active).
function askForTargetWhileRunning() {
  if (!recognition) {
    speak("Voice target not available. Use the dropdown to change target.");
    return;
  }
  guidanceState = GUIDANCE_STATE.ASK_TARGET;
  askTargetMode = "while_running";
  const prompt = "Say the new target object, or say any for all objects.";
  speak(prompt);
  getStatusLabel().textContent = "🎤 Listening for new target…";
  setTimeout(() => {
    try {
      recognition.start();
    } catch (e) {
      console.warn("Recognition start failed", e);
      guidanceState = GUIDANCE_STATE.RUNNING;
      askTargetMode = null;
    }
  }, 2500);
}

// Double-tap gesture: STOPPED → start; RUNNING → stop or ask target; ASK_TARGET → cancel/stop.
function handleDoubleTap() {
  const now = Date.now();
  if (now - lastTapTime > DOUBLE_TAP_THRESHOLD_MS) return;
  lastTapTime = 0;

  switch (guidanceState) {
    case GUIDANCE_STATE.STOPPED:
      startApp().catch(() => {
        getStatusLabel().textContent = "Tap Start Guidance to allow camera.";
      });
      break;

    case GUIDANCE_STATE.CURRENCY_DETECTION:
      lastRunDoubleTapWasStop = true;
      stopDetection();
      break;

    case GUIDANCE_STATE.RUNNING:
      if (lastRunDoubleTapWasStop) {
        lastRunDoubleTapWasStop = false;
        askForTargetWhileRunning();
      } else {
        lastRunDoubleTapWasStop = true;
        stopDetection();
      }
      break;

    case GUIDANCE_STATE.ASK_TARGET:
      if (askTargetMode === "while_running") {
        if (recognition && isListeningForTarget) recognition.stop();
        guidanceState = GUIDANCE_STATE.RUNNING;
        askTargetMode = null;
        getStatusLabel().textContent = currentTarget
          ? `Guidance running. Looking for ${currentTarget}…`
          : "Guidance running. Detecting all objects…";
        speak("Resuming guidance.");
      } else {
        if (recognition && isListeningForTarget) recognition.stop();
        waitingForInitialTarget = false;
        stopDetection();
        askTargetMode = null;
        lastRunDoubleTapWasStop = false;
      }
      break;
  }
}

function initDoubleTapHandler() {
  const container = document.querySelector(".detect-page") || document.body;
  if (!container) return;
  container.addEventListener("click", (e) => {
    const now = Date.now();
    if (now - lastTapTime <= DOUBLE_TAP_THRESHOLD_MS) {
      handleDoubleTap();
    } else {
      lastTapTime = now;
    }
  });
}

// -------- Currency detection: voice command only, Guidance Mode only --------

function initCommandRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;
  commandRecognition = new SpeechRecognition();
  commandRecognition.lang = "en-US";
  commandRecognition.continuous = true;
  commandRecognition.interimResults = true;

  commandRecognition.onresult = (event) => {
    if (!event.results || event.results.length === 0) return;
    const result = event.results[event.results.length - 1];
    if (!result.isFinal) return;
    const transcript = (result[0] && result[0].transcript) ? result[0].transcript.toLowerCase().trim() : "";
    if (!transcript) return;
    const matched = CURRENCY_COMMAND_PHRASES.some((phrase) =>
      transcript.includes(phrase)
    );
    if (matched) handleCurrencyCommand(transcript);
  };

  commandRecognition.onerror = (event) => {
    if (event.error === "no-speech" || event.error === "aborted") return;
    console.warn("Command recognition error", event.error);
  };
}

function startCommandRecognitionIfNeeded() {
  if (!commandRecognition || commandRecognitionStarted) return;
  try {
    commandRecognition.start();
    commandRecognitionStarted = true;
  } catch (e) {
    console.warn("Command recognition start failed", e);
  }
}

function handleCurrencyCommand(transcript) {
  const now = Date.now();
  if (now - lastCurrencyCommandTime < CURRENCY_COMMAND_COOLDOWN_MS) return;
  lastCurrencyCommandTime = now;

  const inGuidanceMode =
    guidanceState === GUIDANCE_STATE.RUNNING ||
    guidanceState === GUIDANCE_STATE.CURRENCY_DETECTION;

  if (!inGuidanceMode) {
    speak("Currency detection is only available in Guidance Mode.");
    return;
  }

  if (guidanceState === GUIDANCE_STATE.CURRENCY_DETECTION) return;

  runCurrencyDetection();
}

async function runCurrencyDetection() {
  const wasRunning = guidanceState === GUIDANCE_STATE.RUNNING;
  guidanceState = GUIDANCE_STATE.CURRENCY_DETECTION;
  getStatusLabel().textContent = "Detecting currency…";

  speak("Detecting currency.");

  try {
    const result = await detectCurrencyFromFrame();
    if (guidanceState === GUIDANCE_STATE.STOPPED) return;
    guidanceState = GUIDANCE_STATE.RUNNING;
    getStatusLabel().textContent = result;
    speak(result);
  } catch (err) {
    console.error("Currency detection error", err);
    if (guidanceState !== GUIDANCE_STATE.STOPPED) {
      guidanceState = GUIDANCE_STATE.RUNNING;
      const msg = "Currency detection failed. Please try again.";
      getStatusLabel().textContent = msg;
      speak(msg);
    }
  }
}

/**
 * Analyze current video frame for currency. Replace with real model/API.
 * @returns {Promise<string>} Denomination text to announce (e.g. "20 dollar bill").
 */
async function detectCurrencyFromFrame() {
  if (!videoEl || videoEl.readyState < 2) {
    return "Camera not ready. Please try again.";
  }
  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  if (!w || !h) return "Unable to capture image.";

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, w, h);
  const imageData = ctx.getImageData(0, 0, w, h);

  // Placeholder: no real currency model. Integrate your model or API here.
  // Example: call an external API or run a local TensorFlow/ONNX currency model.
  await new Promise((r) => setTimeout(r, 800));

  return "Currency detection is not configured. Add a model or API to identify denominations.";
}

// Start the detection loop
function startDetection() {
  if (detectionStarted) return;
  detectionStarted = true;
  guidanceState = GUIDANCE_STATE.RUNNING;

  getStatusLabel().textContent = currentTarget
    ? `Guidance running. Looking for ${currentTarget}…`
    : "Guidance running. Detecting all objects…";

  const startBtn = document.getElementById("start-btn");
  if (startBtn) {
    startBtn.textContent = "Running";
    startBtn.disabled = true;
  }
  startCommandRecognitionIfNeeded();
  detectionLoop();
}

let lastDetectionTime = 0;
const DETECTION_INTERVAL_MS = 150; // tune this for performance

async function detectionLoop(timestamp) {
  if (!isRunning || !yoloSession) return;

  requestAnimationFrame(detectionLoop);

  // Ensure we have a valid timestamp even on the very first call
  if (typeof timestamp !== "number") {
    timestamp = performance.now();
  }

  if (timestamp - lastDetectionTime < DETECTION_INTERVAL_MS) {
    return;
  }
  lastDetectionTime = timestamp;

  if (videoEl.readyState < 2) return;

  try {
    const predictions = await runYoloOnFrame();
    if (predictions && Array.isArray(predictions)) {
      renderDetections(predictions);
      handleVoiceGuidance(predictions);
    }
  } catch (err) {
    console.error("Detection error", err);
    // Don't spam the status label, but log errors
    if (err.message && !err.message.includes("already running")) {
      console.warn("Detection failed:", err.message);
    }
  }
}

// -------- YOLOv8 inference via ONNX Runtime Web --------

// Offscreen canvas for 640x640 YOLO input
const yoloCanvas = document.createElement("canvas");
const yoloCtx = yoloCanvas.getContext("2d");
yoloCanvas.width = YOLO_INPUT_SIZE;
yoloCanvas.height = YOLO_INPUT_SIZE;

async function runYoloOnFrame() {
  // Check if we're using COCO-SSD fallback
  if (yoloSession && yoloSession.isCocoSsd && yoloSession.cocoSsdModel) {
    const predictions = await yoloSession.cocoSsdModel.detect(videoEl);
    // Convert COCO-SSD format to our format
    return predictions.map(pred => ({
      bbox: pred.bbox, // [x, y, width, height]
      score: pred.score,
      class: pred.class
    }));
  }

  // YOLO inference path
  // Draw current video frame to YOLO input size
  yoloCtx.drawImage(videoEl, 0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const imageData = yoloCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const { data } = imageData;

  // Convert RGBA to CHW float32 normalized [0,1]
  const area = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  const floatData = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    floatData[i] = r;
    floatData[i + area] = g;
    floatData[i + 2 * area] = b;
  }

  const inputTensor = new ort.Tensor("float32", floatData, [
    1,
    3,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
  ]);

  const inputName = yoloSession.inputNames[0];
  const feeds = { [inputName]: inputTensor };

  const results = await yoloSession.run(feeds);
  const outputName = yoloSession.outputNames[0];
  const outputTensor = results[outputName];
  const outputData = outputTensor.data;
  const dims = outputTensor.dims;

  // YOLOv8 ONNX typical shapes: [1, 84, 8400] or [1, 8400, 84]
  let numDetections;
  let layout; // "channels_first" or "channels_last"
  if (dims.length === 3 && dims[1] === 84) {
    // [1, 84, N]
    layout = "channels_first";
    numDetections = dims[2];
  } else if (dims.length === 3 && dims[2] === 84) {
    // [1, N, 84]
    layout = "channels_last";
    numDetections = dims[1];
  } else {
    console.warn("Unexpected YOLO output dims", dims);
    return [];
  }

  const imgW = canvasEl.width;
  const imgH = canvasEl.height;
  const numClasses = COCO_CLASSES.length;

  const CONF_THRESHOLD = Math.max(confidenceThreshold, 0.3);
  const IOU_THRESHOLD = 0.45;

  const detections = [];

  if (layout === "channels_first") {
    const N = numDetections;
    for (let i = 0; i < N; i++) {
      const cx = outputData[0 * N + i];
      const cy = outputData[1 * N + i];
      const w = outputData[2 * N + i];
      const h = outputData[3 * N + i];
      const objConf = outputData[4 * N + i];
      if (objConf < CONF_THRESHOLD) continue;

      let bestClass = -1;
      let bestScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const classScore = outputData[(5 + c) * N + i];
        if (classScore > bestScore) {
          bestScore = classScore;
          bestClass = c;
        }
      }
      const score = objConf * bestScore;
      if (score < CONF_THRESHOLD || bestClass < 0) continue;

      const left = ((cx - w / 2) / YOLO_INPUT_SIZE) * imgW;
      const top = ((cy - h / 2) / YOLO_INPUT_SIZE) * imgH;
      const width = (w / YOLO_INPUT_SIZE) * imgW;
      const height = (h / YOLO_INPUT_SIZE) * imgH;

      detections.push({
        bbox: [left, top, width, height],
        score,
        class: COCO_CLASSES[bestClass] || `class_${bestClass}`,
      });
    }
  } else {
    // channels_last: [1, N, 84]
    const stride = 84;
    for (let i = 0; i < numDetections; i++) {
      const offset = i * stride;
      const cx = outputData[offset + 0];
      const cy = outputData[offset + 1];
      const w = outputData[offset + 2];
      const h = outputData[offset + 3];
      const objConf = outputData[offset + 4];
      if (objConf < CONF_THRESHOLD) continue;

      let bestClass = -1;
      let bestScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const classScore = outputData[offset + 5 + c];
        if (classScore > bestScore) {
          bestScore = classScore;
          bestClass = c;
        }
      }
      const score = objConf * bestScore;
      if (score < CONF_THRESHOLD || bestClass < 0) continue;

      const left = ((cx - w / 2) / YOLO_INPUT_SIZE) * imgW;
      const top = ((cy - h / 2) / YOLO_INPUT_SIZE) * imgH;
      const width = (w / YOLO_INPUT_SIZE) * imgW;
      const height = (h / YOLO_INPUT_SIZE) * imgH;

      detections.push({
        bbox: [left, top, width, height],
        score,
        class: COCO_CLASSES[bestClass] || `class_${bestClass}`,
      });
    }
  }

  // NMS
  detections.sort((a, b) => b.score - a.score);
  const finalDetections = [];
  const used = new Array(detections.length).fill(false);

  for (let i = 0; i < detections.length; i++) {
    if (used[i]) continue;
    const a = detections[i];
    finalDetections.push(a);
    used[i] = true;

    for (let j = i + 1; j < detections.length; j++) {
      if (used[j]) continue;
      const b = detections[j];
      if (iou(a.bbox, b.bbox) > IOU_THRESHOLD) {
        used[j] = true;
      }
    }
  }

  return finalDetections;
}

function iou(boxA, boxB) {
  const [ax, ay, aw, ah] = boxA;
  const [bx, by, bw, bh] = boxB;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - inter;
  return union <= 0 ? 0 : inter / union;
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

// Estimate approximate steps to object using bbox height and simple pinhole geometry.
function estimateSteps(pred) {
  if (!pred || !pred.bbox) return null;
  const [x, y, width, height] = pred.bbox;
  if (height < MIN_BOX_HEIGHT_PX) return null;

  // Use relative height in the frame as a robust proxy for distance.
  const frameHeight = canvasEl && canvasEl.height ? canvasEl.height : videoEl?.videoHeight;
  if (!frameHeight || frameHeight <= 0) return null;

  const relativeHeight = height / frameHeight; // 0..1
  if (relativeHeight <= 0) return null;

  // Heuristic: more steps when box is small, fewer when big.
  // baseSteps ~ 1 when object fills most of the frame, grows as it gets smaller.
  const baseSteps = 1 / relativeHeight;
  let steps = baseSteps * STEP_SCALE_FACTOR;

  if (!isFinite(steps) || steps <= 0) return null;

  // Keep steps in a reasonable range
  steps = Math.max(1, Math.min(20, steps));

  return steps;
}

function getSpatialDescriptor(pred) {
  const [x, y, width, height] = pred.bbox;
  const centerX = x + width / 2;
  const frameWidth = canvasEl.width;
  const frameHeight = canvasEl.height;

  const normalizedCenterX = centerX / frameWidth;
  let horizontal;
  // Make left/right zones a bit wider so guidance is more decisive.
  if (normalizedCenterX < 0.4) horizontal = "left";
  else if (normalizedCenterX > 0.6) horizontal = "right";
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
  if (guidanceState === GUIDANCE_STATE.CURRENCY_DETECTION) return;

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

  // --- Step-based guidance logic ---
  const estimatedSteps = estimateSteps(best);

  // Fallback to old behavior if we cannot estimate steps reliably
  if (estimatedSteps == null) {
    const keyFallback = `${best.class}-${spatial.horizontal}-${spatial.distance}`;
    const lastFallback = lastSpoken[keyFallback];
    if (lastFallback && now - lastFallback.timestamp < SPEAK_COOLDOWN_MS) {
      return;
    }
    const fallbackPhrase = currentTarget
      ? buildTargetPhrase(best, spatial)
      : buildPhrase(best, spatial);
    speak(fallbackPhrase);
    lastSpoken[keyFallback] = { timestamp: now };
    lastSpeakTime = now;
    getStatusLabel().textContent = fallbackPhrase;
    return;
  }

  const roundedSteps = Math.max(1, Math.round(estimatedSteps));

  // If object is extremely close (1–2 steps), say "Target found" and stop guidance.
  if (roundedSteps <= TARGET_FOUND_MAX_STEPS) {
    const foundText = currentTarget
      ? `Target ${best.class} found. It is within ${roundedSteps} step${roundedSteps === 1 ? "" : "s"
      }. Stopping guidance.`
      : `Object ${best.class} found. It is within ${roundedSteps} step${roundedSteps === 1 ? "" : "s"
      }. Stopping guidance.`;

    speak(foundText);
    getStatusLabel().textContent = foundText;
    if ("vibrate" in navigator) {
      navigator.vibrate([100, 60, 100]);
    }

    // Stop detection loop
    stopDetection();
    lastSpeakTime = now;
    return;
  }

  const sideText =
    spatial.horizontal === "center"
      ? "ahead"
      : spatial.horizontal === "left"
        ? "to your left"
        : "to your right";

  const mapKey = `${best.class}-${spatial.horizontal}`;
  const lastInfo = lastStepsMap[mapKey];

  // Only speak if step count actually changed by at least threshold
  if (lastInfo && Math.abs(roundedSteps - lastInfo.steps) < STEP_CHANGE_THRESHOLD) {
    return;
  }

  // Also respect per-object cooldown
  const keyCooldown = `${mapKey}-${roundedSteps}`;
  const lastSpokenEntry = lastSpoken[keyCooldown];
  if (lastSpokenEntry && now - lastSpokenEntry.timestamp < SPEAK_COOLDOWN_MS) {
    return;
  }

  let trendText = "";
  if (lastInfo) {
    if (roundedSteps < lastInfo.steps) trendText = "getting closer";
    else if (roundedSteps > lastInfo.steps) trendText = "getting farther";
  }

  let phrase;
  if (currentTarget) {
    phrase = `Target ${best.class}, about ${roundedSteps} step${roundedSteps === 1 ? "" : "s"
      } ${sideText}${trendText ? `, ${trendText}` : ""}.`;
  } else {
    phrase = `${capitalize(best.class)}, about ${roundedSteps} step${roundedSteps === 1 ? "" : "s"
      } ${sideText}${trendText ? `, ${trendText}` : ""}.`;
  }

  speak(phrase);
  lastSpeakTime = now;
  lastStepsMap[mapKey] = { steps: roundedSteps, timestamp: now };
  lastSpoken[keyCooldown] = { timestamp: now };

  if (currentTarget && best.class === currentTarget && "vibrate" in navigator) {
    navigator.vibrate([80, 40, 80]);
  }

  getStatusLabel().textContent = phrase;
}

function stopDetection() {
  isRunning = false;
  detectionStarted = false;
  guidanceState = GUIDANCE_STATE.STOPPED;
  const startBtn = document.getElementById("start-btn");
  if (startBtn) {
    startBtn.textContent = "Start Guidance";
    startBtn.disabled = false;
  }
  // Stop camera stream if any
  if (videoEl && videoEl.srcObject) {
    const tracks = videoEl.srcObject.getTracks();
    tracks.forEach((t) => t.stop && t.stop());
    videoEl.srcObject = null;
  }
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

// -------- Voice input for target object (Web Speech API) --------

function initSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn("SpeechRecognition not supported in this browser.");
    const hint = document.getElementById("voice-target-hint");
    if (hint) {
      hint.textContent =
        "Voice target is not supported in this browser. Please use Chrome or Edge.";
    }
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    isListeningForTarget = true;
    const btn = document.getElementById("voice-target-btn");
    if (btn) {
      btn.textContent = "🎤 Listening…";
      btn.disabled = true;
    }
    getStatusLabel().textContent = "Listening for target object…";
  };

  recognition.onerror = (event) => {
    console.warn("Speech recognition error", event.error);

    // If we're waiting for initial target and get an error, start detection without target
    if (waitingForInitialTarget) {
      if (event.error === "no-speech" || event.error === "audio-capture") {
        // User didn't speak or mic not available - start detection anyway
        waitingForInitialTarget = false;
        speak("Starting detection for all objects. You can set a target later using the dropdown or voice button.");
        setTimeout(() => {
          startDetection();
        }, 2000);
        return;
      }
    }

    getStatusLabel().textContent =
      "Voice error: " + (event.error || "unknown error");
  };

  recognition.onresult = (event) => {
    if (!event.results || !event.results[0]) return;
    const transcript = event.results[0][0].transcript.toLowerCase().trim();
    const isAskWhileRunning = guidanceState === GUIDANCE_STATE.ASK_TARGET && askTargetMode === "while_running";
    handleTargetFromSpeech(transcript, waitingForInitialTarget, isAskWhileRunning);
  };

  recognition.onend = () => {
    isListeningForTarget = false;
    const btn = document.getElementById("voice-target-btn");
    if (btn) {
      btn.textContent = "🎤 Set Target by Voice";
      btn.disabled = false;
    }

    // If we were waiting for initial target and didn't get a valid one, retry
    if (waitingForInitialTarget && !currentTarget) {
      setTimeout(() => {
        if (waitingForInitialTarget && !currentTarget) {
          speak("I didn't catch that. Please say an object name like person, chair, bottle, or any.");
          setTimeout(() => {
            try {
              recognition.start();
            } catch (e) {
              console.error("Failed to restart recognition:", e);
              waitingForInitialTarget = false;
              startDetection();
            }
          }, 2000);
        }
      }, 2000);
    }
  };
}

async function toggleVoiceTargetListening() {
  if (!recognition) {
    getStatusLabel().textContent =
      "Voice target not available. Use the dropdown instead.";
    return;
  }

  if (isListeningForTarget) {
    recognition.stop();
  } else {
    try {
      // If user taps "Set Target by Voice" first, bootstrap camera + model now.
      if (!isRunning || !videoEl?.srcObject || !yoloSession) {
        const btn = document.getElementById("voice-target-btn");
        if (btn) {
          btn.disabled = true;
          btn.textContent = "Starting…";
        }
        await startApp({ skipAutoTargetPrompt: true });
        if (btn) {
          btn.disabled = false;
          btn.textContent = "🎤 Set Target by Voice";
        }
      }

      getStatusLabel().textContent = "🎤 Listening for target object…";
      recognition.start();
    } catch (e) {
      console.error("Failed to start recognition", e);
      getStatusLabel().textContent = "Voice recognition error. Use the dropdown instead.";
    }
  }
}

function handleTargetFromSpeech(transcript, isInitialPrompt = false, isAskWhileRunning = false) {
  if (CURRENCY_COMMAND_PHRASES.some((phrase) => transcript.includes(phrase))) return;
  // Basic normalization and mapping to model class names
  const norm = transcript
    .replace(/[^a-zA-Z ]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const words = norm.split(" ");

  // Comprehensive mapping of voice commands to COCO class names
  // Includes synonyms and common variations
  const voiceToClassMap = {
    // Clear target
    "any": "any",
    "all": "any",
    "everything": "any",
    "clear": "any",

    // People & Animals
    "person": "person",
    "people": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "cat": "cat",
    "kitten": "cat",
    "dog": "dog",
    "puppy": "dog",
    "bird": "bird",
    "horse": "horse",
    "cow": "cow",
    "sheep": "sheep",

    // Vehicles
    "car": "car",
    "vehicle": "car",
    "automobile": "car",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "train": "train",
    "boat": "boat",
    "ship": "boat",
    "airplane": "airplane",
    "plane": "airplane",
    "aircraft": "airplane",

    // Furniture
    "chair": "chair",
    "seat": "chair",
    "couch": "couch",
    "sofa": "couch",
    "bed": "bed",
    "dining table": "dining table",
    "table": "dining table",
    "bench": "bench",

    // Electronics
    "phone": "cell phone",
    "mobile": "cell phone",
    "cell phone": "cell phone",
    "smartphone": "cell phone",
    "laptop": "laptop",
    "computer": "laptop",
    "tv": "tv",
    "television": "tv",
    "remote": "remote",
    "remote control": "remote",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "clock": "clock",
    "watch": "clock",

    // Kitchen Items
    "bottle": "bottle",
    "cup": "cup",
    "mug": "cup",
    "bowl": "bowl",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "microwave": "microwave",
    "oven": "oven",
    "toaster": "toaster",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "sink": "sink",

    // Food & Drinks
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "sandwich": "sandwich",
    "pizza": "pizza",
    "hot dog": "hot dog",
    "cake": "cake",
    "donut": "donut",
    "doughnut": "donut",
    "wine glass": "wine glass",
    "wine": "wine glass",

    // Personal Items
    "backpack": "backpack",
    "bag": "backpack",
    "handbag": "handbag",
    "purse": "handbag",
    "suitcase": "suitcase",
    "luggage": "suitcase",
    "umbrella": "umbrella",
    "tie": "tie",
    "book": "book",
    "scissors": "scissors",
    "toothbrush": "toothbrush",
    "hair drier": "hair drier",
    "hair dryer": "hair drier",
    "hairdryer": "hair drier",

    // Sports & Recreation
    "sports ball": "sports ball",
    "ball": "sports ball",
    "baseball bat": "baseball bat",
    "bat": "baseball bat",
    "baseball glove": "baseball glove",
    "glove": "baseball glove",
    "tennis racket": "tennis racket",
    "racket": "tennis racket",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "skis": "skis",
    "ski": "skis",
    "snowboard": "snowboard",
    "kite": "kite",
    "frisbee": "frisbee",

    // Other Objects
    "door": "door",
    "traffic light": "traffic light",
    "stop sign": "stop sign",
    "sign": "stop sign",
    "fire hydrant": "fire hydrant",
    "hydrant": "fire hydrant",
    "parking meter": "parking meter",
    "meter": "parking meter",
    "potted plant": "potted plant",
    "plant": "potted plant",
    "toilet": "toilet",
    "vase": "vase",
    "teddy bear": "teddy bear",
    "teddy": "teddy bear",
    "bear": "teddy bear",
  };

  // Try to match the transcript
  let matched = "";
  const lowerNorm = norm.toLowerCase();

  // First, try exact match
  if (voiceToClassMap[lowerNorm]) {
    matched = voiceToClassMap[lowerNorm];
  } else {
    // Try word-by-word matching
    for (const [key, value] of Object.entries(voiceToClassMap)) {
      const keyWords = key.split(" ");
      if (keyWords.length === 1) {
        if (words.includes(key)) {
          matched = value;
          break;
        }
      } else {
        // Multi-word match
        let allWordsMatch = true;
        for (const kw of keyWords) {
          if (!lowerNorm.includes(kw)) {
            allWordsMatch = false;
            break;
          }
        }
        if (allWordsMatch) {
          matched = value;
          break;
        }
      }
    }
  }

  if (!matched) {
    if (isAskWhileRunning) {
      getStatusLabel().textContent = "Did not recognize. Say person, chair, bottle, or any.";
      speak("Did not recognize. Try again.");
      return;
    }
    getStatusLabel().textContent =
      'Did not recognize a target. Try saying "person", "chair", "bottle", "car", "phone", or "any".';
    if (isInitialPrompt) {
      speak("I did not catch a valid target. Please try again.");
      return;
    }
    speak("I did not catch a valid target. Please try again.");
    return;
  }

  // Ask-while-running: only update target and state; detection already running.
  if (isAskWhileRunning) {
    if (matched === "any") {
      currentTarget = "";
      const sel = document.getElementById("target-select");
      if (sel) sel.value = "";
      speak("Target set to any. Detecting all objects.");
    } else {
      currentTarget = matched;
      const sel = document.getElementById("target-select");
      if (sel) {
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === matched) {
            sel.selectedIndex = i;
            break;
          }
        }
      }
      speak(`Target set to ${matched}.`);
    }
    getStatusLabel().textContent = currentTarget
      ? `Guidance running. Looking for ${currentTarget}…`
      : "Guidance running. Detecting all objects…";
    guidanceState = GUIDANCE_STATE.RUNNING;
    lastRunDoubleTapWasStop = false;
    askTargetMode = null;
    return;
  }

  if (matched === "any") {
    currentTarget = "";
    const select = document.getElementById("target-select");
    if (select) select.value = "";
    const msg = "Target set to any. Detecting all objects.";
    getStatusLabel().textContent = msg;
    speak(msg);

    // If this was the initial prompt, start detection now
    if (isInitialPrompt) {
      waitingForInitialTarget = false;
      setTimeout(() => {
        startDetection();
      }, 1500);
    } else if (!detectionStarted && isRunning) {
      // If user used "Set Target by Voice" and guidance isn't running yet,
      // start detection automatically after confirming the target.
      setTimeout(() => {
        startDetection();
      }, 1500);
    }
    return;
  }

  currentTarget = matched;
  const select = document.getElementById("target-select");
  if (select) {
    // Find the option with matching value
    for (let i = 0; i < select.options.length; i++) {
      if (select.options[i].value === matched) {
        select.selectedIndex = i;
        break;
      }
    }
  }

  const msg = `Target set to ${matched}. Starting detection now.`;
  getStatusLabel().textContent = msg;
  speak(msg);

  // If this was the initial prompt, start detection now
  if (isInitialPrompt) {
    waitingForInitialTarget = false;
    setTimeout(() => {
      startDetection();
    }, 2000); // Wait for confirmation speech to finish
  } else if (!detectionStarted && isRunning) {
    // If user clicked "Set Target by Voice" after Start Guidance
    // and detection hasn't begun yet, start it automatically.
    setTimeout(() => {
      startDetection();
    }, 2000);
  }
}

// ================================
// CURRENCY BUTTON - HOLD TO DETECT, TAP TO START GUIDANCE
// ================================

let currencyHoldTimer = null;
let currencyHoldStartTime = 0;
const CURRENCY_HOLD_DURATION_MS = 1500; // 1.5 seconds to trigger currency detection
let currencyTapHandled = false;

function initCurrencyButton() {
  const currencyBtn = document.getElementById("currency-detect-btn");
  if (!currencyBtn) return;

  // Add progress bar element for hold indication
  let progressBar = currencyBtn.querySelector(".hold-progress");
  if (!progressBar) {
    progressBar = document.createElement("div");
    progressBar.className = "hold-progress";
    currencyBtn.appendChild(progressBar);
  }

  // Touch events for mobile
  currencyBtn.addEventListener("touchstart", (e) => {
    e.preventDefault();
    currencyTapHandled = false;
    startCurrencyHold(currencyBtn, progressBar);
  }, { passive: false });

  currencyBtn.addEventListener("touchend", (e) => {
    e.preventDefault();
    endCurrencyHold(currencyBtn, progressBar);
  });

  currencyBtn.addEventListener("touchcancel", (e) => {
    cancelCurrencyHold(currencyBtn, progressBar);
  });

  // Mouse events for desktop
  currencyBtn.addEventListener("mousedown", (e) => {
    currencyTapHandled = false;
    startCurrencyHold(currencyBtn, progressBar);
  });

  currencyBtn.addEventListener("mouseup", (e) => {
    endCurrencyHold(currencyBtn, progressBar);
  });

  currencyBtn.addEventListener("mouseleave", (e) => {
    cancelCurrencyHold(currencyBtn, progressBar);
  });
}

function startCurrencyHold(btn, progressBar) {
  currencyHoldStartTime = Date.now();
  btn.classList.add("pressing");

  // Animate progress bar
  progressBar.style.transition = `width ${CURRENCY_HOLD_DURATION_MS}ms linear`;
  progressBar.style.width = "100%";

  // Vibrate feedback if supported
  if (navigator.vibrate) {
    navigator.vibrate(50);
  }

  // Set timer for hold action (currency detection)
  currencyHoldTimer = setTimeout(() => {
    currencyTapHandled = true;
    btn.classList.remove("pressing");
    progressBar.style.transition = "width 0.2s ease";
    progressBar.style.width = "0";

    // Vibrate to confirm hold completed
    if (navigator.vibrate) {
      navigator.vibrate([100, 50, 100]);
    }

    // Trigger currency detection
    triggerCurrencyDetection();
  }, CURRENCY_HOLD_DURATION_MS);
}

function endCurrencyHold(btn, progressBar) {
  const holdDuration = Date.now() - currencyHoldStartTime;

  // Clear the hold timer
  if (currencyHoldTimer) {
    clearTimeout(currencyHoldTimer);
    currencyHoldTimer = null;
  }

  // Reset UI
  btn.classList.remove("pressing");
  progressBar.style.transition = "width 0.2s ease";
  progressBar.style.width = "0";

  // If hold wasn't completed (it was a tap), start guidance mode
  if (!currencyTapHandled && holdDuration < CURRENCY_HOLD_DURATION_MS) {
    startGuidanceMode();
  }
}

function cancelCurrencyHold(btn, progressBar) {
  if (currencyHoldTimer) {
    clearTimeout(currencyHoldTimer);
    currencyHoldTimer = null;
  }

  btn.classList.remove("pressing");
  progressBar.style.transition = "width 0.2s ease";
  progressBar.style.width = "0";
}

function startGuidanceMode() {
  // Start the main guidance/detection mode
  if (!isRunning || !yoloSession) {
    speak("Starting guidance mode.");
    getStatusLabel().textContent = "Starting guidance...";
    startApp().catch((err) => {
      speak("Could not start guidance. Please try again.");
      getStatusLabel().textContent = "Failed to start. Tap to retry.";
    });
  } else {
    // Already running, ask for target
    speak("Guidance is running. Double tap to change target.");
    getStatusLabel().textContent = currentTarget
      ? `Looking for ${currentTarget}...`
      : "Detecting all objects...";
  }
}

function triggerCurrencyDetection() {
  // Ensure camera and model are ready
  if (!isRunning || !yoloSession) {
    speak("Starting camera for currency detection.");
    getStatusLabel().textContent = "Starting camera...";
    startApp().then(() => {
      setTimeout(() => {
        runCurrencyDetectionFromButton();
      }, 1000);
    }).catch((err) => {
      speak("Camera failed. Please try again.");
      getStatusLabel().textContent = "Camera error. Tap to retry.";
    });
    return;
  }

  runCurrencyDetectionFromButton();
}

async function runCurrencyDetectionFromButton() {
  const btn = document.getElementById("currency-detect-btn");
  if (btn) btn.classList.add("active");

  guidanceState = GUIDANCE_STATE.CURRENCY_DETECTION;
  getStatusLabel().textContent = "📸 Detecting currency...";
  speak("Detecting currency. Please hold the note steady.");

  try {
    const result = await detectCurrencyWithModel();
    if (guidanceState === GUIDANCE_STATE.STOPPED) return;

    guidanceState = GUIDANCE_STATE.RUNNING;
    getStatusLabel().textContent = result;
    speak(result);
  } catch (err) {
    console.error("Currency detection error:", err);
    if (guidanceState !== GUIDANCE_STATE.STOPPED) {
      guidanceState = GUIDANCE_STATE.RUNNING;
      const msg = "Currency detection failed. Please try again.";
      getStatusLabel().textContent = msg;
      speak(msg);
    }
  } finally {
    if (btn) btn.classList.remove("active");
  }
}

// Currency detection using ONNX model from models folder
let currencySession = null;
const CURRENCY_MODEL_URL = "models/currency_detector.onnx";
const CURRENCY_INPUT_SIZE = 150; // Model expects 150x150 input

async function loadCurrencyModel() {
  if (currencySession) return currencySession;

  if (!window.ort) {
    throw new Error("ONNX Runtime Web not loaded.");
  }

  getStatusLabel().textContent = "Loading currency detection model...";

  try {
    currencySession = await ort.InferenceSession.create(CURRENCY_MODEL_URL, {
      executionProviders: ["wasm"],
    });
    console.log("Currency detection model loaded successfully");
    console.log("Model inputs:", currencySession.inputNames);
    console.log("Model outputs:", currencySession.outputNames);
    return currencySession;
  } catch (err) {
    console.error("Failed to load currency model:", err);
    throw err;
  }
}

async function detectCurrencyWithModel() {
  if (!videoEl || videoEl.readyState < 2) {
    return "Camera not ready. Please try again.";
  }

  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  if (!w || !h) return "Unable to capture image.";

  try {
    // Load the currency model if not already loaded
    const session = await loadCurrencyModel();

    // Create canvas for preprocessing
    const canvas = document.createElement("canvas");
    canvas.width = CURRENCY_INPUT_SIZE;
    canvas.height = CURRENCY_INPUT_SIZE;
    const captureCtx = canvas.getContext("2d");

    // Draw and resize the video frame
    captureCtx.drawImage(videoEl, 0, 0, CURRENCY_INPUT_SIZE, CURRENCY_INPUT_SIZE);
    const imageData = captureCtx.getImageData(0, 0, CURRENCY_INPUT_SIZE, CURRENCY_INPUT_SIZE);
    const { data } = imageData;

    // Preprocess: Convert to HWC format (channels last), normalize to [0,1]
    // Model expects shape [1, 150, 150, 3] (NHWC format)
    const area = CURRENCY_INPUT_SIZE * CURRENCY_INPUT_SIZE;
    const floatData = new Float32Array(area * 3);

    for (let i = 0; i < area; i++) {
      floatData[i * 3] = data[i * 4] / 255.0;       // R
      floatData[i * 3 + 1] = data[i * 4 + 1] / 255.0; // G
      floatData[i * 3 + 2] = data[i * 4 + 2] / 255.0; // B
    }

    // Create input tensor - shape [1, H, W, 3] (NHWC format)
    const inputTensor = new ort.Tensor("float32", floatData, [1, CURRENCY_INPUT_SIZE, CURRENCY_INPUT_SIZE, 3]);

    const inputName = session.inputNames[0];
    const feeds = { [inputName]: inputTensor };

    // Run inference
    getStatusLabel().textContent = "Analyzing currency...";
    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];
    const outputData = outputTensor.data;

    // Class labels - adjust based on your model
    const classLabels = [
      "10 Rupees", "20 Rupees", "50 Rupees", "100 Rupees",
      "200 Rupees", "500 Rupees", "2000 Rupees", "Not a currency"
    ];

    // Find highest probability class
    let maxIdx = 0;
    let maxProb = outputData[0];
    for (let i = 1; i < outputData.length; i++) {
      if (outputData[i] > maxProb) {
        maxProb = outputData[i];
        maxIdx = i;
      }
    }

    const confidence = maxProb;
    if (confidence < 0.5) {
      return "Could not identify currency. Please hold the note closer.";
    }

    const detectedCurrency = classLabels[maxIdx] || `Class ${maxIdx}`;
    return `Detected: ${detectedCurrency} (${Math.round(confidence * 100)}% confidence)`;

  } catch (err) {
    console.error("Currency detection error:", err);
    if (err.message && err.message.includes("fetch")) {
      return "Currency model not found. Check models folder.";
    }
    return `Error: ${err.message || "Unknown error"}`;
  }
}

// ================================
// EMERGENCY MODE
// ================================

let emergencyActive = false;
let emergencyHoldTimer = null;
let emergencyAlarmAudio = null;
let emergencyHoldStartTime = 0;
const EMERGENCY_HOLD_DURATION_MS = 2000; // 2 seconds to activate

function initEmergencyButton() {
  const emergencyBtn = document.getElementById("emergency-btn");
  if (!emergencyBtn) return;

  // Add progress bar element
  const progressBar = document.createElement("div");
  progressBar.className = "hold-progress";
  emergencyBtn.appendChild(progressBar);

  // Touch events for mobile
  emergencyBtn.addEventListener("touchstart", (e) => {
    e.preventDefault();
    startEmergencyHold(emergencyBtn, progressBar);
  }, { passive: false });

  emergencyBtn.addEventListener("touchend", (e) => {
    e.preventDefault();
    cancelEmergencyHold(emergencyBtn, progressBar);
  });

  emergencyBtn.addEventListener("touchcancel", (e) => {
    cancelEmergencyHold(emergencyBtn, progressBar);
  });

  // Mouse events for desktop
  emergencyBtn.addEventListener("mousedown", (e) => {
    startEmergencyHold(emergencyBtn, progressBar);
  });

  emergencyBtn.addEventListener("mouseup", (e) => {
    cancelEmergencyHold(emergencyBtn, progressBar);
  });

  emergencyBtn.addEventListener("mouseleave", (e) => {
    cancelEmergencyHold(emergencyBtn, progressBar);
  });
}

function startEmergencyHold(btn, progressBar) {
  if (emergencyActive) return;

  emergencyHoldStartTime = Date.now();
  btn.classList.add("pressing");

  // Animate progress bar
  progressBar.style.transition = `width ${EMERGENCY_HOLD_DURATION_MS}ms linear`;
  progressBar.style.width = "100%";

  // Vibrate if supported (pattern: hold vibration)
  if (navigator.vibrate) {
    navigator.vibrate(100);
  }

  emergencyHoldTimer = setTimeout(() => {
    activateEmergency();
  }, EMERGENCY_HOLD_DURATION_MS);
}

function cancelEmergencyHold(btn, progressBar) {
  if (emergencyHoldTimer) {
    clearTimeout(emergencyHoldTimer);
    emergencyHoldTimer = null;
  }

  btn.classList.remove("pressing");
  progressBar.style.transition = "width 0.2s ease";
  progressBar.style.width = "0";
}

function activateEmergency() {
  if (emergencyActive) return;
  emergencyActive = true;

  const overlay = document.getElementById("emergency-overlay");
  if (overlay) {
    overlay.setAttribute("aria-hidden", "false");
  }

  document.body.classList.add("emergency-active");

  // Strong vibration pattern for emergency
  if (navigator.vibrate) {
    // SOS pattern: 3 short, 3 long, 3 short
    navigator.vibrate([100, 50, 100, 50, 100, 200, 300, 100, 300, 100, 300, 200, 100, 50, 100, 50, 100]);
  }

  // Play alarm sound
  playEmergencyAlarm();

  // Speak emergency message
  speak("Emergency alert activated! Help is on the way. Notifying your emergency contacts.");

  // Get and display location
  fetchEmergencyLocation();

  // Simulate notifying contacts (in a real app, this would send SMS/push notifications)
  console.log("🚨 EMERGENCY ACTIVATED - Would notify emergency contacts here");
}

function cancelEmergency() {
  if (!emergencyActive) return;
  emergencyActive = false;

  const overlay = document.getElementById("emergency-overlay");
  if (overlay) {
    overlay.setAttribute("aria-hidden", "true");
  }

  document.body.classList.remove("emergency-active");

  // Stop vibration
  if (navigator.vibrate) {
    navigator.vibrate(0);
  }

  // Stop alarm
  stopEmergencyAlarm();

  speak("Emergency cancelled.");
  getStatusLabel().textContent = "Emergency cancelled.";

  // Reset button state
  const btn = document.getElementById("emergency-btn");
  const progressBar = btn?.querySelector(".hold-progress");
  if (btn && progressBar) {
    btn.classList.remove("pressing");
    progressBar.style.width = "0";
  }
}

function playEmergencyAlarm() {
  try {
    // Create alarm using Web Audio API
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();

    function playAlarmLoop() {
      if (!emergencyActive) return;

      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      // Siren-like sound
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
      oscillator.frequency.exponentialRampToValueAtTime(1200, audioContext.currentTime + 0.5);
      oscillator.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 1);

      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 1);

      // Loop the alarm
      setTimeout(() => {
        if (emergencyActive) playAlarmLoop();
      }, 1100);
    }

    playAlarmLoop();
    emergencyAlarmAudio = audioContext;

  } catch (err) {
    console.error("Could not play alarm:", err);
  }
}

function stopEmergencyAlarm() {
  if (emergencyAlarmAudio) {
    try {
      emergencyAlarmAudio.close();
    } catch (e) {
      // Ignore errors on close
    }
    emergencyAlarmAudio = null;
  }
}

function fetchEmergencyLocation() {
  const locationText = document.getElementById("emergency-location-text");
  if (!locationText) return;

  if (!navigator.geolocation) {
    locationText.textContent = "Location unavailable";
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (position) => {
      const { latitude, longitude } = position.coords;
      locationText.textContent = `Lat: ${latitude.toFixed(5)}, Lng: ${longitude.toFixed(5)}`;

      // In a real app, you would:
      // 1. Reverse geocode to get address
      // 2. Send this location to emergency contacts
      // 3. Possibly send to emergency services API
      console.log("Emergency location:", { latitude, longitude });
    },
    (error) => {
      console.error("Location error:", error);
      locationText.textContent = "Could not fetch location";
    },
    {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 0
    }
  );
}
