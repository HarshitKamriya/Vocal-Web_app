## Vocal Path – Full-Stack Structure

This project is now split into a **frontend client** and a **backend server**:

- `client/` – Pure browser app (HTML/CSS/JS) that:
  - Opens the **phone camera** (rear preferred).
  - Runs **real-time object detection** fully in the browser using **TensorFlow.js + COCO-SSD**.
  - Draws **bounding boxes and labels** over the live video.
  - Provides **spoken guidance** (“Person in front, near.”) using the **Web Speech API**.
  - Adds **spatial hints**: Left / Center / Right + Near / Far.
  - Supports **target object mode**, **vibration on target**, and a **confidence slider**.

- `server/` – Minimal **Node/Express** backend that:
  - Serves the static files from `client/`.
  - Exposes a health route at `/api/health`.
  - Can be extended with more APIs (e.g., analytics, user profiles) without touching the client.

### Client Files (`client/`)

- `index.html` – Layout, camera/video elements, controls, and script imports.
- `style.css` – High-contrast, mobile-first, accessible styles.
- `app.js` – Camera access, detection pipeline, drawing, voice guidance, and controls.

### Server Files (`server/`)

- `server.js` – Express server that serves `client/` and exposes `/api/health`.
- `package.json` – Node dependencies and scripts.

### How to Run (Recommended: via backend)

1. Open a terminal in the project root and install server dependencies:
   - `cd server`
   - `npm install`
2. Start the backend (which serves the frontend):
   - `npm start`
3. Open `http://localhost:4000` in your desktop or mobile browser (same network).
4. On mobile, for best experience:
   - Use Chrome or Edge on Android, or Safari on iOS.
   - Allow **camera access** when prompted.
   - Tap **“Start Guidance”** if auto-start is blocked by the browser.

### Detection & Voice Flow (Client)

1. **Camera**: `navigator.mediaDevices.getUserMedia` with `facingMode: "environment"` tries to use the **rear camera**, falling back to any camera if needed.
2. **Model**: `cocoSsd.load({ base: "lite_mobilenet_v2" })` loads a light, fast COCO-SSD model.
3. **Loop**:
   - Every ~150 ms, run `model.detect(video)` on the live video element.
   - Filter predictions by **confidence** and optional **target object**.
   - Draw bounding boxes and labels onto the canvas overlay.
   - Compute **Left / Center / Right** and **Near / Far** from the bounding boxes.
   - Decide if and what to speak:
     - Avoid repeating the same object-position combo for at least 4 seconds.
     - Enforce a global minimum gap between voice messages.
4. **Voice**: Uses `speechSynthesis.speak()` for clear, slightly slowed speech.
5. **Accessibility**:
   - High-contrast visuals for helpers.
   - Large buttons and simple copy for blind/low-vision users.
   - Optional **vibration** when the target object is detected.

You can customize the browser behavior (thresholds, phrases, detection interval, target list) in `client/app.js`,
and expand backend APIs in `server/server.js`.


