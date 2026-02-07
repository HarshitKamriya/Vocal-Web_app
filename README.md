# Vocal Path - AI Vision Assistant

AI-powered vision assistant for visually impaired users featuring real-time object detection, currency detection, and emergency mode.

## Features

- 🎯 **Object Detection** - Real-time detection using YOLOv8 with voice guidance
- 💵 **Currency Detection** - Identify Indian currency notes (₹10-₹2000)
- 🆘 **Emergency Mode** - SOS alert with location sharing (hold for 2 seconds)
- 🎤 **Voice Guidance** - Hands-free navigation with audio feedback
- 📱 **Mobile First** - Optimized for mobile devices with PWA support

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd vocal-path

# Install dependencies
npm install

# Start the server
npm start
```

Visit `http://localhost:4000` in your browser.

## Deployment

### Option 1: Render (Recommended)
1. Push your code to GitHub
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repository
4. Render will automatically detect the configuration from `render.yaml`
5. Click "Create Web Service"

### Option 2: Railway
1. Push your code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically deploy using the `Procfile`

### Option 3: Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel` in the project directory
3. Follow the prompts

### Option 4: Heroku
```bash
# Login to Heroku
heroku login

# Create a new app
heroku create your-app-name

# Deploy
git push heroku main

# Open the app
heroku open
```

### Option 5: Ngrok (For Testing)
```bash
# Start the server
npm start

# In another terminal, expose it
ngrok http 4000
```

## Project Structure

```
vocal-path/
├── client/           # Frontend files
│   ├── index.html    # Home page
│   ├── detect.html   # Detection page
│   ├── guide.html    # Help/Guide page
│   ├── app.js        # Main JavaScript
│   └── style.css     # Styles
├── models/           # AI Models (ONNX)
│   ├── yolov8n.onnx  # Object detection
│   └── currency_detector.onnx  # Currency detection
├── server/           # Backend
│   └── server.js     # Express server
├── package.json      # Dependencies
├── Procfile          # Heroku/Railway config
├── vercel.json       # Vercel config
└── render.yaml       # Render config
```

## Usage

1. **Home Page**: Tap "Start" to begin detection or "Help" for instructions
2. **Detection Mode**: 
   - Tap green button = Start guidance
   - Hold green button (1.5s) = Currency detection
   - Hold red button (2s) = Emergency mode
3. **Voice Commands**: Double-tap to set target object by voice

## Tech Stack

- **Frontend**: Vanilla JS, HTML5, CSS3
- **Backend**: Node.js, Express
- **AI**: ONNX Runtime Web, YOLOv8
- **APIs**: Web Speech API, Geolocation API

## License

MIT
