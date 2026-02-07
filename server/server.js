const path = require("path");
const express = require("express");
const cors = require("cors");
const morgan = require("morgan");

const app = express();
const PORT = process.env.PORT || 4000;

// Paths
const CLIENT_BUILD_PATH = path.join(__dirname, "..", "client");
const MODELS_PATH = path.join(__dirname, "..", "models");

// Middleware
app.use(morgan("dev"));
app.use(cors());

// Static models folder (from root)
app.use("/models", express.static(MODELS_PATH));

// Static frontend
app.use(express.static(CLIENT_BUILD_PATH));

// Simple health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "Vocal Path server running" });
});

// All other routes serve index.html (for SPA-style navigation if extended)
app.get("*", (req, res) => {
  res.sendFile(path.join(CLIENT_BUILD_PATH, "index.html"));
});

app.listen(PORT, () => {
  console.log(`Vocal Path server listening on http://localhost:${PORT}`);
});

