# 🚦 Traffic Flow Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org)

**Traffic Flow Analyzer** is a computer vision and deep learning system for automated highway and urban road monitoring. It performs real-time vehicle detection, multi-object centroid tracking, bidirectional tripwire counting, and speed estimation from CCTV feeds, drone footage, recorded video files, and RTSP IP camera streams.

---

## 🌟 Key Features

- **🚗 Real-Time Detection**: Powered by YOLOv8 for detecting multiple vehicle categories: Cars, Motorcycles, Buses, and Trucks.
- **📍 Trajectory Tracking**: Robust centroid-based object tracking with persistent ID management and anti-flicker filtering.
- **⚡ Bidirectional Tripwire Counting**: High-precision virtual counting lines differentiating inbound and outbound vehicular traffic flows.
- **🏎️ Speed Estimation**: Calibrated displacement-over-time velocity measurement in km/h.
- **🖥️ Dynamic HUD & Overlay**: Real-time traffic statistics, per-class counts, bounding boxes, trajectories, and directional indicators.
- **🎮 Built-In Demo Mode**: Includes an offline synthetic video generator for immediate out-of-the-box testing without external dependencies or media files.

---

## 🏗️ System Architecture

```
+--------------------+      +--------------------+      +--------------------+
|  Video Feed / RTSP | ---> |   YOLOv8 Detector  | ---> |  Centroid Tracker  |
+--------------------+      +--------------------+      +--------------------+
                                                                  |
                                                                  v
+--------------------+      +--------------------+      +--------------------+
| Display / MP4 File | <--- |   HUD Visualizer   | <--- | Counter & Velocity |
+--------------------+      +--------------------+      +--------------------+
```

---

## 📂 Project Structure

```
traffic-flow-analyzer/
├── config/
│   └── config.yaml          # Detection, tracking, and calibration settings
├── traffic_monitor/
│   ├── __init__.py          # Package metadata
│   ├── detector.py          # YOLOv8 vehicle detection wrapper
│   ├── tracker.py           # Multi-object centroid tracker
│   ├── counting.py          # Bidirectional tripwire counting logic
│   ├── speed_estimator.py   # Pixel-to-metric velocity estimation
│   ├── visualizer.py        # Heads-up display (HUD) and bounding box renderer
│   └── main.py              # Application entry point & stream pipeline
├── tests/
│   ├── __init__.py
│   ├── test_tracker.py      # Unit tests for object tracking
│   └── test_counting.py     # Unit tests for tripwire logic
├── .gitignore
├── LICENSE                  # MIT License
├── pyproject.toml           # Modern package configuration
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/samba18071-bot/traffic-flow-analyzer.git
cd traffic-flow-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Synthetic Demo (No Video File Needed!)

Run the built-in synthetic traffic demonstration:

```bash
python -m traffic_monitor.main --source demo --max-frames 300
```

### 3. Analyze Video File or Webcam

```bash
# Process a recorded traffic video
python -m traffic_monitor.main --source sample_traffic.mp4 --output output_analyzed.mp4

# Run with webcam (Camera index 0)
python -m traffic_monitor.main --source 0

# Run with RTSP IP Camera stream in headless mode
python -m traffic_monitor.main --source "rtsp://user:pass@camera_ip:554/live" --no-display
```

---

## ⚙️ Configuration

Modify `config/config.yaml` or use CLI flags:

| Parameter | Default | Description |
|---|---|---|
| `--source` | `demo` | Path to video file, RTSP URL, webcam index, or `demo` |
| `--output` | `None` | Path to save annotated output video (.mp4) |
| `--conf` | `0.4` | YOLO detection confidence threshold |
| `--line-ratio` | `0.6` | Vertical position of the counting line (0.0 to 1.0) |
| `--ppm` | `15.0` | Calibration factor (pixels per real-world meter) |
| `--no-display` | `false` | Run in headless mode without GUI window |

---

## 🧪 Running Tests

Validate tracker and counting logic with `unittest` or `pytest`:

```bash
python3 -m unittest discover tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
