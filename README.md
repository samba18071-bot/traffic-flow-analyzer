# 🚦 Traffic Flow Analyzer & Intelligent Dynamic Route Guidance System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Routing-orange.svg)](https://networkx.org)

**Traffic Flow Analyzer** is a real-time computer vision, deep learning, and dynamic route optimization platform. It combines edge vehicle detection, multi-object tracking, speed estimation, road width measurement, and Level of Service (LOS A–F) telemetry with a **Bureau of Public Roads (BPR) Graph Routing Engine** to dynamically recommend the fastest, greenest, and smoothest routes to any destination.

---

## 🌟 Key Features

- **🚗 Real-Time Detection**: Powered by YOLOv8 for detecting multiple vehicle categories: Cars, Motorcycles, Buses, and Trucks.
- **📍 Trajectory Tracking**: Robust centroid-based object tracking with persistent ID management and anti-flicker filtering.
- **⚡ Bidirectional Tripwire Counting**: High-precision virtual counting lines differentiating inbound and outbound vehicular traffic flows.
- **🏎️ Speed Estimation**: Calibrated displacement-over-time velocity measurement in km/h.
- **🛣️ Road Width & Geometry Estimation**: Real-time road carriageway width measurement in meters and feet, lane capacity estimation, and engineering caliper overlay.
- **📊 Level of Service (LOS) & Occupancy**: Real-time highway congestion density analysis (LOS A to F per HCM standards) and corridor occupancy percentage.
- **🚨 Safety Anomaly Detection**: Automatic detection and alerting for Wrong-Way Driving and Stalled/Stopped Vehicles.
- **🧭 Dynamic Route Guidance Engine**: Graph-based multi-objective route guidance using the Bureau of Public Roads (BPR) congestion function.
  - **⚡ Fastest Route**: Minimizes total travel time based on live congestion.
  - **🍃 Eco-Friendly Route**: Minimizes fuel consumption ($L$) and carbon emissions ($\text{kg CO}_2$).
  - **🧘 Smoothest Route**: Actively diverts drivers away from stop-and-go bottlenecks and road hazards.
- **🗺️ Interactive HTML Dashboard**: Self-contained SVG/HTML interactive network map with live congestion color coding, camera sensor nodes, and turn-by-turn navigation.
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
| Display / MP4 File | <--- |   HUD Visualizer   | <--- | Counter, Velocity  |
|                    |      |  (Calipers & HUD)  |      |   & Road Width     |
+--------------------+      +--------------------+      +--------------------+
                                                                  |
                                     (Live Telemetry & LOS)       v
                            +--------------------+      +--------------------+
                            | Interactive HTML   | <--- | BPR Dynamic Route  |
                            | Web Dashboard      |      | Guidance Engine    |
                            +--------------------+      +--------------------+
```

---

## 📂 Project Structure

```
traffic-flow-analyzer/
├── config/
│   └── config.yaml          # Detection, tracking, road geometry & routing settings
├── traffic_monitor/
│   ├── __init__.py          # Package metadata & public exports
│   ├── detector.py          # YOLOv8 vehicle detection wrapper
│   ├── tracker.py           # Multi-object centroid tracker
│   ├── counting.py          # Bidirectional tripwire counting logic
│   ├── speed_estimator.py   # Pixel-to-metric velocity estimation
│   ├── road_width_estimator.py # Road width and lane capacity estimation
│   ├── traffic_analytics.py # Level of Service (LOS) & safety anomaly detector
│   ├── routing_engine.py    # BPR dynamic graph routing & multi-objective solver
│   ├── route_map_generator.py # Standalone interactive HTML route map generator
│   ├── route_navigator.py   # CLI entry point for route queries & simulation
│   ├── visualizer.py        # Heads-up display (HUD), calipers, & route badge renderer
│   └── main.py              # Video stream pipeline with real-time routing link
├── tests/
│   ├── __init__.py
│   ├── test_tracker.py      # Unit tests for object tracking
│   ├── test_counting.py     # Unit tests for tripwire logic
│   ├── test_road_width.py   # Unit tests for road width estimator
│   ├── test_traffic_analytics.py # Unit tests for LOS and anomaly alerts
│   └── test_routing_engine.py # Unit tests for dynamic routing & BPR costs
├── .gitignore
├── LICENSE                  # MIT License
├── pyproject.toml           # Modern package configuration
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Launch the Streamlit Web Application (GUI Mode)

Run the modern Streamlit dashboard for recorded video analysis and live camera streams:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to:
- Select **Recorded Videos** (upload any MP4/AVI/MOV file or use the built-in corridor sample).
- Switch to **Live Camera / Webcam** (device index 0, USB camera, or RTSP/IP camera URL).
- Watch live YOLOv8 tracking, road width calipers, Level of Service (LOS) density, and real-time destination route guidance.
- Explore the interactive road network map and download session analytics reports.

---

### 2. Run Synthetic Demo via CLI

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

### 4. Dynamic Route Guidance & Interactive Dashboard

Find optimal traffic-aware routes across the road network using real-time BPR link performance modeling:

```bash
# Query best route from Downtown (A) to Airport (F)
python -m traffic_monitor.route_navigator --origin A --destination F

# Simulate real-time camera gridlock on primary highway link and generate interactive HTML dashboard
python -m traffic_monitor.route_navigator --origin A --destination F --simulate-congestion --html route_guidance_dashboard.html

# Run live video pipeline with real-time HUD route navigation guidance to Airport (F)
python -m traffic_monitor.main --source demo --route-dest F
```

---

## 🧭 Dynamic Route Guidance Engine (BPR Function)

The routing engine bridges live computer vision metrics with graph network optimization using the **Bureau of Public Roads (BPR)** Link Performance Function:

$$T_e = T_0 \cdot \left[1 + \alpha \left(\frac{V_e}{C_e}\right)^\beta\right]$$

- **$T_0$**: Free-flow travel time ($\text{segment length} / \text{speed limit}$)
- **$V_e$**: Vehicle volume (from YOLOv8 tripwire counter)
- **$C_e$**: Practical road capacity, calculated from measured **road width** & **lanes** ($C_e = \text{lanes} \times 1800\,\text{vph}$)
- **$\alpha, \beta$**: Civil engineering calibration parameters ($\alpha = 0.15, \beta = 4.0$)

### Multi-Objective Route Strategies:
1. **⚡ Fastest (Min Time)**: Optimizes strictly for lowest estimated traversal duration in minutes.
2. **🍃 Eco-Friendly (Min Emissions & Fuel)**: Penalizes stop-and-go congestion cycles ($< 30\,\text{km/h}$ / LOS E & F) to minimize liters of fuel and $\text{kg CO}_2$ emissions.
3. **🧘 Smoothest (Min Stress)**: Actively diverts vehicles away from severe congestion bottlenecks and camera-flagged safety hazards (stalled vehicles, wrong-way traffic).

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
| `--lane-width` | `3.5` | Standard lane width in meters for lane count estimation |
| `--road-left` | `None` | Optional X pixel coordinate of left road boundary |
| `--road-right` | `None` | Optional X pixel coordinate of right road boundary |
| `--route-dest` | `None` | Optional destination node ID (e.g. `F`) for live HUD navigation |
| `--no-display` | `false` | Run in headless mode without GUI window |

---

## 🛣️ Road Width & Geometry Estimation

The road width estimation module measures carriageway dimensions and layout in real time:
1. **Calibration**: Converts pixel distance to metric dimensions via spatial scale ratio ($\text{ppm} = \text{pixels per meter}$).
2. **Lane Estimation**: Computes estimated lane count:
   $$\text{Estimated Lanes} = \max\left(1, \operatorname{round}\left(\frac{\text{Width (m)}}{\text{Standard Lane Width (m)}}\right)\right)$$
3. **Automated Tracking & Refinement**: Dynamic vehicle envelope tracking refines road bounds based on observed vehicular pathways plus safety shoulder margins.
4. **Engineering Caliper Overlay**: Displays an on-screen dimension line with end-caps, measurement badge, and dual-unit metric/imperial metrics (meters & feet).

---

## 🧪 Running Tests

Validate tracker and counting logic with `unittest` or `pytest`:

```bash
python3 -m unittest discover tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

