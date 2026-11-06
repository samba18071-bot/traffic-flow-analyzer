# 🚦 Traffic Flow Analyzer & Intelligent Dynamic Route Guidance System

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Deployed-FF4B4B.svg?logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Routing-orange.svg)](https://networkx.org)
[![Tests: 21 Passed](https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](tests/)

**An end-to-end Intelligent Transportation System (ITS) combining real-time computer vision edge telemetry (YOLOv8 vehicle detection, centroid tracking, speed estimation, road width measurement, and Level of Service density scoring) with a Bureau of Public Roads (BPR) dynamic graph routing engine to recommend optimal, traffic-aware routes to any destination.**

[Live Streamlit Dashboard](#-streamlit-web-application) • [Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Mathematical Formulations](#-mathematical-foundations) • [Quick Start](#-quick-start) • [Routing Engine](#-dynamic-route-guidance-engine-bpr) • [Configuration](#%EF%B8%8F-configuration)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Mathematical Foundations & Formulas](#-mathematical-foundations)
- [Project Directory Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [1. Streamlit Web Application (GUI Mode)](#1-launch-the-streamlit-web-application-gui-mode)
  - [2. Analyzing Recorded Video Files](#2-analyze-recorded-video-files)
  - [3. Live Webcam & RTSP IP Streams](#3-live-webcam--rtsp-camera-streams)
  - [4. Dynamic Route Navigator CLI](#4-dynamic-route-navigator-cli)
- [Dynamic Route Guidance Engine (BPR)](#-dynamic-route-guidance-engine-bpr)
- [Road Width & Geometry Calipers](#-road-width--geometry-estimation)
- [Configuration Reference](#%EF%B8%8F-configuration)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Cloud Deployment Guide](#-cloud-deployment-guide)
- [License & Acknowledgments](#-license)

---

## 🌐 Overview

Traditional navigation apps rely on lagging, crowdsourced GPS pings that fail to identify localized bottleneck onset. Meanwhile, typical computer vision vehicle counters remain isolated from navigation engines.

**Traffic Flow Analyzer** bridges this gap. It extracts rich, real-time spatial telemetry from camera feeds:
1. **Micro-level**: Vehicle counts, speeds, classifications, and trajectory headings.
2. **Meso-level**: Carriageway physical width, lane capacity, and road surface occupancy percentage.
3. **Macro-level**: Highway Capacity Manual (HCM) Level of Service (LOS A–F), incident anomalies, and dynamic travel-time costs injected into a graph routing network to reroute drivers around emerging traffic jams.

---

## 🌟 Key Features

### 👁️ Computer Vision & Tracking
- **Multi-Class Vehicle Detection**: Powered by Ultralytics YOLOv8 for recognizing Cars, Motorcycles, Buses, and Trucks with confidence filtering.
- **Centroid Trajectory Tracking**: Robust Euclidean object tracker with persistent ID assignment, trajectory trail history, and anti-flicker deregistration.
- **Bidirectional Virtual Tripwire**: High-precision crossing detection differentiating inbound and outbound vehicular streams with per-class aggregation.
- **Calibrated Speed Estimation**: Real-time displacement-over-time velocity calculation in km/h with exponential moving average (EMA) smoothing.

### 🛣️ Road Geometry & Civil Engineering Metrics
- **Dynamic Road Width Measurement**: Calculates road carriageway width in real-world units (meters and feet) using calibrated spatial scaling.
- **Lane Capacity Estimation**: Automatically estimates total traffic lanes based on standard civil guidelines ($3.5\,\text{m}$ standard lane width).
- **Engineering Dimension Calipers**: Renders technical dimension markings, curb end-ticks, and dual-unit measurement badges on the video frame.
- **HCM Level of Service (LOS)**: Real-time roadway space occupancy calculation grading corridor health from **LOS A** (Free Flow) through **LOS F** (Breakdown/Gridlock).

### 🚨 Active Safety Anomaly Detection
- **Wrong-Way Driving Detection**: Flags vehicles moving in reverse against designated lane flow vectors.
- **Stalled / Stopped Vehicle Alerts**: Detects stationary vehicles on active carriageways and logs safety incidents.

### 🧭 Intelligent Dynamic Route Guidance
- **BPR Link Performance Modeling**: Dynamic edge traversal cost calculation based on volume-to-capacity ratios:
  $$T_e = T_0 \left[1 + \alpha \left(\frac{V_e}{C_e}\right)^\beta\right]$$
- **Multi-Objective Routing Strategies**:
  - ⚡ **Fastest Route**: Minimizes absolute travel duration.
  - 🍃 **Eco-Friendly Route**: Minimizes fuel consumption ($L$) and $\text{CO}_2$ emissions ($\text{kg}$) by avoiding stop-and-go speed drops.
  - 🧘 **Smoothest Route**: Diverts traffic away from severe congestion bottlenecks and camera-flagged safety hazards.
- **Interactive SVG/HTML Dashboard**: Self-contained route network visualizer with live congestion color coding, sensor node overlays, and turn-by-turn navigation cards.

---

## 🏗️ System Architecture

```
                 +-----------------------------------------------+
                 |  Video Source (Recorded / Live RTSP / Webcam) |
                 +-----------------------------------------------+
                                         │
                                         ▼
                 +-----------------------------------------------+
                 |        YOLOv8 Real-Time Vehicle Detector      |
                 +-----------------------------------------------+
                                         │
                                         ▼
                 +-----------------------------------------------+
                 |     Multi-Object Centroid Trajectory Tracker  |
                 +-----------------------------------------------+
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
+───────────────+                +───────────────+                +───────────────+
| Tripwire Flow |                | Velocity /    |                | Road Width &  |
| Counting      |                | Speed (km/h)  |                | Lane Geometry |
+───────────────+                +───────────────+                +───────────────+
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         ▼
                 +-----------------------------------------------+
                 |  Traffic Analytics & Level of Service (LOS)   |
                 |  (Occupancy %, Wrong-Way & Stalled Alerts)   |
                 +-----------------------------------------------+
                                         │
                                         ▼
                 +-----------------------------------------------+
                 |  Bureau of Public Roads (BPR) Routing Engine  |
                 |  (Fastest vs. Eco-Friendly vs. Smoothest)     |
                 +-----------------------------------------------+
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
+─────────────────────────────────+             +─────────────────────────────────+
| Streamlit Live Dashboard & Map  |             | HUD Video Stream & Calipers     |
| (Interactive Web UI / Export)   |             | (Annotated MP4 / Live Feed)     |
+─────────────────────────────────+             +─────────────────────────────────+
```

---

## 📐 Mathematical Foundations

### 1. Velocity Estimation
Calculates real-world speed from pixel displacement over a rolling time window:
$$v = \frac{\|\mathbf{p}_t - \mathbf{p}_{t-\Delta t}\|_2}{\text{PPM} \cdot \left(\frac{\Delta t}{\text{FPS}}\right)} \times 3.6 \quad [\text{km/h}]$$
- $\mathbf{p}_t$: Vehicle centroid coordinates at frame $t$
- $\text{PPM}$: Spatial calibration factor ($\text{pixels per meter}$)
- $\text{FPS}$: Video stream frame rate

### 2. Road Width & Lane Capacity
$$\text{Width (m)} = \frac{\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}}{\text{PPM}}$$
$$\text{Estimated Lanes} = \max\left(1, \operatorname{round}\left(\frac{\text{Width (m)}}{W_{\text{standard}}}\right)\right) \quad (W_{\text{standard}} = 3.5\,\text{m})$$

### 3. Roadway Occupancy & Level of Service (LOS)
$$\text{Occupancy \%} = \min\left(100.0, \frac{\sum_{i=1}^N \text{Area}_i}{\text{Width}_{\text{road}} \times \text{Length}_{\text{road}}} \times 100\right)$$

| LOS Grade | Flow State | Occupancy % | Average Speed | Color Code |
|:---:|:---:|:---:|:---:|:---:|
| **A** | Free Flow | $< 10\%$ | $\ge 50\,\text{km/h}$ | 🟢 Bright Green |
| **B** | Reasonably Free Flow | $10 - 18\%$ | $\ge 40\,\text{km/h}$ | 🟢 Light Green |
| **C** | Stable Flow | $18 - 27\%$ | $\ge 30\,\text{km/h}$ | 🟡 Yellow |
| **D** | Approaching Capacity | $27 - 38\%$ | $\ge 22\,\text{km/h}$ | 🟠 Amber |
| **E** | Unstable / Near Jam | $38 - 50\%$ | $\ge 15\,\text{km/h}$ | 🟠 Deep Orange |
| **F** | Forced Flow / Gridlock | $> 50\%$ | $< 15\,\text{km/h}$ | 🔴 Red |

### 4. Bureau of Public Roads (BPR) Link Cost Function
Civil engineering standard for dynamic road link impedance:
$$T_e = T_0 \cdot \left[1 + \alpha \left(\frac{V_e}{C_e}\right)^\beta\right]$$
- $T_0$: Free-flow traversal time ($\frac{\text{length}}{\text{speed limit}} \times 60\,\text{min}$)
- $V_e$: Real-time traffic volume (vehicles per hour equivalent)
- $C_e$: Carriageway capacity ($\text{lanes} \times 1800\,\text{vph}$)
- $\alpha = 0.15, \beta = 4.0$: Standard BPR parameters

---

## 📂 Project Structure

```
traffic-flow-analyzer/
├── config/
│   └── config.yaml              # Detection, tracking, geometry & routing settings
├── traffic_monitor/
│   ├── __init__.py              # Package metadata and public exports
│   ├── detector.py              # YOLOv8 inference wrapper & class filtering
│   ├── tracker.py               # Centroid tracker with persistent ID association
│   ├── counting.py              # Virtual tripwire counting logic
│   ├── speed_estimator.py       # Pixel-to-metric velocity estimation
│   ├── road_width_estimator.py   # Road width calipers & lane capacity estimator
│   ├── traffic_analytics.py     # Level of Service (LOS) & safety anomaly detector
│   ├── routing_engine.py        # BPR dynamic graph routing engine
│   ├── route_map_generator.py   # Standalone interactive SVG/HTML map generator
│   ├── route_navigator.py       # CLI route optimization & scenario runner
│   ├── visualizer.py            # HUD, bounding boxes, calipers, & route badge renderer
│   └── main.py                  # Core video pipeline with real-time routing link
├── tests/
│   ├── __init__.py
│   ├── test_tracker.py          # Unit tests for object tracking
│   ├── test_counting.py         # Unit tests for tripwire logic
│   ├── test_road_width.py       # Unit tests for road width & lane estimation
│   ├── test_traffic_analytics.py# Unit tests for LOS & anomaly detection
│   └── test_routing_engine.py   # Unit tests for BPR costs & dynamic rerouting
├── app.py                       # Streamlit web application entrypoint
├── packages.txt                 # Linux system dependencies for Streamlit Cloud
├── pyproject.toml               # Modern build configuration
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### 1. Launch the Streamlit Web Application (GUI Mode)

The most intuitive way to run the project with recorded videos or live webcams:

```bash
# Clone the repository
git clone https://github.com/samba18071-bot/traffic-flow-analyzer.git
cd traffic-flow-analyzer

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to:
- Select **Recorded Videos** (drag-and-drop your video or use the built-in sample).
- Connect **Live Cameras** (built-in webcam index `0`, USB cameras, or RTSP streams).
- Tune confidence thresholds, tripwire line position, and spatial calibration sliders.
- Select target destinations and watch the route recalculate in real time.
- Explore the interactive road network map and export session analytics in JSON/CSV.

---

### 2. Analyze Recorded Video Files

Process any MP4/AVI/MOV traffic video and export an annotated output video:

```bash
python -m traffic_monitor.main \
  --source sample_traffic.mp4 \
  --output output_analyzed.mp4 \
  --conf 0.4 \
  --ppm 15.0 \
  --lane-width 3.5
```

---

### 3. Live Webcam & RTSP Camera Streams

Run live video streams in interactive or headless server mode:

```bash
# Run with local webcam (device index 0)
python -m traffic_monitor.main --source 0

# Run with RTSP IP Camera in headless mode with destination guidance to Airport (F)
python -m traffic_monitor.main \
  --source "rtsp://user:pass@camera_ip:554/live" \
  --route-dest F \
  --no-display
```

---

### 4. Dynamic Route Navigator CLI

Query optimal routes across the road network or simulate incident bottlenecks:

```bash
# 1. Query best route from Downtown (A) to Airport (F)
python -m traffic_monitor.route_navigator --origin A --destination F

# 2. Simulate heavy gridlock on Highway Link A → B and generate interactive HTML dashboard
python -m traffic_monitor.route_navigator \
  --origin A \
  --destination F \
  --simulate-congestion \
  --html route_guidance_dashboard.html
```

---

## 🧭 Dynamic Route Guidance Engine (BPR)

The routing engine models road networks as directed graphs $G = (V, E)$, where edges store physical carriageway attributes and dynamic telemetry:

- **Edge Capacity ($C_e$)**: Dynamically computed from road width and lane count.
- **Dynamic Impedance**: Evaluated per edge using the BPR link performance function.
- **Automatic Rerouting**: When a primary corridor experiences severe congestion (LOS E/F) or an active road hazard, the engine dynamically discovers detours to minimize travel delay and emissions.

```
▶ STRATEGY: FASTEST
  • Path:         Downtown Central ➔ Harbor Expressway ➔ Metro Ring Interchange ➔ Airport
  • Total Time:   11.9 mins
  • Distance:     12.7 km
  • Avg Speed:    64.2 km/h
  • Fuel Usage:   0.83 Liters (~1.91 kg CO2)
```

---

## ⚙️ Configuration

Tune parameters via [`config/config.yaml`](config/config.yaml) or CLI arguments:

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| `source` | `--source` | `demo` | Video path, webcam index (`0`), RTSP URL, or `demo` |
| `output` | `--output` | `None` | Path to save annotated output video (.mp4) |
| `confidence_threshold` | `--conf` | `0.40` | YOLOv8 detection confidence threshold |
| `line_position_ratio` | `--line-ratio`| `0.60` | Vertical position ratio of counting line ($0.0 - 1.0$) |
| `pixels_per_meter` | `--ppm` | `15.0` | Spatial calibration factor ($\text{px} / \text{meter}$) |
| `standard_lane_width_m`| `--lane-width`| `3.5` | Standard highway lane width in meters |
| `road_left` | `--road-left` | `None` | Optional left road boundary coordinate ($X$) |
| `road_right` | `--road-right`| `None` | Optional right road boundary coordinate ($X$) |
| `route_dest` | `--route-dest`| `None` | Destination junction ID for live HUD navigation (`B`-`F`) |
| `no_display` | `--no-display`| `false` | Run in headless mode without GUI window |

---

## 🧪 Testing & Quality Assurance

The codebase includes comprehensive unit tests verifying tracking, tripwire counting, geometry estimation, Level of Service calculations, and dynamic routing:

```bash
# Run complete test suite with pytest
pytest -v
```

```
============================== test session starts ==============================
tests/test_counting.py::test_initial_state PASSED                         [  4%]
tests/test_counting.py::test_inbound_crossing PASSED                      [  9%]
tests/test_counting.py::test_outbound_crossing PASSED                     [ 14%]
tests/test_road_width.py::test_initial_state PASSED                       [ 19%]
tests/test_road_width.py::test_explicit_horizontal_boundaries PASSED      [ 23%]
tests/test_road_width.py::test_diagonal_boundaries PASSED                 [ 28%]
tests/test_road_width.py::test_update_from_vehicle_boxes PASSED           [ 33%]
tests/test_road_width.py::test_edge_detection_on_synthetic_frame PASSED   [ 38%]
tests/test_routing_engine.py::test_default_network PASSED                 [ 42%]
tests/test_routing_engine.py::test_free_flow_fastest_route PASSED         [ 47%]
tests/test_routing_engine.py::test_dynamic_rerouting_on_congestion PASSED [ 52%]
tests/test_routing_engine.py::test_bpr_cost_scaling PASSED                [ 57%]
tests/test_routing_engine.py::test_compare_routes_and_html_generation PASSED [ 61%]
tests/test_tracker.py::test_tracker_initial_state PASSED                  [ 66%]
tests/test_tracker.py::test_object_registration_and_update PASSED        [ 71%]
tests/test_tracker.py::test_object_deregistration PASSED                  [ 76%]
tests/test_traffic_analytics.py::test_los_empty_traffic PASSED            [ 80%]
tests/test_traffic_analytics.py::test_los_free_flow PASSED                [ 85%]
tests/test_traffic_analytics.py::test_los_congested PASSED                [ 90%]
tests/test_traffic_analytics.py::test_stalled_vehicle_detection PASSED   [ 95%]
tests/test_traffic_analytics.py::test_wrong_way_detection PASSED         [100%]
============================== 21 passed in 0.22s ===============================
```

---

## ☁️ Cloud Deployment Guide

To deploy this application to **Streamlit Community Cloud**:
1. Fork or push this repository to your GitHub account.
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and click **New app**.
3. Fill in the repository settings:
   - **Repository**: `samba18071-bot/traffic-flow-analyzer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!** The bundled [`packages.txt`](packages.txt) ensures headless OpenCV Linux dependencies are provisioned automatically.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

---

<div align="center">

Developed with ❤️ for intelligent, greener, and safer urban mobility.

⭐ **If you find this project useful, please consider giving it a star on GitHub!**

</div>
