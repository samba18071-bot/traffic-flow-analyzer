"""
Streamlit Web Application for Traffic Flow Analyzer & Intelligent Route Guidance.
Supports recorded videos (upload or sample), live webcam, RTSP streams,
real-time YOLOv8 detection, centroid tracking, speed & road width estimation,
HCM Level of Service (LOS), safety anomaly detection, and BPR dynamic routing.
"""

import os
import tempfile
import time
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from traffic_monitor.detector import VehicleDetector
from traffic_monitor.tracker import CentroidTracker
from traffic_monitor.counting import TrafficCounter
from traffic_monitor.speed_estimator import SpeedEstimator
from traffic_monitor.road_width_estimator import RoadWidthEstimator
from traffic_monitor.traffic_analytics import TrafficAnalytics
from traffic_monitor.routing_engine import DynamicRoutingEngine
from traffic_monitor.route_map_generator import RouteMapGenerator
from traffic_monitor.visualizer import Visualizer
from traffic_monitor.main import generate_synthetic_frame


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Traffic Flow Analyzer & Dynamic Router",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS STYLING ----------------
st.markdown("""
<style>
    .reportview-container { background: #0b0f19; }
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 16px 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-title {
        color: #38bdf8;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .metric-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .los-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .route-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38bdf8;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.image("https://img.shields.io/badge/YOLOv8-Computer%20Vision-green?style=for-the-badge&logo=yolo", use_container_width=True)
st.sidebar.title("🎛️ Control Panel")

# 1. Source Selection
st.sidebar.subheader("1. Video Input Stream")
source_type = st.sidebar.radio(
    "Select Video Source:",
    (
        "Built-in Sample Video",
        "Upload Recorded Video",
        "Live Camera / Webcam",
        "Synthetic Road Simulation"
    )
)

video_source = None
is_synthetic = False

if source_type == "Built-in Sample Video":
    sample_path = "sample_traffic.mp4"
    if not os.path.exists(sample_path):
        with st.spinner("Generating sample traffic video..."):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(sample_path, fourcc, 30.0, (1280, 720))
            for i in range(180):
                f, _ = generate_synthetic_frame(i, 1280, 720)
                out.write(f)
            out.release()
    video_source = sample_path
    st.sidebar.info("Using built-in `sample_traffic.mp4` corridor feed.")

elif source_type == "Upload Recorded Video":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video file (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_source = tfile.name
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    else:
        st.sidebar.warning("Please upload a video file to proceed.")

elif source_type == "Live Camera / Webcam":
    camera_mode = st.sidebar.selectbox("Camera Connection Type:", ["Local Webcam Index", "RTSP / IP Camera Stream"])
    if camera_mode == "Local Webcam Index":
        cam_idx = st.sidebar.number_input("Webcam Device Index (0=Built-in):", min_value=0, max_value=5, value=0, step=1)
        video_source = int(cam_idx)
    else:
        rtsp_url = st.sidebar.text_input("Stream URL:", value="rtsp://admin:password@192.168.1.100:554/live")
        video_source = rtsp_url

elif source_type == "Synthetic Road Simulation":
    is_synthetic = True
    st.sidebar.info("Running synthetic procedural traffic generator.")

# 2. Calibration & Detection Settings
st.sidebar.subheader("2. Model & Geometry Tuning")
conf_thresh = st.sidebar.slider("YOLO Confidence Threshold:", 0.10, 0.95, 0.40, 0.05)
line_ratio = st.sidebar.slider("Counting Tripwire Height (Y Ratio):", 0.20, 0.90, 0.60, 0.05)
pixels_per_meter = st.sidebar.number_input("Spatial Scale (Pixels per Meter):", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
lane_width_m = st.sidebar.number_input("Standard Lane Width (meters):", min_value=2.5, max_value=5.0, value=3.5, step=0.1)

# 3. Dynamic Route Guidance Destination
st.sidebar.subheader("3. Dynamic Route Guidance Target")
destination_options = {
    "F": "International Airport (F)",
    "D": "Metro Ring Interchange (D)",
    "E": "Old Town Boulevard (E)",
    "C": "Harbor Expressway (C)",
    "B": "Tech Park Junction (B)"
}
dest_code = st.sidebar.selectbox(
    "Choose Destination from Downtown (A):",
    list(destination_options.keys()),
    format_func=lambda k: destination_options[k]
)
route_strategy = st.sidebar.selectbox(
    "Optimization Objective:",
    ["fastest", "eco", "smooth"],
    format_func=lambda s: {"fastest": "⚡ Fastest (Min Time)", "eco": "🍃 Eco-Green (Min Fuel)", "smooth": "🧘 Smoothest (Min Congestion)"}[s]
)

# 4. Execution Controls
st.sidebar.subheader("4. Streaming Execution")
col_start, col_stop = st.sidebar.columns(2)
start_clicked = col_start.button("▶ Start", use_container_width=True, type="primary")
stop_clicked = col_stop.button("⏹ Stop", use_container_width=True)

if "running" not in st.session_state:
    st.session_state["running"] = False

if start_clicked:
    st.session_state["running"] = True
if stop_clicked:
    st.session_state["running"] = False


# ---------------- MAIN DASHBOARD HEADER ----------------
st.markdown("""
<div class="main-header">
    <div>
        <h1 class="main-title">🚦 Traffic Flow Analyzer & Intelligent Dynamic Route Guidance</h1>
        <p style="color:#94a3b8; margin-top:4px; font-size:0.9rem;">
            Real-Time YOLOv8 Detection • Multi-Object Tracking • Road Width Calipers • HCM Level of Service (LOS) • BPR Dynamic Routing
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- LAYOUT COLUMNS ----------------
col_video, col_telemetry = st.columns([1.6, 1.0])

with col_video:
    st.subheader("🎥 Real-Time Video Stream with Analytics HUD")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col_telemetry:
    st.subheader("📊 Live Telemetry & Guidance")
    metrics_placeholder = st.empty()
    route_card_placeholder = st.empty()


# ---------------- TABS FOR MAP AND EXPORT ----------------
tab_map, tab_incidents, tab_export = st.tabs([
    "🗺️ Interactive Road Network Map & Routes",
    "⚠️ Safety Anomalies & Incident Log",
    "📈 Traffic Analytics & Data Export"
])


# ---------------- VIDEO STREAMING & INFERENCE LOOP ----------------
if st.session_state["running"]:
    if not is_synthetic and video_source is None:
        st.error("Please provide a valid video file or camera input.")
    else:
        # Initialize modules
        cap = None
        if not is_synthetic:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.error(f"Failed to open video source: {video_source}")
                st.stop()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            width, height, fps = 1280, 720, 30.0

        line_y = int(height * line_ratio)

        detector = VehicleDetector(conf_thresh=conf_thresh)
        tracker = CentroidTracker(max_disappeared=30, max_distance=80.0)
        counter = TrafficCounter(line_y=line_y)
        speed_estimator = SpeedEstimator(pixels_per_meter=pixels_per_meter, fps=fps)
        road_estimator = RoadWidthEstimator(pixels_per_meter=pixels_per_meter, standard_lane_width_m=lane_width_m)
        traffic_analytics = TrafficAnalytics(pixels_per_meter=pixels_per_meter)
        routing_engine = DynamicRoutingEngine()
        visualizer = Visualizer()

        # Preset default road boundaries for synthetic demo
        if is_synthetic or source_type == "Built-in Sample Video":
            road_x1, road_x2 = int(width * 0.25), int(width * 0.75)
            road_estimator.set_horizontal_boundaries(road_x1, road_x2, line_y)

        frame_idx = 0
        latest_route = None

        try:
            while st.session_state.get("running", True):
                if is_synthetic:
                    frame, detections = generate_synthetic_frame(frame_idx, width, height)
                    time.sleep(1.0 / max(fps, 1.0))
                else:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video for continuous monitoring
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    detections = detector.detect(frame)

                    # Auto curb detection on first frame if not set
                    if frame_idx == 0 and road_estimator.p1 is None:
                        road_estimator.detect_road_edges_from_frame(frame, line_y)

                # Refine road width estimate from vehicle bounding boxes
                boxes = [d["box"] for d in detections]
                road_estimator.update_from_vehicle_boxes(boxes, line_y)

                # Update centroid tracker
                tracked_objects = tracker.update(detections)

                # Update velocities
                speeds = {}
                for obj_id, data in tracked_objects.items():
                    speeds[obj_id] = speed_estimator.estimate_speed(obj_id, data["trajectory"])

                # Tripwire counting
                inbound, outbound, counts_by_class = counter.process_objects(tracked_objects)

                # Road width metrics
                road_metrics = road_estimator.calculate_width()

                # Level of Service (LOS) & Occupancy
                los_data = traffic_analytics.compute_level_of_service(
                    tracked_objects=tracked_objects,
                    road_width_m=road_metrics["width_meters"],
                    frame_height_px=height,
                    current_speeds=speeds
                )

                # Detect safety anomalies
                alerts = traffic_analytics.detect_anomalies(
                    tracked_objects=tracked_objects,
                    current_speeds=speeds,
                    frame_idx=frame_idx,
                    frame_width=width
                )

                # Dynamic Route Guidance recalculation
                if frame_idx % 10 == 0:
                    routing_engine.update_traffic_telemetry(
                        u="A",
                        v="B",
                        vehicle_volume=counter.total_count,
                        avg_speed_kmh=los_data.get("avg_speed_kmh", 50.0),
                        road_width_m=road_metrics.get("width_meters"),
                        lanes=road_metrics.get("estimated_lanes"),
                        los_grade=los_data.get("los_grade"),
                        has_hazard=len(alerts) > 0
                    )
                    latest_route = routing_engine.find_route("A", dest_code, strategy=route_strategy)

                route_info = None
                if latest_route:
                    route_info = {
                        "dest": dest_code,
                        "eta": latest_route["total_time_min"],
                        "dist": latest_route["total_distance_km"],
                        "via": "➔".join(latest_route["path_nodes"])
                    }

                # Render annotated video frame
                annotated_frame = visualizer.draw_scene(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    line_y=line_y,
                    inbound_count=inbound,
                    outbound_count=outbound,
                    counts_by_class=counts_by_class,
                    speeds=speeds,
                    road_estimator=road_estimator,
                    road_metrics=road_metrics,
                    los_data=los_data,
                    alerts=alerts,
                    route_info=route_info
                )

                # Convert BGR to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update live alerts banner
                if alerts:
                    alert_msgs = " | ".join([f"⚠️ {a['message']}" for a in alerts])
                    alert_placeholder.error(alert_msgs)
                else:
                    alert_placeholder.empty()

                # Update live telemetry metrics panel
                with metrics_placeholder.container():
                    m_c1, m_c2 = st.columns(2)
                    m_c1.metric("🚗 Total Count", counter.total_count, delta=f"+{inbound} in / {outbound} out")
                    
                    los_grade = los_data.get("los_grade", "A")
                    los_desc = los_data.get("los_description", "Free Flow")
                    occ = los_data.get("occupancy_pct", 0.0)
                    m_c2.metric("🚦 Level of Service (LOS)", f"LOS {los_grade}", delta=f"{occ}% Occupancy", delta_color="inverse" if los_grade in ["D","E","F"] else "normal")

                    m_c3, m_c4 = st.columns(2)
                    road_m = road_metrics.get("width_meters", 0.0)
                    lanes = road_metrics.get("estimated_lanes", 0)
                    m_c3.metric("🛣️ Measured Road Width", f"{road_m:.1f} m", delta=f"{lanes} Lanes")
                    avg_v = los_data.get("avg_speed_kmh", 0.0)
                    m_c4.metric("⚡ Average Speed", f"{avg_v:.1f} km/h")

                # Update dynamic route card
                if latest_route:
                    with route_card_placeholder.container():
                        st.markdown(f"""
                        <div class="route-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:#38bdf8; font-size:1.05rem;">📍 Route: Downtown (A) ➔ {destination_options[dest_code]}</strong>
                                <span style="background:rgba(56,189,248,0.2); color:#38bdf8; padding:3px 8px; border-radius:6px; font-size:0.8rem;">
                                    {route_strategy.upper()}
                                </span>
                            </div>
                            <div style="margin-top:8px; font-size:0.95rem; color:#f8fafc;">
                                <strong>Path:</strong> {' ➔ '.join(latest_route['path_names'])}
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-top:10px; font-size:0.88rem; color:#94a3b8;">
                                <span>⏱️ <strong>ETA:</strong> {latest_route['total_time_min']} mins</span>
                                <span>📏 <strong>Dist:</strong> {latest_route['total_distance_km']} km</span>
                                <span>⛽ <strong>Fuel:</strong> {latest_route['total_fuel_liters']} L ({latest_route['co2_emissions_kg']} kg CO₂)</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                frame_idx += 1

        finally:
            if cap:
                cap.release()

        # Update tabs after run / on stop
        with tab_map:
            st.subheader("🗺️ Live Road Network Graph & Multi-Route Comparison")
            comparison = routing_engine.compare_routes("A", dest_code)
            html_file = "route_guidance_dashboard.html"
            RouteMapGenerator.generate_html(routing_engine, comparison, output_filepath=html_file)
            with open(html_file, "r", encoding="utf-8") as f:
                html_code = f.read()
            components.html(html_code, height=650)

        with tab_incidents:
            st.subheader("⚠️ Safety Anomaly Incident Log")
            if traffic_analytics.incident_log:
                df_inc = pd.DataFrame(traffic_analytics.incident_log)
                st.dataframe(df_inc[["frame_idx", "type", "obj_id", "message"]], use_container_width=True)
            else:
                st.info("No safety violations or anomalies logged during this session.")

        with tab_export:
            st.subheader("📈 Session Summary & Export")
            summary_data = {
                "total_vehicles": counter.total_count,
                "inbound_flow": counter.inbound_count,
                "outbound_flow": counter.outbound_count,
                "vehicle_breakdown": counter.counts_by_class,
                "road_width_meters": road_metrics.get("width_meters", 0.0),
                "estimated_lanes": road_metrics.get("estimated_lanes", 0),
                "final_los_grade": los_data.get("los_grade", "A"),
                "occupancy_pct": los_data.get("occupancy_pct", 0.0),
                "avg_speed_kmh": los_data.get("avg_speed_kmh", 0.0),
                "recommended_route_eta_min": latest_route["total_time_min"] if latest_route else None,
                "recommended_path": latest_route["path_nodes"] if latest_route else []
            }
            st.json(summary_data)
            st.download_button(
                "📥 Download Session Analytics (JSON)",
                data=json.dumps(summary_data, indent=2),
                file_name="traffic_analytics_report.json",
                mime="application/json"
            )

else:
    # Idle welcome screen when stopped
    with col_video:
        st.info("👈 Select your video source (Built-in sample, Upload recorded video, or Live Camera) in the sidebar and click **Start** to begin real-time analysis.")
        if os.path.exists("sample_traffic.mp4"):
            st.markdown("**Preview Frame:**")
            f_prev, _ = generate_synthetic_frame(0, 1280, 720)
            st.image(cv2.cvtColor(f_prev, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    with col_telemetry:
        st.markdown("""
        <div class="metric-container">
            <h4>💡 System Capabilities Ready</h4>
            <ul style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
                <li><strong>Recorded Video:</strong> Upload your own or use the built-in corridor video.</li>
                <li><strong>Live Camera:</strong> Plug in your webcam or enter an RTSP stream.</li>
                <li><strong>Road Width:</strong> Real-time caliper overlay in meters and feet.</li>
                <li><strong>LOS Density:</strong> Automatic HCM Grade (A–F) congestion scoring.</li>
                <li><strong>Route Guidance:</strong> Dynamic BPR cost optimization to destination.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_map:
        st.subheader("🗺️ Pre-Loaded Road Network Map")
        engine_preview = DynamicRoutingEngine()
        comp_prev = engine_preview.compare_routes("A", "F")
        html_file = "route_guidance_dashboard.html"
        RouteMapGenerator.generate_html(engine_preview, comp_prev, output_filepath=html_file)
        with open(html_file, "r", encoding="utf-8") as f:
            html_code = f.read()
        components.html(html_code, height=650)
