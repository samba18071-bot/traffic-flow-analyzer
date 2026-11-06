"""
Main CLI entrypoint for Traffic Flow Analyzer.
Processes video streams, runs detection, tracking, speed calculation, and visualization.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from traffic_monitor.detector import VehicleDetector
from traffic_monitor.tracker import CentroidTracker
from traffic_monitor.counting import TrafficCounter
from traffic_monitor.speed_estimator import SpeedEstimator
from traffic_monitor.road_width_estimator import RoadWidthEstimator
from traffic_monitor.traffic_analytics import TrafficAnalytics
from traffic_monitor.visualizer import Visualizer


def generate_synthetic_frame(frame_idx: int, width: int = 1280, height: int = 720) -> Tuple[np.ndarray, list]:
    """Generates synthetic road frames and vehicles for offline demonstration/testing."""
    frame = np.full((height, width, 3), 40, dtype=np.uint8)

    # Road lanes
    road_x1, road_x2 = int(width * 0.25), int(width * 0.75)
    cv2.rectangle(frame, (road_x1, 0), (road_x2, height), (70, 70, 70), -1)

    # Center lane divider (dashed line)
    center_x = int(width * 0.5)
    for y in range(0, height, 40):
        cv2.line(frame, (center_x, y), (center_x, y + 20), (255, 255, 255), 2)

    # Synthetic vehicle positions
    v1_y = (frame_idx * 6) % (height + 100) - 50
    v2_y = height - ((frame_idx * 8) % (height + 100))

    detections = []
    # Inbound car (left lane)
    if -40 < v1_y < height:
        x1, y1 = center_x - 120, int(v1_y)
        detections.append({
            "box": (x1, y1, x1 + 60, y1 + 100),
            "conf": 0.95,
            "class_id": 2,
            "class_name": "car"
        })
        cv2.rectangle(frame, (x1, y1), (x1 + 60, y1 + 100), (180, 50, 50), -1)

    # Outbound truck (right lane)
    if -80 < v2_y < height:
        x2, y2 = center_x + 50, int(v2_y)
        detections.append({
            "box": (x2, y2, x2 + 80, y2 + 140),
            "conf": 0.92,
            "class_id": 7,
            "class_name": "truck"
        })
        cv2.rectangle(frame, (x2, y2), (x2 + 80, y2 + 140), (50, 150, 220), -1)

    return frame, detections


def run_pipeline(args):
    is_demo = args.source.lower() == "demo"
    cap = None

    if not is_demo:
        source_val = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {args.source}")
            sys.exit(1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        width, height, fps = 1280, 720, 30.0
        print("[INFO] Running in synthetic DEMO mode...")

    # Calculate tripwire position
    line_y = int(height * args.line_ratio)

    detector = VehicleDetector(conf_thresh=args.conf)
    tracker = CentroidTracker(max_disappeared=args.max_disappeared, max_distance=args.max_distance)
    counter = TrafficCounter(line_y=line_y)
    speed_estimator = SpeedEstimator(pixels_per_meter=args.ppm, fps=fps)
    road_estimator = RoadWidthEstimator(
        pixels_per_meter=args.ppm,
        standard_lane_width_m=args.lane_width
    )
    traffic_analytics = TrafficAnalytics(pixels_per_meter=args.ppm)
    visualizer = Visualizer()

    if args.road_left is not None and args.road_right is not None:
        road_estimator.set_horizontal_boundaries(args.road_left, args.road_right, line_y)
    elif is_demo:
        road_x1, road_x2 = int(width * 0.25), int(width * 0.75)
        road_estimator.set_horizontal_boundaries(road_x1, road_x2, line_y)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"[INFO] Initialized Traffic Flow Analyzer | Resolution: {width}x{height} | Line Y: {line_y}")

    frame_idx = 0
    latest_los = {}
    try:
        while True:
            if is_demo:
                frame, detections = generate_synthetic_frame(frame_idx, width, height)
                time.sleep(1.0 / fps)
                if args.max_frames and frame_idx >= args.max_frames:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detector.detect(frame)

                # Attempt automatic road edge detection on first frame if not set
                if frame_idx == 0 and road_estimator.p1 is None:
                    road_estimator.detect_road_edges_from_frame(frame, line_y)

            # Refine road width estimate from vehicle bounding boxes if needed
            boxes = [d["box"] for d in detections]
            road_estimator.update_from_vehicle_boxes(boxes, line_y)

            # Update tracker
            tracked_objects = tracker.update(detections)

            # Update speeds
            speeds = {}
            for obj_id, data in tracked_objects.items():
                speeds[obj_id] = speed_estimator.estimate_speed(obj_id, data["trajectory"])

            # Tripwire counting
            inbound, outbound, counts_by_class = counter.process_objects(tracked_objects)

            # Current road width metrics
            road_metrics = road_estimator.calculate_width()

            # Level of Service (LOS) & Roadway Occupancy
            latest_los = traffic_analytics.compute_level_of_service(
                tracked_objects=tracked_objects,
                road_width_m=road_metrics["width_meters"],
                frame_height_px=height,
                current_speeds=speeds
            )

            # Detect safety anomalies (wrong-way driving, stalled vehicles)
            alerts = traffic_analytics.detect_anomalies(
                tracked_objects=tracked_objects,
                current_speeds=speeds,
                frame_idx=frame_idx,
                frame_width=width
            )

            # Render visuals
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
                los_data=latest_los,
                alerts=alerts
            )

            if writer:
                writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow("Traffic Flow Analyzer", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[INFO] Exiting on user request...")
                    break

            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"[STATUS] Frames: {frame_idx} | Total: {counter.total_count} | LOS: {latest_los.get('los_grade')} ({latest_los.get('occupancy_pct')}%)")

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        road_summary = road_estimator.calculate_width()
        print("\n=== FINAL TRAFFIC & ROAD SUMMARY ===")
        print(f"Total Vehicles Counted: {counter.total_count}")
        print(f"Inbound Flow:  {counter.inbound_count}")
        print(f"Outbound Flow: {counter.outbound_count}")
        print(f"Breakdown by Class: {counter.counts_by_class}")
        if road_summary["width_meters"] > 0:
            print(f"Road Width: {road_summary['width_meters']} m ({road_summary['width_feet']} ft)")
            print(f"Estimated Lanes: {road_summary['estimated_lanes']} (Avg: {road_summary['avg_lane_width_m']} m/lane)")
        else:
            print("Road Width: Not calibrated")
        if latest_los:
            print(f"Final Level of Service: LOS {latest_los.get('los_grade')} ({latest_los.get('los_description')})")
            print(f"Corridor Occupancy: {latest_los.get('occupancy_pct')}% | Avg Corridor Speed: {latest_los.get('avg_speed_kmh')} km/h")
        if traffic_analytics.incident_log:
            print(f"Safety Incidents Logged: {len(traffic_analytics.incident_log)}")
            for inc in traffic_analytics.incident_log[-5:]:
                print(f"  - [Frame {inc['frame_idx']}] {inc['type']} on vehicle #{inc['obj_id']}")


def main():
    parser = argparse.ArgumentParser(description="Traffic Flow Analyzer - Intelligent Vehicle Monitoring")
    parser.add_argument("--source", type=str, default="demo", help="Path to video file, webcam index, RTSP stream URL, or 'demo'")
    parser.add_argument("--output", type=str, default=None, help="Optional output MP4 path")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--line-ratio", type=float, default=0.6, help="Counting tripwire position ratio along vertical axis")
    parser.add_argument("--ppm", type=float, default=15.0, help="Pixels per meter calibration for speed & road width")
    parser.add_argument("--lane-width", type=float, default=3.5, help="Standard lane width in meters for lane count estimation")
    parser.add_argument("--road-left", type=int, default=None, help="Optional X pixel coordinate of left road boundary")
    parser.add_argument("--road-right", type=int, default=None, help="Optional X pixel coordinate of right road boundary")
    parser.add_argument("--max-disappeared", type=int, default=30, help="Frames allowed before track deregistration")
    parser.add_argument("--max-distance", type=float, default=80.0, help="Max distance px for track association")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for testing/demo)")
    parser.add_argument("--no-display", action="store_true", help="Run without opening GUI window")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

