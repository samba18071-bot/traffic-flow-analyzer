"""
Unit tests for the RoadWidthEstimator module.
"""

import pytest
import numpy as np
from traffic_monitor.road_width_estimator import RoadWidthEstimator


def test_initial_state():
    estimator = RoadWidthEstimator(pixels_per_meter=10.0)
    metrics = estimator.calculate_width()
    assert metrics["width_meters"] == 0.0
    assert metrics["pixel_distance"] == 0.0
    assert metrics["estimated_lanes"] == 0


def test_explicit_horizontal_boundaries():
    # 300 pixels at 15 px/meter -> 20 meters.
    # 20 meters / 3.5m standard lane = 5.71 -> 6 lanes.
    estimator = RoadWidthEstimator(pixels_per_meter=15.0, standard_lane_width_m=3.5)
    estimator.set_horizontal_boundaries(left_x=100, right_x=400, y=360)

    metrics = estimator.calculate_width()
    assert metrics["pixel_distance"] == 300.0
    assert metrics["width_meters"] == 20.0
    assert pytest.approx(metrics["width_feet"], rel=1e-2) == 65.62
    assert metrics["estimated_lanes"] == 6
    assert pytest.approx(metrics["avg_lane_width_m"], rel=1e-2) == 3.33


def test_diagonal_boundaries():
    # dx=300, dy=400 -> distance = 500 px
    # At 20 px/m -> 25 meters
    estimator = RoadWidthEstimator(pixels_per_meter=20.0)
    estimator.set_boundaries(p1=(100, 100), p2=(400, 500))

    metrics = estimator.calculate_width()
    assert metrics["pixel_distance"] == 500.0
    assert metrics["width_meters"] == 25.0


def test_update_from_vehicle_boxes():
    estimator = RoadWidthEstimator(pixels_per_meter=10.0, auto_refine_with_vehicles=True)
    boxes = [
        (200, 300, 260, 400),
        (500, 310, 580, 420),
    ]
    # Shoulder margin = 1m * 10 px/m = 10 px
    # min_x = 200 - 10 = 190, max_x = 580 + 10 = 590 -> width = 400 px -> 40 meters
    estimator.update_from_vehicle_boxes(boxes, frame_y=350)
    metrics = estimator.calculate_width()
    assert metrics["pixel_distance"] == 400.0
    assert metrics["width_meters"] == 40.0


def test_edge_detection_on_synthetic_frame():
    estimator = RoadWidthEstimator(pixels_per_meter=10.0)
    # Create synthetic frame with bright road edges
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    # Draw road between x=150 and x=450
    frame[:, 150:450] = 70
    # Road edges with high contrast
    frame[:, 148:152] = 255
    frame[:, 448:452] = 255

    res = estimator.detect_road_edges_from_frame(frame, scan_y=200)
    assert res is not None
    left_x, right_x = res
    assert left_x < right_x
    metrics = estimator.calculate_width()
    assert metrics["width_meters"] > 0

