"""
Unit tests for TrafficAnalytics and Level of Service (LOS) calculations.
"""

from traffic_monitor.traffic_analytics import TrafficAnalytics


def test_los_empty_traffic():
    analytics = TrafficAnalytics(pixels_per_meter=10.0)
    res = analytics.compute_level_of_service(
        tracked_objects={},
        road_width_m=20.0,
        frame_height_px=500,
        current_speeds={}
    )
    assert res["los_grade"] == "A"
    assert res["occupancy_pct"] == 0.0


def test_los_free_flow():
    analytics = TrafficAnalytics(pixels_per_meter=10.0)
    # 1 car: 2m x 4m = 8 m2 on road 20m x 50m = 1000 m2 -> 0.8% occupancy
    tracked_objects = {
        1: {
            "box": (100, 100, 120, 140),  # 20px x 40px -> 2m x 4m
            "trajectory": [(110, 100), (110, 120)]
        }
    }
    speeds = {1: 65.0}
    res = analytics.compute_level_of_service(
        tracked_objects=tracked_objects,
        road_width_m=20.0,
        frame_height_px=500,
        current_speeds=speeds
    )
    assert res["los_grade"] == "A"
    assert res["occupancy_pct"] < 5.0
    assert res["avg_speed_kmh"] == 65.0


def test_los_congested():
    analytics = TrafficAnalytics(pixels_per_meter=10.0)
    # Simulate high density / heavy vehicle coverage
    tracked_objects = {}
    speeds = {}
    for i in range(25):
        tracked_objects[i] = {
            "box": (100, i * 20, 250, i * 20 + 20),
            "trajectory": [(100, i * 20)]
        }
        speeds[i] = 12.0

    res = analytics.compute_level_of_service(
        tracked_objects=tracked_objects,
        road_width_m=15.0,
        frame_height_px=500,
        current_speeds=speeds
    )
    assert res["los_grade"] in ["E", "F"]
    assert res["occupancy_pct"] > 35.0


def test_stalled_vehicle_detection():
    analytics = TrafficAnalytics(pixels_per_meter=10.0, stalled_threshold_frames=5, speed_stalled_kmh=2.0)
    obj_data = {
        1: {
            "box": (200, 200, 250, 280),
            "trajectory": [(225, 240)] * 12
        }
    }
    speeds = {1: 0.0}

    # Run for 6 frames at 0 km/h
    alerts = []
    for f in range(6):
        alerts = analytics.detect_anomalies(obj_data, speeds, frame_idx=f, frame_width=1000)

    assert len(alerts) == 1
    assert alerts[0]["type"] == "STALLED_VEHICLE"
    assert alerts[0]["obj_id"] == 1


def test_wrong_way_detection():
    analytics = TrafficAnalytics(pixels_per_meter=10.0)
    # Left lane (cx < 500) moving strongly UP (-dy)
    obj_data = {
        2: {
            "box": (200, 200, 260, 290),
            "trajectory": [(230, 400 - i * 10) for i in range(10)]
        }
    }
    speeds = {2: 45.0}
    alerts = analytics.detect_anomalies(obj_data, speeds, frame_idx=10, frame_width=1000)
    assert len(alerts) == 1
    assert alerts[0]["type"] == "WRONG_WAY"
    assert alerts[0]["obj_id"] == 2

