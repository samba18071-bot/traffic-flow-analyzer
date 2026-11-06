"""
Traffic Analytics & Safety Anomaly Detection module.
Computes Roadway Occupancy, Level of Service (LOS A-F), Wrong-Way Driving,
and Stalled / Stopped Vehicle detection.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class TrafficAnalytics:
    """
    Computes traffic density, Level of Service (LOS A to F per Highway Capacity Manual),
    and detects traffic anomalies including wrong-way driving and stalled vehicles.
    """

    def __init__(
        self,
        pixels_per_meter: float = 15.0,
        stalled_threshold_frames: int = 45,
        speed_stalled_kmh: float = 3.0,
    ):
        """
        :param pixels_per_meter: Spatial scale calibration (px / m).
        :param stalled_threshold_frames: Consecutive frames below speed threshold to flag stalled vehicle.
        :param speed_stalled_kmh: Maximum speed considered stationary (km/h).
        """
        self.ppm = max(pixels_per_meter, 1e-4)
        self.stalled_threshold_frames = stalled_threshold_frames
        self.speed_stalled_kmh = speed_stalled_kmh

        # Track stationary history: obj_id -> count of stationary frames
        self.stationary_counters: Dict[int, int] = {}
        # Track active incident alerts
        self.active_alerts: List[Dict[str, Any]] = []
        # Persistent incident history log
        self.incident_log: List[Dict[str, Any]] = []

    def compute_level_of_service(
        self,
        tracked_objects: Dict[int, Dict],
        road_width_m: float,
        frame_height_px: int,
        current_speeds: Dict[int, float]
    ) -> Dict[str, Any]:
        """
        Calculates Roadway Occupancy Percentage and Level of Service (LOS A to F).
        """
        if road_width_m <= 0 or not tracked_objects:
            return {
                "occupancy_pct": 0.0,
                "los_grade": "A",
                "los_description": "Free Flow",
                "los_color": (0, 255, 0),  # Green
                "avg_speed_kmh": 0.0,
                "active_vehicle_count": 0,
            }

        # Road section area in square meters
        road_length_m = frame_height_px / self.ppm
        road_area_m2 = max(1.0, road_width_m * road_length_m)

        # Compute total vehicle footprint area
        total_vehicle_area_m2 = 0.0
        speeds_list = []

        for obj_id, data in tracked_objects.items():
            x1, y1, x2, y2 = data["box"]
            width_m = (x2 - x1) / self.ppm
            length_m = (y2 - y1) / self.ppm
            total_vehicle_area_m2 += width_m * length_m

            speed = current_speeds.get(obj_id, 0.0)
            if speed > 0:
                speeds_list.append(speed)

        occupancy_pct = min(100.0, (total_vehicle_area_m2 / road_area_m2) * 100.0)
        avg_speed = float(np.mean(speeds_list)) if speeds_list else 0.0

        # Determine LOS based on occupancy and average speed (Highway Capacity Manual)
        if occupancy_pct < 10.0 and (avg_speed >= 50.0 or avg_speed == 0.0):
            los_grade = "A"
            description = "Free Flow"
            color = (0, 255, 0)        # Bright Green
        elif occupancy_pct < 18.0 and avg_speed >= 40.0:
            los_grade = "B"
            description = "Reasonably Free"
            color = (50, 220, 100)     # Light Green
        elif occupancy_pct < 27.0 and avg_speed >= 30.0:
            los_grade = "C"
            description = "Stable Flow"
            color = (0, 255, 255)      # Yellow
        elif occupancy_pct < 38.0 and avg_speed >= 22.0:
            los_grade = "D"
            description = "Approaching Capacity"
            color = (0, 140, 255)      # Amber / Orange
        elif occupancy_pct < 50.0 and avg_speed >= 15.0:
            los_grade = "E"
            description = "Unstable / Near Jam"
            color = (0, 80, 255)       # Deep Orange
        else:
            los_grade = "F"
            description = "Forced Flow / Gridlock"
            color = (0, 0, 255)        # Red

        return {
            "occupancy_pct": round(occupancy_pct, 1),
            "los_grade": los_grade,
            "los_description": description,
            "los_color": color,
            "avg_speed_kmh": round(avg_speed, 1),
            "active_vehicle_count": len(tracked_objects),
        }

    def detect_anomalies(
        self,
        tracked_objects: Dict[int, Dict],
        current_speeds: Dict[int, float],
        frame_idx: int,
        frame_width: int
    ) -> List[Dict[str, Any]]:
        """
        Scans active tracks for:
        1. Wrong-Way Driving (e.g. Inbound traffic moving upward or Outbound moving downward).
        2. Stalled / Stationary Vehicles on the road carriageway.
        """
        current_alerts = []
        center_x = frame_width // 2

        # Cleanup stationary counters for lost objects
        active_ids = set(tracked_objects.keys())
        self.stationary_counters = {k: v for k, v in self.stationary_counters.items() if k in active_ids}

        for obj_id, data in tracked_objects.items():
            traj = data.get("trajectory", [])
            speed = current_speeds.get(obj_id, 0.0)
            box = data["box"]
            cx = (box[0] + box[2]) // 2

            # 1. Stalled Vehicle Detection
            if len(traj) >= 10:
                if speed < self.speed_stalled_kmh:
                    self.stationary_counters[obj_id] = self.stationary_counters.get(obj_id, 0) + 1
                else:
                    self.stationary_counters[obj_id] = max(0, self.stationary_counters.get(obj_id, 0) - 2)

                if self.stationary_counters.get(obj_id, 0) >= self.stalled_threshold_frames:
                    alert = {
                        "obj_id": obj_id,
                        "type": "STALLED_VEHICLE",
                        "box": box,
                        "message": f"ALERT: Vehicle #{obj_id} Stalled / Stopped",
                        "frame_idx": frame_idx
                    }
                    current_alerts.append(alert)
                    self._log_incident(alert)

            # 2. Wrong-Way Driving Detection
            # Compare directional displacement across recent trajectory
            if len(traj) >= 8:
                dy = traj[-1][1] - traj[-8][1]
                # In standard roadway layouts:
                # Left side of road (x < center_x): expected Inbound (moving downwards, dy > 0)
                # Right side of road (x >= center_x): expected Outbound (moving upwards, dy < 0)
                is_wrong_way = False
                if cx < center_x and dy < -25:  # Moving strongly UP on left side
                    is_wrong_way = True
                elif cx >= center_x and dy > 25:  # Moving strongly DOWN on right side
                    is_wrong_way = True

                if is_wrong_way:
                    alert = {
                        "obj_id": obj_id,
                        "type": "WRONG_WAY",
                        "box": box,
                        "message": f"CRITICAL: Vehicle #{obj_id} Wrong-Way Driving!",
                        "frame_idx": frame_idx
                    }
                    current_alerts.append(alert)
                    self._log_incident(alert)

        self.active_alerts = current_alerts
        return current_alerts

    def _log_incident(self, alert: Dict[str, Any]):
        """Avoid duplicate consecutive spamming of the same incident log."""
        for logged in self.incident_log[-10:]:
            if logged["obj_id"] == alert["obj_id"] and logged["type"] == alert["type"]:
                if alert["frame_idx"] - logged["frame_idx"] < 60:
                    return
        self.incident_log.append(alert)
