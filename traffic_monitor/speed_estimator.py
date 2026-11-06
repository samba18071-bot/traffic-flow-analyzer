"""
Vehicle Velocity and Speed Estimation module.
Converts pixel displacement over time into real-world speeds (km/h).
"""

from typing import Dict, Tuple
import numpy as np


class SpeedEstimator:
    def __init__(self, pixels_per_meter: float = 15.0, fps: float = 30.0):
        """
        :param pixels_per_meter: Spatial calibration ratio (pixels / meter).
        :param fps: Video frame rate.
        """
        self.ppm = pixels_per_meter
        self.fps = fps
        self.speeds: Dict[int, float] = {}

    def estimate_speed(self, obj_id: int, trajectory: list) -> float:
        """
        Calculates instantaneous speed over recent trajectory points in km/h.
        """
        if len(trajectory) < 5:
            return self.speeds.get(obj_id, 0.0)

        # Compute displacement between point 5 frames ago and current frame
        p_start = np.array(trajectory[-5])
        p_end = np.array(trajectory[-1])
        pixel_distance = np.linalg.norm(p_end - p_start)

        meters = pixel_distance / self.ppm
        time_seconds = 4.0 / self.fps

        if time_seconds > 0:
            speed_mps = meters / time_seconds
            speed_kmh = speed_mps * 3.6
        else:
            speed_kmh = 0.0

        # Exponential moving average filter for smooth readings
        if obj_id in self.speeds:
            smoothed_speed = 0.7 * self.speeds[obj_id] + 0.3 * speed_kmh
        else:
            smoothed_speed = speed_kmh

        self.speeds[obj_id] = round(smoothed_speed, 1)
        return self.speeds[obj_id]
