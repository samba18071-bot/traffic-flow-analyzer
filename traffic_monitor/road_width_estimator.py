"""
Road Width and Geometry Estimation module.
Calculates road width in pixels and real-world units (meters and feet),
estimates lane counts, and provides engineering dimension calipers.
"""

from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import cv2


class RoadWidthEstimator:
    """
    Estimates road width and lane layout from calibrated camera geometry,
    explicit cross-section points, computer vision edge analysis, or vehicle envelope.
    """

    def __init__(
        self,
        pixels_per_meter: float = 15.0,
        standard_lane_width_m: float = 3.5,
        boundary_p1: Optional[Tuple[int, int]] = None,
        boundary_p2: Optional[Tuple[int, int]] = None,
        auto_refine_with_vehicles: bool = True
    ):
        """
        :param pixels_per_meter: Spatial calibration ratio (pixels / meter).
        :param standard_lane_width_m: Typical standard highway/urban lane width (default 3.5 m).
        :param boundary_p1: (x, y) coordinates of left road boundary.
        :param boundary_p2: (x, y) coordinates of right road boundary.
        :param auto_refine_with_vehicles: Whether to refine road boundary extents as vehicles pass.
        """
        self.ppm = max(pixels_per_meter, 1e-4)
        self.standard_lane_width_m = standard_lane_width_m
        self.p1 = boundary_p1
        self.p2 = boundary_p2
        self.auto_refine = auto_refine_with_vehicles

        # Envelope tracking from vehicle detections
        self.min_vehicle_x: Optional[int] = None
        self.max_vehicle_x: Optional[int] = None
        self.measured_y: Optional[int] = None

    def set_boundaries(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """Sets the explicit left and right boundary coordinates of the road."""
        self.p1 = (int(p1[0]), int(p1[1]))
        self.p2 = (int(p2[0]), int(p2[1]))
        self.measured_y = int((self.p1[1] + self.p2[1]) / 2)

    def set_horizontal_boundaries(self, left_x: int, right_x: int, y: int):
        """Sets horizontal road boundaries at a specific Y scanline (e.g. tripwire line)."""
        self.p1 = (int(left_x), int(y))
        self.p2 = (int(right_x), int(y))
        self.measured_y = int(y)

    def update_from_vehicle_boxes(self, boxes: List[Tuple[int, int, int, int]], frame_y: int):
        """
        Refines empirical carriageway width based on the spatial envelope
        of observed vehicle trajectories with a safe shoulder margin.
        """
        if not self.auto_refine or not boxes:
            return

        for x1, y1, x2, y2 in boxes:
            if self.min_vehicle_x is None or x1 < self.min_vehicle_x:
                self.min_vehicle_x = x1
            if self.max_vehicle_x is None or x2 > self.max_vehicle_x:
                self.max_vehicle_x = x2

        # If explicit boundaries were not set, use vehicle envelope + 1.0m shoulder margin
        if self.p1 is None or self.p2 is None:
            shoulder_margin_px = int(1.0 * self.ppm)
            left_x = max(0, self.min_vehicle_x - shoulder_margin_px)
            right_x = self.max_vehicle_x + shoulder_margin_px
            self.set_horizontal_boundaries(left_x, right_x, frame_y)

    def detect_road_edges_from_frame(
        self,
        frame: np.ndarray,
        scan_y: int,
        search_margin_ratio: float = 0.05
    ) -> Optional[Tuple[int, int]]:
        """
        Uses edge detection and gradient intensity along a horizontal transect
        to locate road curbs/shoulders if explicit boundaries are not provided.
        """
        height, width = frame.shape[:2]
        scan_y = int(np.clip(scan_y, 0, height - 1))

        # Sample strip around scanline
        half_band = max(3, int(height * search_margin_ratio))
        y_min = max(0, scan_y - half_band)
        y_max = min(height, scan_y + half_band)

        roi = frame[y_min:y_max, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Collapse vertically to get horizontal edge profile
        edge_profile = np.sum(edges, axis=0)

        # Look for prominent peaks in the left third and right third of frame
        left_region = edge_profile[: width // 2]
        right_region = edge_profile[width // 2 :]

        if len(left_region) > 0 and len(right_region) > 0:
            left_peaks = np.where(left_region > np.mean(left_region) + np.std(left_region))[0]
            right_peaks = np.where(right_region > np.mean(right_region) + np.std(right_region))[0]

            left_x = int(left_peaks[0]) if len(left_peaks) > 0 else int(width * 0.2)
            right_x = int(width // 2 + right_peaks[-1]) if len(right_peaks) > 0 else int(width * 0.8)

            if right_x > left_x + 50:
                self.set_horizontal_boundaries(left_x, right_x, scan_y)
                return (left_x, right_x)

        return None

    def calculate_width(self) -> Dict[str, Any]:
        """
        Computes road width metrics including pixel distance, meters, feet,
        and estimated lane capacity.
        """
        if self.p1 is None or self.p2 is None:
            return {
                "pixel_distance": 0.0,
                "width_meters": 0.0,
                "width_feet": 0.0,
                "estimated_lanes": 0,
                "avg_lane_width_m": 0.0,
                "p1": (0, 0),
                "p2": (0, 0),
            }

        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        pixel_distance = float(np.sqrt(dx * dx + dy * dy))

        width_meters = pixel_distance / self.ppm
        width_feet = width_meters * 3.28084

        estimated_lanes = max(1, int(round(width_meters / self.standard_lane_width_m)))
        avg_lane_width = width_meters / estimated_lanes if estimated_lanes > 0 else 0.0

        return {
            "pixel_distance": round(pixel_distance, 1),
            "width_meters": round(width_meters, 2),
            "width_feet": round(width_feet, 2),
            "estimated_lanes": estimated_lanes,
            "avg_lane_width_m": round(avg_lane_width, 2),
            "p1": self.p1,
            "p2": self.p2,
        }

    def draw_measurement(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 165, 255),
        offset_y: int = -28
    ) -> np.ndarray:
        """
        Renders an engineering-style dimension caliper with arrows and measurement text.
        """
        metrics = self.calculate_width()
        if metrics["width_meters"] <= 0:
            return frame

        p1 = metrics["p1"]
        p2 = metrics["p2"]
        width_m = metrics["width_meters"]
        width_ft = metrics["width_feet"]
        lanes = metrics["estimated_lanes"]

        # Shift caliper line slightly above the actual road line for clean visualization
        y_meas = int(p1[1] + offset_y)
        x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])

        # Vertical end markers (curb ticks)
        tick_h = 10
        cv2.line(frame, (x1, y_meas - tick_h), (x1, y_meas + tick_h), color, 2)
        cv2.line(frame, (x2, y_meas - tick_h), (x2, y_meas + tick_h), color, 2)

        # Dimension line with double arrowheads
        cv2.arrowedLine(frame, ((x1 + x2) // 2, y_meas), (x1, y_meas), color, 2, tipLength=0.04)
        cv2.arrowedLine(frame, ((x1 + x2) // 2, y_meas), (x2, y_meas), color, 2, tipLength=0.04)

        # Measurement Label Badge
        label = f"ROAD WIDTH: {width_m:.2f}m ({width_ft:.1f}ft) | ~{lanes} Lanes"
        (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

        center_x = (x1 + x2) // 2
        badge_x1 = center_x - w_txt // 2 - 8
        badge_y1 = y_meas - h_txt - 10
        badge_x2 = center_x + w_txt // 2 + 8
        badge_y2 = y_meas - 2

        # Draw semi-transparent badge background
        overlay = frame.copy()
        cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)

        cv2.putText(
            frame,
            label,
            (badge_x1 + 8, y_meas - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return frame

