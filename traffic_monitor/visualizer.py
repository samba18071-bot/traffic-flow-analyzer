"""
Visualizer and Heads-Up Display (HUD) renderer.
Renders trajectories, bounding boxes, speed indicators, tripwire lines, and telemetry.
"""

from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np


class Visualizer:
    def __init__(self, line_color: Tuple[int, int, int] = (0, 255, 255), box_color: Tuple[int, int, int] = (0, 255, 0)):
        self.line_color = line_color
        self.box_color = box_color

    def draw_scene(
        self,
        frame: np.ndarray,
        tracked_objects: Dict[int, Dict],
        line_y: int,
        inbound_count: int,
        outbound_count: int,
        counts_by_class: Dict[str, int],
        speeds: Dict[int, float],
        road_estimator: Optional[Any] = None,
        road_metrics: Optional[Dict[str, Any]] = None,
        los_data: Optional[Dict[str, Any]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        output_frame = frame.copy()
        height, width = output_frame.shape[:2]

        # Draw road width measurement caliper if road estimator is provided
        if road_estimator is not None:
            output_frame = road_estimator.draw_measurement(output_frame, offset_y=-25)

        # Draw tripwire counting line
        cv2.line(output_frame, (0, line_y), (width, line_y), self.line_color, 2)
        cv2.putText(
            output_frame,
            "COUNTING TRIPWIRE",
            (20, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.line_color,
            2,
            cv2.LINE_AA
        )

        alert_map = {a["obj_id"]: a for a in (alerts or [])}

        # Draw tracked objects
        for obj_id, data in tracked_objects.items():
            x1, y1, x2, y2 = data["box"]
            cls_name = data.get("class_name", "vehicle")
            traj = data.get("trajectory", [])
            speed = speeds.get(obj_id, 0.0)

            has_alert = obj_id in alert_map
            current_box_color = (0, 0, 255) if has_alert else self.box_color

            # Draw trajectory path
            traj_color = (0, 0, 255) if has_alert else (255, 100, 0)
            for j in range(1, len(traj)):
                cv2.line(output_frame, traj[j - 1], traj[j], traj_color, 2)

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), current_box_color, 3 if has_alert else 2)

            # Draw label banner
            label = f"ID:{obj_id} {cls_name}"
            if speed > 0:
                label += f" | {speed} km/h"

            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_frame, (x1, y1 - 22), (x1 + w_txt + 10, y1), current_box_color, -1)
            text_color = (255, 255, 255) if has_alert else (0, 0, 0)
            cv2.putText(output_frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # If vehicle has an active alert, render warning badge above
            if has_alert:
                alert_type = alert_map[obj_id]["type"]
                alert_badge = f"! {alert_type.replace('_', ' ')} !"
                (w_ab, _), _ = cv2.getTextSize(alert_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(output_frame, (x1, y1 - 42), (x1 + w_ab + 10, y1 - 24), (0, 0, 220), -1)
                cv2.putText(output_frame, alert_badge, (x1 + 5, y1 - 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw incident banners in top-right corner if any active
        if alerts:
            self._render_alerts(output_frame, alerts, width)

        # Draw modern HUD overlay banner in top-left corner
        self._render_hud(output_frame, inbound_count, outbound_count, counts_by_class, road_metrics, los_data)

        return output_frame

    def _render_alerts(self, frame: np.ndarray, alerts: List[Dict[str, Any]], width: int):
        y_offset = 20
        for alert in alerts[:3]:
            msg = alert["message"]
            (w, h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x_start = width - w - 30
            # Red warning banner
            cv2.rectangle(frame, (x_start, y_offset), (width - 15, y_offset + h + 14), (0, 0, 180), -1)
            cv2.rectangle(frame, (x_start, y_offset), (width - 15, y_offset + h + 14), (0, 0, 255), 2)
            cv2.putText(frame, msg, (x_start + 8, y_offset + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += h + 22

    def _render_hud(
        self,
        frame: np.ndarray,
        inbound: int,
        outbound: int,
        counts_by_class: Dict[str, int],
        road_metrics: Optional[Dict[str, Any]] = None,
        los_data: Optional[Dict[str, Any]] = None
    ):
        hud_w = 350
        hud_h = 135
        if road_metrics and road_metrics.get("width_meters", 0) > 0:
            hud_h += 24
        if los_data and los_data.get("los_grade"):
            hud_h += 26

        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border
        cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (100, 100, 100), 1)

        # HUD Content
        cv2.putText(frame, "TRAFFIC FLOW MONITOR", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Total Vehicles: {inbound + outbound}", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Inbound:  {inbound}   |   Outbound: {outbound}", (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)

        breakdown = " | ".join([f"{k}:{v}" for k, v in counts_by_class.items()]) if counts_by_class else "Awaiting traffic..."
        cv2.putText(frame, f"Types: {breakdown}", (25, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        curr_y = 136
        if road_metrics and road_metrics.get("width_meters", 0) > 0:
            w_m = road_metrics["width_meters"]
            w_ft = road_metrics["width_feet"]
            lanes = road_metrics["estimated_lanes"]
            lane_w = road_metrics.get("avg_lane_width_m", 0.0)
            road_txt = f"Road: {w_m:.1f}m ({w_ft:.1f}ft) | {lanes}L (~{lane_w:.1f}m/lane)"
            cv2.putText(frame, road_txt, (25, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 215, 255), 1, cv2.LINE_AA)
            curr_y += 24

        if los_data and los_data.get("los_grade"):
            grade = los_data["los_grade"]
            desc = los_data.get("los_description", "")
            occ = los_data.get("occupancy_pct", 0.0)
            color = los_data.get("los_color", (0, 255, 0))
            los_txt = f"LOS {grade} ({desc}) | Occ: {occ}%"
            cv2.putText(frame, los_txt, (25, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


