"""
Visualizer and Heads-Up Display (HUD) renderer.
Renders trajectories, bounding boxes, speed indicators, tripwire lines, and telemetry.
"""

from typing import Dict, Tuple
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
        speeds: Dict[int, float]
    ) -> np.ndarray:
        output_frame = frame.copy()
        height, width = output_frame.shape[:2]

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

        # Draw tracked objects
        for obj_id, data in tracked_objects.items():
            x1, y1, x2, y2 = data["box"]
            cls_name = data.get("class_name", "vehicle")
            traj = data.get("trajectory", [])
            speed = speeds.get(obj_id, 0.0)

            # Draw trajectory path
            for j in range(1, len(traj)):
                cv2.line(output_frame, traj[j - 1], traj[j], (255, 100, 0), 2)

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), self.box_color, 2)

            # Draw label banner
            label = f"ID:{obj_id} {cls_name}"
            if speed > 0:
                label += f" | {speed} km/h"

            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_frame, (x1, y1 - 22), (x1 + w_txt + 10, y1), self.box_color, -1)
            cv2.putText(output_frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw modern HUD overlay banner in top-left corner
        self._render_hud(output_frame, inbound_count, outbound_count, counts_by_class)

        return output_frame

    def _render_hud(self, frame: np.ndarray, inbound: int, outbound: int, counts_by_class: Dict[str, int]):
        hud_w, hud_h = 320, 140
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
        cv2.putText(frame, f"Types: {breakdown}", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
