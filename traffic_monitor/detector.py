"""
Vehicle Detector module.
Wraps YOLOv8 model inference with class filtering and fallback handling.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.4, target_classes: List[int] = None):
        self.conf_thresh = conf_thresh
        self.target_classes = target_classes or [2, 3, 5, 7]
        self.model = None
        self._is_yolo_available = False

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self._is_yolo_available = True
        except (ImportError, Exception) as e:
            # Fallback for environments without ultralytics installed or model weight downloading
            self._is_yolo_available = False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single video frame.
        Returns a list of detections: [{'box': (x1, y1, x2, y2), 'conf': float, 'class_id': int, 'class_name': str}]
        """
        if not self._is_yolo_available or self.model is None:
            return []

        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in self.target_classes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].item())
                    detections.append({
                        "box": (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])),
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": VEHICLE_CLASS_NAMES.get(cls_id, f"vehicle_{cls_id}")
                    })

        return detections

