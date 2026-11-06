"""
Centroid-based Multi-Object Tracker.
Assigns persistent IDs to detected vehicles and tracks their trajectories over time.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        """
        :param max_disappeared: Number of consecutive frames an object can be lost before deregistration.
        :param max_distance: Maximum pixel distance allowed between consecutive frames to match the same object.
        """
        self.next_object_id = 1
        self.objects: Dict[int, Tuple[int, int]] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()
        self.class_names: Dict[int, str] = OrderedDict()
        self.boxes: Dict[int, Tuple[int, int, int, int]] = OrderedDict()
        self.trajectories: Dict[int, List[Tuple[int, int]]] = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], box: Tuple[int, int, int, int], class_name: str) -> int:
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.class_names[object_id] = class_name
        self.boxes[object_id] = box
        self.trajectories[object_id] = [centroid]
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.class_names[object_id]
        del self.boxes[object_id]
        del self.trajectories[object_id]

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Updates the tracker with newly detected bounding boxes.
        Returns a dictionary of active objects:
        {
            object_id: {
                'centroid': (cx, cy),
                'box': (x1, y1, x2, y2),
                'class_name': str,
                'trajectory': [(cx1, cy1), ...]
            }
        }
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._build_result()

        input_centroids = np.zeros((len(detections), 2), dtype=int)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self.register(tuple(input_centroids[i]), det["box"], det.get("class_name", "vehicle"))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Compute Euclidean distance matrix between existing centroids and input centroids
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                new_centroid = tuple(input_centroids[col])
                self.objects[object_id] = new_centroid
                self.boxes[object_id] = detections[col]["box"]
                self.class_names[object_id] = detections[col].get("class_name", "vehicle")
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(new_centroid)

                # Keep trajectory bounded to last 50 points
                if len(self.trajectories[object_id]) > 50:
                    self.trajectories[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(tuple(input_centroids[col]), detections[col]["box"], detections[col].get("class_name", "vehicle"))

        return self._build_result()

    def _build_result(self) -> Dict[int, Dict]:
        res = {}
        for obj_id in self.objects:
            res[obj_id] = {
                "centroid": self.objects[obj_id],
                "box": self.boxes[obj_id],
                "class_name": self.class_names[obj_id],
                "trajectory": self.trajectories[obj_id]
            }
        return res

