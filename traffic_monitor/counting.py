"""
Tripwire Counting and Direction Analysis.
Monitors objects crossing predefined virtual counting lines or detection zones.
"""

from typing import Dict, Tuple, Set


class TrafficCounter:
    def __init__(self, line_y: int, direction_threshold_px: int = 15):
        """
        :param line_y: Vertical pixel coordinate representing the tripwire.
        :param direction_threshold_px: Minimum pixel travel required to confirm direction.
        """
        self.line_y = line_y
        self.direction_threshold_px = direction_threshold_px

        self.inbound_count = 0
        self.outbound_count = 0
        self.counts_by_class: Dict[str, int] = {}
        self.counted_ids: Set[int] = set()

    def process_objects(self, tracked_objects: Dict[int, Dict]) -> Tuple[int, int, Dict[str, int]]:
        """
        Evaluates tracked trajectories to detect tripwire crossings.
        """
        for obj_id, data in tracked_objects.items():
            if obj_id in self.counted_ids:
                continue

            traj = data["trajectory"]
            if len(traj) < 2:
                continue

            prev_y = traj[-2][1]
            curr_y = traj[-1][1]
            class_name = data.get("class_name", "unknown")

            # Check if crossing the line from above (moving downward / inbound)
            if prev_y < self.line_y <= curr_y:
                self.inbound_count += 1
                self.counts_by_class[class_name] = self.counts_by_class.get(class_name, 0) + 1
                self.counted_ids.add(obj_id)

            # Check if crossing the line from below (moving upward / outbound)
            elif prev_y > self.line_y >= curr_y:
                self.outbound_count += 1
                self.counts_by_class[class_name] = self.counts_by_class.get(class_name, 0) + 1
                self.counted_ids.add(obj_id)

        return self.inbound_count, self.outbound_count, self.counts_by_class

    @property
    def total_count(self) -> int:
        return self.inbound_count + self.outbound_count
