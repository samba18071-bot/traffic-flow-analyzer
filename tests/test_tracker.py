import unittest
from traffic_monitor.tracker import CentroidTracker

class TestCentroidTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=50.0)

    def test_registration(self):
        detections = [{"box": (100, 100, 150, 150), "conf": 0.9, "class_name": "car"}]
        tracked = self.tracker.update(detections)
        self.assertEqual(len(tracked), 1)
        self.assertIn(1, tracked)
        self.assertEqual(tracked[1]["centroid"], (125, 125))

    def test_movement_and_trajectory(self):
        # Frame 1
        self.tracker.update([{"box": (100, 100, 150, 150), "conf": 0.9, "class_name": "car"}])
        # Frame 2: moved slightly
        tracked = self.tracker.update([{"box": (105, 105, 155, 155), "conf": 0.9, "class_name": "car"}])
        self.assertEqual(len(tracked), 1)
        self.assertEqual(len(tracked[1]["trajectory"]), 2)

    def test_deregistration_on_disappearance(self):
        self.tracker.update([{"box": (100, 100, 150, 150), "conf": 0.9, "class_name": "car"}])
        # Disappear for 6 frames
        for _ in range(6):
            tracked = self.tracker.update([])
        self.assertEqual(len(tracked), 0)

if __name__ == "__main__":
    unittest.main()

