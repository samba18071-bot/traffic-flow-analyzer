import unittest
from traffic_monitor.counting import TrafficCounter

class TestTrafficCounter(unittest.TestCase):
    def setUp(self):
        self.counter = TrafficCounter(line_y=300)

    def test_inbound_crossing(self):
        # Moving downwards across y=300
        tracked_objects = {
            1: {
                "trajectory": [(200, 280), (200, 310)],
                "class_name": "car"
            }
        }
        inbound, outbound, by_class = self.counter.process_objects(tracked_objects)
        self.assertEqual(inbound, 1)
        self.assertEqual(outbound, 0)
        self.assertEqual(self.counter.total_count, 1)
        self.assertEqual(by_class.get("car"), 1)

    def test_outbound_crossing(self):
        # Moving upwards across y=300
        tracked_objects = {
            2: {
                "trajectory": [(400, 320), (400, 290)],
                "class_name": "truck"
            }
        }
        inbound, outbound, by_class = self.counter.process_objects(tracked_objects)
        self.assertEqual(inbound, 0)
        self.assertEqual(outbound, 1)
        self.assertEqual(self.counter.total_count, 1)
        self.assertEqual(by_class.get("truck"), 1)

    def test_no_double_counting(self):
        # Object crosses line
        tracked_objects = {
            1: {
                "trajectory": [(200, 280), (200, 310)],
                "class_name": "car"
            }
        }
        self.counter.process_objects(tracked_objects)

        # Same object moves further down
        tracked_objects[1]["trajectory"].append((200, 340))
        inbound, outbound, _ = self.counter.process_objects(tracked_objects)
        self.assertEqual(inbound, 1)
        self.assertEqual(self.counter.total_count, 1)

if __name__ == "__main__":
    unittest.main()

