import unittest

from src.inference import FaceTracker


class FaceTrackerTests(unittest.TestCase):
    def test_ids_follow_faces_when_detection_order_changes(self):
        tracker = FaceTracker()
        first_frame = [(0, 0, 100, 100), (200, 0, 300, 100)]
        second_frame = [(205, 5, 305, 105), (5, 5, 105, 105)]

        self.assertEqual(tracker.update(first_frame), [1, 2])
        self.assertEqual(tracker.update(second_frame), [2, 1])

    def test_id_survives_ten_missed_frames(self):
        tracker = FaceTracker()
        box = (0, 0, 100, 100)

        self.assertEqual(tracker.update([box]), [1])
        for _ in range(10):
            self.assertEqual(tracker.update([]), [])

        self.assertEqual(tracker.update([(4, 4, 104, 104)]), [1])

    def test_id_expires_after_more_than_ten_missed_frames(self):
        tracker = FaceTracker()
        box = (0, 0, 100, 100)

        self.assertEqual(tracker.update([box]), [1])
        for _ in range(11):
            self.assertEqual(tracker.update([]), [])

        self.assertEqual(tracker.update([(4, 4, 104, 104)]), [2])

    def test_reset_restarts_id_sequence(self):
        tracker = FaceTracker()

        self.assertEqual(tracker.update([(0, 0, 100, 100)]), [1])
        tracker.reset()

        self.assertEqual(tracker.update([(0, 0, 100, 100)]), [1])


if __name__ == "__main__":
    unittest.main()
