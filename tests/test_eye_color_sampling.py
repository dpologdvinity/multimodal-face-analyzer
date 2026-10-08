"""Test iris sampling without relying on photograph-specific colors or coordinates."""
import unittest

import cv2
import numpy as np

from face_analyzer import inference


class NoEyes:
    def detectMultiScale(self, *args, **kwargs):
        return ()


class EyeColorSamplingTests(unittest.TestCase):
    def make_eye(self, color):
        image = np.full((100, 100, 3), (160, 180, 220), dtype=np.uint8)
        points = [(0.5, 0.5)] * 478
        for index, cx in ((468, 30), (473, 70)):
            cv2.circle(image, (cx, 45), 10, color, -1)
            cv2.circle(image, (cx, 45), 3, (0, 0, 0), -1)
            points[index:index + 5] = [(cx / 100, .45), ((cx + 10) / 100, .45),
                                       (cx / 100, .35), ((cx - 10) / 100, .45), (cx / 100, .55)]
        return image, points

    def test_dark_blue_iris_is_not_brown_due_to_pupil_or_shadow(self):
        image, points = self.make_eye((45, 20, 10))
        self.assertEqual(inference.predict_eye_color_colorimetric(NoEyes(), image, points), "blue")

    def test_dark_brown_iris_stays_brown(self):
        image, points = self.make_eye((10, 20, 40))
        self.assertEqual(inference.predict_eye_color_colorimetric(NoEyes(), image, points), "brown")

    def test_missing_or_invalid_landmarks_fall_back_safely(self):
        image, _ = self.make_eye((45, 20, 10))
        for points in (None, [], [(float("nan"), .5)] * 478):
            with self.subTest(points=bool(points)):
                self.assertEqual(inference.predict_eye_color_colorimetric(NoEyes(), image, points), "unknown")

    def test_iris_outside_frame_cannot_wrap_around_array(self):
        image, points = self.make_eye((45, 20, 10))
        points[468:] = [(-1, -1)] * 10
        self.assertEqual(inference.predict_eye_color_colorimetric(NoEyes(), image, points), "unknown")


if __name__ == "__main__":
    unittest.main()
