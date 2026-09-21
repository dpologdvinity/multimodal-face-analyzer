import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src import inference


class RecordingNet:
    def setInput(self, blob):
        self.blob = blob.copy()

    def forward(self):
        return np.array([[0.1, 0.9, 0.31]], dtype=np.float32)


class InsightFacePreprocessingTests(unittest.TestCase):
    def test_raw_rgb_input_has_reference_bbox_margin(self):
        y, x = np.mgrid[:300, :320]
        frame = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        box = (40, 60, 140, 200)
        net = RecordingNet()
        self.assertEqual(inference.predict_age_insightface(net, frame, box), "31")
        scale = 96 / (140 * 1.5)
        matrix = np.array([[scale, 0, 48 - 90 * scale], [0, scale, 48 - 130 * scale]])
        expected = cv2.warpAffine(frame, matrix, (96, 96))
        np.testing.assert_array_equal(net.blob, expected[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32))

    def test_analyze_uses_original_frame_despite_other_crop_rotation(self):
        y, x = np.mgrid[:320, :320]
        frame = np.stack((x % 256, y % 256, (x+y) % 256), axis=-1).astype(np.uint8)
        box = (80, 90, 210, 250)
        net = RecordingNet()
        inference.predict_age_insightface(net, frame, box)
        expected = net.blob.copy()
        models = inference.Models(face_net=None, age_nets={"insightface": net},
                                  gender_nets={"insightface": net},
                                  eye_color_nets={"colorimetric": object()})
        with patch.object(inference, "detect_faces", return_value=[box]), \
             patch.object(inference, "_estimate_roll_angle", return_value=20):
            output = inference.analyze_frame(
                models, frame, 0.5, active_age={"insightface"}, active_gender={"insightface"},
                active_emotion=set(), active_race=set(), active_recognition=set(), gallery={},
                active_glasses=set(), active_mask=set(), active_hair_color=set(), active_eye_color=set(),
                active_face_landmarks=set(), active_hands=set(), active_gaze=set(),
                global_adjustments={}, face_adjustments={},
            )
        self.assertEqual(output[1][0]["raw_columns"]["age_insightface"], "31")
        self.assertEqual(output[1][0]["raw_columns"]["gender_insightface"], "Male")
        np.testing.assert_array_equal(net.blob, expected)

    def test_roll_correction_levels_both_positive_and_negative_slopes(self):
        for rise in (20, -20):
            with self.subTest(rise=rise):
                frame = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.circle(frame, (65, 85), 3, (255, 0, 0), -1)
                cv2.circle(frame, (135, 85 + rise), 3, (0, 255, 0), -1)
                angle = np.degrees(np.arctan2(rise, 70))
                aligned, _ = inference._rotate_region(frame, (40, 40, 160, 160), angle)
                y_left = np.where(aligned[:, :, 0] > 128)[0].mean()
                y_right = np.where(aligned[:, :, 1] > 128)[0].mean()
                self.assertLess(abs(y_left - y_right), 1)
