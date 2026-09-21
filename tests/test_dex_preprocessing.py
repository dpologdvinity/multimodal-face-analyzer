import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src import inference


class RecordingNet:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def setInput(self, blob):
        self.blob = blob.copy()

    def forward(self):
        return self.probabilities[None]


class DexPreprocessingTests(unittest.TestCase):
    def test_crop_uses_separate_width_and_height_margins(self):
        frame = np.arange(60 * 80 * 3).reshape(60, 80, 3).astype(np.uint8)
        # 20x10 face: eight pixels left/right, four above/below.
        np.testing.assert_array_equal(
            inference.crop_face_dex(frame, (20, 20, 40, 30)), frame[16:34, 12:48],
        )

    def test_crop_replicates_missing_border_instead_of_changing_scale(self):
        frame = np.arange(10 * 10 * 3).reshape(10, 10, 3).astype(np.uint8)
        expected = np.pad(frame, ((4, 4), (4, 4), (0, 0)), mode="edge")
        np.testing.assert_array_equal(inference.crop_face_dex(frame, (0, 0, 10, 10)), expected)

    def test_expected_value_normalizes_distribution_not_argmax(self):
        probabilities = np.zeros(101)
        probabilities[20], probabilities[40] = 1, 3
        self.assertEqual(inference.predict_age_dex(RecordingNet(probabilities), np.zeros((10, 10, 3), np.uint8)), "35")

    def test_broad_distribution_does_not_present_precise_age(self):
        self.assertEqual(
            inference.predict_age_dex(RecordingNet(np.ones(101) / 101), np.zeros((10, 10, 3), np.uint8)),
            "uncertain (mean 50, SD 29)",
        )

    def test_invalid_distribution_returns_unknown(self):
        for probabilities in (np.zeros(101), np.ones(100), np.full(101, np.nan),
                              np.full(101, np.inf), np.full(101, -1.0)):
            with self.subTest(probabilities=probabilities[:2]):
                self.assertEqual(inference.predict_age_dex(RecordingNet(probabilities), np.zeros((10, 10, 3), np.uint8)), "unknown")

    def test_pipeline_uses_original_dex_crop_and_cache_includes_its_context(self):
        frame = np.full((180, 180, 3), 80, np.uint8)
        # Detector box is 50x50, DEX needs 20px context on each side.
        box = (60, 60, 110, 110)
        probabilities = np.eye(101)[31]
        net = RecordingNet(probabilities)
        models = inference.Models(face_net=None, age_nets={"dex": net},
                                  eye_color_nets={"colorimetric": object()})
        options = dict(active_age={"dex"}, active_gender=set(), active_emotion=set(),
                       active_race=set(), active_recognition=set(), gallery={}, active_glasses=set(),
                       active_mask=set(), active_hair_color=set(), active_eye_color=set(),
                       active_face_landmarks=set(), active_hands=set(), active_gaze=set(),
                       global_adjustments={}, face_adjustments={"brightness": 10})
        with patch.object(inference, "detect_faces", return_value=[box]), \
             patch.object(inference, "_estimate_roll_angle", return_value=20):
            for context_pixel in (20, 200):
                frame[40:55, 40:130] = context_pixel
                output = inference.analyze_frame(models, frame, 0.5, **options)
                crop = inference.apply_image_adjustments(frame[40:130, 40:130], {"brightness": 10})
                expected = cv2.dnn.blobFromImage(crop, 1.0, (224, 224), inference.DEX_MEAN_VALUES, swapRB=False, crop=False)
                np.testing.assert_array_equal(net.blob, expected)
                self.assertEqual(output[1][0]["raw_columns"]["age_dex"], "31")


if __name__ == "__main__":
    unittest.main()
