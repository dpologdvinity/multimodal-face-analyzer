"""Verify FERPlus gets its tight crop while other expression models keep padded crops."""
import unittest
from unittest.mock import patch

import numpy as np

from face_analyzer import inference


class ExpressionCropTests(unittest.TestCase):
    def test_ferplus_uses_detector_crop_and_preserves_other_models(self):
        frame = np.random.default_rng(8).integers(0, 256, (100, 120, 3), dtype=np.uint8)
        models = inference.Models(
            object(), emotion_nets={key: object() for key in ("ferplus", "hsemotion")},
            eye_color_nets={"colorimetric": object()},
        )
        box = (30, 20, 70, 70)
        with (patch("face_analyzer.pipeline.stages.detect_faces", return_value=[box]) as detect,
              patch("face_analyzer.pipeline.stages._estimate_roll_angle", return_value=12) as roll,
              patch("face_analyzer.pipeline.stages._rotate_region", return_value=(255 - frame, box)) as rotate,
              patch("face_analyzer.pipeline.face_tasks.predict_emotion_ferplus", return_value="happiness") as ferplus,
              patch("face_analyzer.pipeline.face_tasks.predict_emotion_hsemotion", return_value="happiness") as hse):
            _, faces, _, _ = inference.analyze_frame(
                models, frame,
                inference.AnalysisConfig(active_emotion={"ferplus", "hsemotion"}, face_detector="ssd"),
            )
        for mocked in (detect, roll, rotate, ferplus, hse):
            mocked.assert_called()
        np.testing.assert_array_equal(ferplus.call_args.args[1], frame[20:70, 30:70])
        rx1, ry1, rx2, ry2 = inference.face_crop_bounds(box, frame.shape[:2])
        np.testing.assert_array_equal(hse.call_args.args[1], (255 - frame)[ry1:ry2, rx1:rx2])
        self.assertEqual(faces[0]["raw_columns"]["emotion_ferplus"], "happiness")


if __name__ == "__main__":
    unittest.main()
