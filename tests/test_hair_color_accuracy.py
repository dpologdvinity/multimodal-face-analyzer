"""Verify calibrated colorimetric hair color prediction accuracy on benchmark images."""
from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from tests._models import require_local_data


class TestHairColorAccuracy(unittest.TestCase):
    """Test calibrated hair color predictions on confirmed ground truth images."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.assets_dir = cls.root / "assets"
        cls.boxes = {
            "joe-biden.jpg": (311, 108, 552, 457),
            "5yr-asian-girl-happy.jpg": (1034, 352, 1426, 826),
            "3yr-white-girl-sad.jpg": (526, 79, 654, 257),
        }

    def _read_asset(self, name: str):
        """Load a benchmark photo, skipping the test when the untracked assets are absent."""
        path = self.assets_dir / name
        require_local_data(path)
        return cv2.imread(str(path))

    def test_hair_color_biden_white(self):
        """Verify white hair detection on Joe Biden."""
        from face_analyzer import inference

        img = self._read_asset("joe-biden.jpg")
        self.assertIsNotNone(img)
        pred = inference.predict_hair_color_colorimetric(img, self.boxes["joe-biden.jpg"])
        self.assertEqual(pred, "white")

    def test_hair_color_asian_child_black(self):
        """Verify black hair detection on Asian child."""
        from face_analyzer import inference

        img = self._read_asset("5yr-asian-girl-happy.jpg")
        self.assertIsNotNone(img)
        pred = inference.predict_hair_color_colorimetric(img, self.boxes["5yr-asian-girl-happy.jpg"])
        self.assertEqual(pred, "black")

    def test_hair_color_white_child_blonde(self):
        """Verify blonde hair detection on White child."""
        from face_analyzer import inference

        img = self._read_asset("3yr-white-girl-sad.jpg")
        self.assertIsNotNone(img)
        pred = inference.predict_hair_color_colorimetric(img, self.boxes["3yr-white-girl-sad.jpg"])
        self.assertEqual(pred, "blonde")

    def test_attribute_reexport_parity(self):
        """Verify attributes subpackage re-exports all attribute functions."""
        import face_analyzer.attributes as attributes
        from face_analyzer import inference

        for fn_name in [
            "predict_hair_color_colorimetric",
            "predict_eye_color_colorimetric",
            "predict_glasses_mobilenet",
            "predict_mask_mobilenetv2",
            "predict_age_caffe",
            "predict_gender_caffe",
            "predict_emotion_ferplus",
        ]:
            self.assertTrue(hasattr(attributes, fn_name), f"Missing {fn_name} in attributes")
            self.assertIs(getattr(attributes, fn_name), getattr(inference, fn_name))


if __name__ == "__main__":
    unittest.main()
