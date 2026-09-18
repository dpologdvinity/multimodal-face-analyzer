import unittest
from pathlib import Path


class FaceDetectorDefaultTests(unittest.TestCase):
    def test_yolo_is_the_runtime_default_with_ssd_fallback(self):
        root = Path(__file__).resolve().parents[1]
        app_source = (root / "src" / "app.py").read_text()
        inference_source = (root / "src" / "inference.py").read_text()

        self.assertIn('active_face_detector = "yolo" if models.yolo_face_nets else "ssd"', app_source)
        self.assertIn('face_detector: str = "yolo"', inference_source)
        self.assertIn('"yolo" if models.yolo_face_nets', app_source)
        self.assertIn('face_detector picks which face detection backend', inference_source)


if __name__ == "__main__":
    unittest.main()
