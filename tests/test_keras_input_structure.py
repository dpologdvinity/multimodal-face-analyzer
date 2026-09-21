import unittest
from pathlib import Path


NETS_DIR = Path(__file__).parents[1] / "src" / "nets"


class KerasInputStructureTests(unittest.TestCase):
    def test_single_input_models_expose_a_tensor_input(self):
        """Keep single-input reconstructed Keras models compatible with tensor calls."""
        for filename in (
            "deepface_gender.py",
            "deepface_race.py",
            "deepface_recognition.py",
            "mask_model.py",
        ):
            source = (NETS_DIR / filename).read_text()
            self.assertIn("inputs=base.inputs[0]", source, filename)

    def test_mediapipe_detection_runs_inside_native_log_suppression(self):
        """Keep MediaPipe's non-actionable native projection diagnostics out of Streamlit logs."""
        source = (NETS_DIR.parent / "inference.py").read_text()
        self.assertIn("with _silence_native_logs():\n            return landmarker.detect(mp_image)", source)
        self.assertIn("with _silence_native_logs():\n            result = landmarker.detect(mp_image)", source)


if __name__ == "__main__":
    unittest.main()
