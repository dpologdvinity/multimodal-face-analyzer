import os
import unittest
from unittest.mock import patch

from face_analyzer.model_selection import native_model_selected


class NativeModelSelectionTests(unittest.TestCase):
    """Verify model selection respects environment variable filtering."""

    def test_unset_selection_preserves_model_discovery(self):
        """Ensure unset env var allows all models to be selected (discovery mode)."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(native_model_selected("AGE_MODEL", "caffe"))

    def test_selected_backends_are_allowed(self):
        """Verify comma-separated env var allows only listed backends."""
        with patch.dict(os.environ, {"AGE_MODEL": "caffe,fairface"}, clear=True):
            self.assertTrue(native_model_selected("AGE_MODEL", "caffe"))
            self.assertTrue(native_model_selected("AGE_MODEL", "fairface"))
            self.assertFalse(native_model_selected("AGE_MODEL", "dex"))

    def test_empty_selection_disables_all_backends(self):
        """Ensure empty env var disables all backends for that feature."""
        with patch.dict(os.environ, {"AGE_MODEL": ""}, clear=True):
            self.assertFalse(native_model_selected("AGE_MODEL", "caffe"))

    def test_hair_color_selection_can_disable_colorimetric_backend(self):
        """Confirm empty HAIR_COLOR_MODEL env var blocks colorimetric backend."""
        with patch.dict(os.environ, {"HAIR_COLOR_MODEL": ""}, clear=True):
            self.assertFalse(native_model_selected("HAIR_COLOR_MODEL", "colorimetric"))

    def test_demo_mode_limits_backends_to_the_demo_allowlist(self):
        """Allow only demo backends in demo mode, even when an explicit selection lists more."""
        with patch.dict(os.environ, {"FACE_ANALYZER_DEMO": "1", "AGE_MODEL": "caffe,fairface"}, clear=True):
            self.assertTrue(native_model_selected("AGE_MODEL", "fairface"))
            self.assertFalse(native_model_selected("AGE_MODEL", "caffe"))
            self.assertFalse(native_model_selected("RECOGNITION_MODEL", "lbph"))
            self.assertFalse(native_model_selected("YOLO_FACE_MODEL", "yolo"))
            self.assertTrue(native_model_selected("RETINAFACE_MODEL", "retinaface"))


if __name__ == "__main__":
    unittest.main()
