import os
import unittest
from unittest.mock import patch

from src.model_selection import native_model_selected


class NativeModelSelectionTests(unittest.TestCase):
    def test_unset_selection_preserves_model_discovery(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(native_model_selected("AGE_MODEL", "caffe"))

    def test_selected_backends_are_allowed(self):
        with patch.dict(os.environ, {"AGE_MODEL": "caffe,fairface"}, clear=True):
            self.assertTrue(native_model_selected("AGE_MODEL", "caffe"))
            self.assertTrue(native_model_selected("AGE_MODEL", "fairface"))
            self.assertFalse(native_model_selected("AGE_MODEL", "ssrnet"))

    def test_empty_selection_disables_all_backends(self):
        with patch.dict(os.environ, {"AGE_MODEL": ""}, clear=True):
            self.assertFalse(native_model_selected("AGE_MODEL", "caffe"))


if __name__ == "__main__":
    unittest.main()
