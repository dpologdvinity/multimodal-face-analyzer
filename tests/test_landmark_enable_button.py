import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.ui import sidebar as sidebar_ui


class LandmarkControlTests(unittest.TestCase):
    """Verify the landmark enable-button helper behaves correctly."""

    def test_enable_button_works_in_sidebar_and_expander(self):
        # Exercise the real UI helper with Streamlit stubbed out, without running the app or loading models.
        for use_expander in (False, True):
            with self.subTest(use_expander=use_expander):
                sidebar, expander = Mock(), Mock()
                container = expander if use_expander else sidebar
                container.button.return_value = False
                streamlit = SimpleNamespace(sidebar=sidebar, session_state={}, rerun=Mock(side_effect=RuntimeError("rerun")))
                with patch.object(sidebar_ui, "st", streamlit):
                    render = sidebar_ui._landmark_enable_button
                    options = {"container": expander} if use_expander else {}
                    self.assertEqual(render("FACE LANDMARKS", {"mediapipe": object()}, "landmarks", **options), {"mediapipe"})
                    container.button.return_value = True
                    with self.assertRaisesRegex(RuntimeError, "rerun"):
                        render("FACE LANDMARKS", {"mediapipe": object()}, "landmarks", **options)
                    self.assertFalse(streamlit.session_state["landmarks"])
                    container.button.return_value = False
                    self.assertEqual(render("FACE LANDMARKS", {"mediapipe": object()}, "landmarks", **options), set())
                    self.assertEqual(container.button.call_args.args[0], "Enable Face Landmarks")
                    if use_expander:
                        sidebar.button.assert_not_called()


if __name__ == "__main__":
    unittest.main()
