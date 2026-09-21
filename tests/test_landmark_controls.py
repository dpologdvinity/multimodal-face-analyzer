import ast
import pathlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


APP_SOURCE = pathlib.Path(__file__).parents[1].joinpath("src", "app.py").read_text()


class LandmarkControlTests(unittest.TestCase):
    """Verify UI exposes face and hand landmark enable buttons, not body landmarks."""

    def test_remaining_landmark_groups_have_explicit_enable_buttons(self):
        """Ensure FACE and HAND landmark toggles exist via sidebar buttons."""
        self.assertIn("def _landmark_enable_button", APP_SOURCE)
        self.assertIn('"FACE LANDMARKS"', APP_SOURCE)
        self.assertIn('"HAND LANDMARKS"', APP_SOURCE)
        self.assertNotIn('"BODY LANDMARKS"', APP_SOURCE)

    def test_enable_button_works_in_sidebar_and_expander(self):
        # Execute the real UI helper without importing the app or loading its models.
        function = next(node for node in ast.parse(APP_SOURCE).body
                        if isinstance(node, ast.FunctionDef) and node.name == "_landmark_enable_button")
        for use_expander in (False, True):
            with self.subTest(use_expander=use_expander):
                sidebar, expander = Mock(), Mock()
                container = expander if use_expander else sidebar
                container.button.return_value = False
                streamlit = SimpleNamespace(sidebar=sidebar, session_state={}, rerun=Mock(side_effect=RuntimeError("rerun")))
                namespace = {"st": streamlit}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(pathlib.Path(__file__)), "exec"), namespace)
                render = namespace["_landmark_enable_button"]
                options = {"container": expander} if use_expander else {}
                self.assertEqual(render("FACE LANDMARKS", {"mediapipe": object()}, "landmarks", **options), {"mediapipe"})
                container.button.return_value = True
                with self.assertRaisesRegex(RuntimeError, "rerun"):
                    render("FACE LANDMARKS", {"mediapipe": object()}, "landmarks", **options)
                self.assertFalse(streamlit.session_state["landmarks"])
                container.button.return_value = False
                self.assertEqual(render("FACE LANDMARKS", {"mediapipe": object()}, "landmarks", **options), set())
                self.assertEqual(container.button.call_args.args[0], "ENABLE FACE LANDMARKS")
                if use_expander:
                    sidebar.button.assert_not_called()


if __name__ == "__main__":
    unittest.main()
