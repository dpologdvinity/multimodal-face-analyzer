import pathlib
import unittest


APP_SOURCE = pathlib.Path(__file__).parents[1].joinpath("src", "app.py").read_text()


class FaceEditorToggleTests(unittest.TestCase):
    def test_face_editor_is_closed_by_default_and_toggled_per_face(self):
        """Verify editor button exists and state is tracked per face with default closed."""
        self.assertIn('face_editor_open_key = f"face_editor_open_', APP_SOURCE)
        self.assertIn('st.session_state.setdefault(face_editor_open_key, False)', APP_SOURCE)
        self.assertIn('"Edit face"', APP_SOURCE)
        self.assertIn('if st.session_state[face_editor_open_key]:', APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
