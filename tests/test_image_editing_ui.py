import pathlib
import unittest


APP_SOURCE = pathlib.Path(__file__).parents[1].joinpath("src", "app.py").read_text()


class ImageEditingUiTests(unittest.TestCase):
    """Verify image editor UI uses mouse-based cropping and layout."""

    def test_uses_mouse_cropper_instead_of_coordinate_transform_form(self):
        """Ensure photo editor uses visual cropper over coordinate-based form."""
        self.assertIn("from streamlit_cropper import st_cropper", APP_SOURCE)
        self.assertIn("def _render_photo_editor", APP_SOURCE)
        self.assertIn('st.button("Crop photo"', APP_SOURCE)
        self.assertNotIn('st.expander("SELECT REGION & TRANSFORM")', APP_SOURCE)
        self.assertNotIn('st.button("APPLY TRANSFORM"', APP_SOURCE)

    def test_editors_are_rendered_next_to_pre_and_post_detection_images(self):
        """Verify editor and detection images are laid out side-by-side."""
        self.assertIn('st.columns([3, 2])', APP_SOURCE)
        self.assertIn('"Source photo"', APP_SOURCE)
        self.assertIn('"Edit this crop only.', APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
