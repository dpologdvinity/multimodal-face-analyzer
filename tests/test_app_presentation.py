import unittest
from pathlib import Path


APP_SOURCE = Path(__file__).parents[1] / "src" / "app.py"


class AppPresentationSourceTests(unittest.TestCase):
    def test_image_output_has_bounded_view_and_fullscreen_viewer(self):
        source = APP_SOURCE.read_text()

        self.assertIn('@st.dialog("IMAGE PREVIEW", width="large")', source)
        self.assertIn("width: int = IMAGE_DISPLAY_WIDTH", source)
        self.assertIn('st.button("🔍"', source)


if __name__ == "__main__":
    unittest.main()
