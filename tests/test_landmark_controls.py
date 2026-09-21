import pathlib
import unittest


APP_SOURCE = pathlib.Path(__file__).parents[1].joinpath("src", "app.py").read_text()


class LandmarkControlTests(unittest.TestCase):
    def test_remaining_landmark_groups_have_explicit_enable_buttons(self):
        self.assertIn("def _landmark_enable_button", APP_SOURCE)
        self.assertIn('"FACE LANDMARKS"', APP_SOURCE)
        self.assertIn('"HAND LANDMARKS"', APP_SOURCE)
        self.assertIn('st.sidebar.button', APP_SOURCE)
        self.assertNotIn('"BODY LANDMARKS"', APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
