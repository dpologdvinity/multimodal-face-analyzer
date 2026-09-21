import pathlib
import unittest


ROOT = pathlib.Path(__file__).parents[1]
APP_SOURCE = ROOT.joinpath("src", "app.py").read_text()
INFERENCE_SOURCE = ROOT.joinpath("src", "inference.py").read_text()
BUILD_SOURCE = ROOT.joinpath("build-and-run.sh").read_text()
INSTALL_SOURCE = ROOT.joinpath("install-and-run.sh").read_text()
DOCKER_SOURCE = ROOT.joinpath("Dockerfile").read_text()


class BlendshapesRemovalTests(unittest.TestCase):
    def test_expression_backend_is_removed_from_runtime(self):
        self.assertNotIn("expression_nets", INFERENCE_SOURCE)
        self.assertNotIn("predict_expression_blendshapes", INFERENCE_SOURCE)
        self.assertNotIn("active_expression", APP_SOURCE)

    def test_blendshapes_is_not_a_build_selection(self):
        for source in (BUILD_SOURCE, INSTALL_SOURCE, DOCKER_SOURCE):
            self.assertNotIn("EXPRESSION_MODEL", source)
            self.assertNotIn("expressions - blendshapes", source)

    def test_face_landmarks_keep_a_separate_mediapipe_selection(self):
        self.assertIn("FACE_LANDMARKS_MODEL", BUILD_SOURCE)
        self.assertIn("FACE_LANDMARKS_MODEL", INSTALL_SOURCE)
        self.assertIn("FACE_LANDMARKS_MODEL", DOCKER_SOURCE)
        self.assertIn('"mediapipe"', INFERENCE_SOURCE)


if __name__ == "__main__":
    unittest.main()
