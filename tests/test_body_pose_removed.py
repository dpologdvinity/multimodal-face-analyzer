import pathlib
import unittest


ROOT = pathlib.Path(__file__).parents[1]
APP_SOURCE = ROOT.joinpath("src", "app.py").read_text()
INFERENCE_SOURCE = ROOT.joinpath("src", "inference.py").read_text()


class BodyPoseRemovalTests(unittest.TestCase):
    def test_body_pose_is_not_exposed_or_loaded(self):
        self.assertNotIn('"BODY LANDMARKS"', APP_SOURCE)
        self.assertNotIn("active_pose", APP_SOURCE)
        self.assertNotIn("pose_nets", INFERENCE_SOURCE)
        self.assertNotIn("detect_pose_mpi", INFERENCE_SOURCE)

    def test_pose_build_option_is_removed(self):
        for filename in ("Dockerfile", "build-and-run.sh", "install-and-run.sh"):
            source = ROOT.joinpath(filename).read_text()
            self.assertNotIn("POSE_MODEL", source)
            self.assertNotIn("body pose - mpi", source)


if __name__ == "__main__":
    unittest.main()
