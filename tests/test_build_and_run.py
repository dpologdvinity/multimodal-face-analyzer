import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "build-and-run.sh"


class BuildPromptFormattingTests(unittest.TestCase):
    def _run_build(self, selections: list[str]) -> str:
        """Run build-and-run.sh with fake docker to test prompts without actually building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_bin = Path(temp_dir) / "bin"
            fake_bin.mkdir()
            # Mock docker to avoid expensive image builds and still exercise prompt paths
            docker = fake_bin / "docker"
            docker.write_text("#!/bin/sh\nexit 0\n")
            docker.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=SCRIPT.parent,
                input="\n".join(selections + ["n", "q"]) + "\n",
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout + result.stderr

    def test_prompts_match_native_visual_format_and_model_order(self):
        """Verify prompt sections are formatted with colors and models in correct order."""
        output = self._run_build(["0"] * 8)

        for section in (
            "FACE DETECTION",
            "AGE",
            "GENDER",
            "RACE",
            "EMOTION",
            "RECOGNITION",
            "ADDITIONAL CLASSIFICATIONS",
            "ADDITIONAL FEATURES",
        ):
            self.assertIn(f"\033[1;34m== {section} ==\033[0m", output)

        self.assertIn("\033[1;32m* 0) ssd\033[0m", output)
        self.assertIn("\033[1;33m  1) yolo\033[0m", output)
        self.assertIn("\033[1;32m* 1) caffe\033[0m", output)
        self.assertIn("\033[1;32m  3) dex\033[0m", output)
        self.assertIn("\033[1;33m  4) ssrnet\033[0m", output)
        self.assertIn("\033[1;32m  3) hsemotion\033[0m", output)
        self.assertIn("\033[1;33m  4) mini_xception\033[0m", output)
        self.assertIn("\033[1;33m  3) 3d reconstruction - deep3d\033[0m", output)
        self.assertIn("\033[1;32m  4) hair color - colorimetric\033[0m", output)

        self.assertIn("  0) none", output)
        self.assertIn("  9) all", output)
        self.assertNotIn("\033[1;34m  0) none", output)
        self.assertNotIn("\033[1;34m  9) all", output)

    def test_reordered_choices_still_forward_the_selected_build_models(self):
        """Verify selected model indices map correctly to docker build args."""
        selections = ["2", "4", "3", "2", "5", "0", "0", "3"]
        output = self._run_build(selections)

        self.assertIn("SCRFD_FACE_MODEL=scrfd", output)
        self.assertIn("AGE_MODEL=ssrnet", output)
        self.assertIn("GENDER_MODEL=deepface", output)
        self.assertIn("EMOTION_MODEL=dan", output)
        self.assertIn("RECONSTRUCTION_3D_MODEL=deep3d", output)

    def test_face_detector_selection_accepts_all_multi_option_separators(self):
        """Support multiple input separators (spaces, commas, or no separator) for multi-select."""
        for selection in ("123", "1 2 3", "1,2,3"):
            with self.subTest(selection=selection):
                output = self._run_build([selection] + ["0"] * 7)

                self.assertIn("YOLO_FACE_MODEL=yolo", output)
                self.assertIn("SCRFD_FACE_MODEL=scrfd", output)
                self.assertIn("RETINAFACE_MODEL=retinaface", output)

    def test_build_context_contains_only_required_models(self):
        """Verify default selection (all 0) stages only the three core model files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            manifest = temp_path / "models.txt"
            docker = fake_bin / "docker"
            # Mock docker to capture which models were passed to the build context
            docker.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = build ]; then\n"
                "  for arg do context=\"$arg\"; done\n"
                "  find \"$context/models\" -type f -printf '%P\\n' | sort > \"$DOCKER_MODEL_MANIFEST\"\n"
                "fi\n"
                "exit 0\n"
            )
            docker.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["DOCKER_MODEL_MANIFEST"] = str(manifest)
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=SCRIPT.parent,
                input="\n".join(["0"] * 8 + ["n", "q"]) + "\n",
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                manifest.read_text().splitlines(),
                ["haarcascade_eye.xml", "opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb"],
            )

    def test_shared_model_is_staged_once_for_multiple_features(self):
        """Verify fairface_7class.onnx appears once when age and gender both select it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            manifest = temp_path / "models.txt"
            docker = fake_bin / "docker"
            # Mock docker to capture which models were passed to the build context
            docker.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = build ]; then\n"
                "  for arg do context=\"$arg\"; done\n"
                "  find \"$context/models\" -type f -printf '%P\\n' | sort > \"$DOCKER_MODEL_MANIFEST\"\n"
                "fi\n"
                "exit 0\n"
            )
            docker.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["DOCKER_MODEL_MANIFEST"] = str(manifest)
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=SCRIPT.parent,
                input="\n".join(["0", "2", "2", "0", "0", "0", "0", "0", "n", "q"]) + "\n",
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                manifest.read_text().splitlines(),
                ["fairface_7class.onnx", "haarcascade_eye.xml", "opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb"],
            )


if __name__ == "__main__":
    unittest.main()
