import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "build-and-run.sh"


class BuildPromptFormattingTests(unittest.TestCase):
    def _run_build(self, selections: list[str]) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_bin = Path(temp_dir) / "bin"
            fake_bin.mkdir()
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
        self.assertIn("\033[1;32m  4) dex\033[0m", output)
        self.assertIn("\033[1;33m  5) ssrnet\033[0m", output)
        self.assertIn("\033[1;32m  3) hsemotion\033[0m", output)
        self.assertIn("\033[1;33m  4) mini_xception\033[0m", output)
        self.assertIn("\033[1;32m  2) body pose - mpi\033[0m", output)
        self.assertIn("\033[1;33m  3) 3d reconstruction - deep3d\033[0m", output)

        self.assertIn("  0) none", output)
        self.assertIn("  9) all", output)
        self.assertNotIn("\033[1;34m  0) none", output)
        self.assertNotIn("\033[1;34m  9) all", output)

    def test_reordered_choices_still_forward_the_selected_build_models(self):
        selections = ["2", "5", "4", "2", "5", "0", "0", "3"]
        output = self._run_build(selections)

        self.assertIn("SCRFD_FACE_MODEL=scrfd", output)
        self.assertIn("AGE_MODEL=ssrnet", output)
        self.assertIn("GENDER_MODEL=deepface", output)
        self.assertIn("EMOTION_MODEL=dan", output)
        self.assertIn("RECONSTRUCTION_3D_MODEL=deep3d", output)


if __name__ == "__main__":
    unittest.main()
