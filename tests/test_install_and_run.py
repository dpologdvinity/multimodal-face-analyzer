import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "install-and-run.sh"


class NativeInstallerVerbosityTests(unittest.TestCase):
    def _run_installer(
        self, *args: str, input_data: str | None = None, capture_env: bool = False
    ) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            fake_bin = temp / "bin"
            fake_bin.mkdir()
            venv_python = temp / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)

            for name in ("python3", "python3.11", "apt-get"):
                (fake_bin / name).write_text("#!/bin/sh\nexit 0\n")
                (fake_bin / name).chmod(0o755)
            (fake_bin / "sudo").write_text("#!/bin/sh\nexec \"$@\"\n")
            (fake_bin / "sudo").chmod(0o755)
            venv_python.write_text(
                '#!/bin/sh\n'
                'if [ -n "$CAPTURE_ENV" ]; then env > "$CAPTURE_ENV"; fi\n'
                'exit 0\n'
            )
            venv_python.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["VENV_DIR"] = str(temp / "venv")
            if capture_env:
                env["CAPTURE_ENV"] = str(temp / "captured-env")
            result = subprocess.run(
                ["bash", str(SCRIPT), *args],
                cwd=SCRIPT.parent,
                input=input_data if input_data is not None else "0\n" * 25,
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout + result.stderr
            if capture_env:
                output += (temp / "captured-env").read_text()
            return output

    def test_default_prompts_hide_dependency_details(self):
        output = self._run_installer()

        self.assertIn("\033[1;32m* 1) caffe\033[0m", output)
        self.assertIn("\033[1;33m  5) ssrnet\033[0m", output)
        self.assertIn("  0) none", output)
        self.assertIn("  9) all", output)
        self.assertNotIn("\033[1;32m0) none\033[0m", output)
        self.assertNotIn("\033[33m9) all\033[0m", output)
        self.assertNotIn("base packages", output)
        self.assertNotIn("      packages:", output)
        self.assertNotIn("[default: 1]", output)

        self.assertIn("\033[1;34m== FACE DETECTION ==\033[0m", output)
        self.assertIn("\033[1;32m* 0) ssd\033[0m", output)
        self.assertIn("\033[1;33m  1) yolo\033[0m", output)
        self.assertIn("\033[1;33m  2) scrfd\033[0m", output)
        self.assertIn("\033[1;33m  3) retinaface\033[0m", output)
        self.assertNotIn("== YOLO FACE DETECTOR", output)
        self.assertNotIn("== SCRFD FACE DETECTOR", output)
        self.assertNotIn("== RETINAFACE FACE DETECTOR", output)
        for section in (
            "AGE",
            "GENDER",
            "RACE",
            "EMOTION",
            "RECOGNITION",
            "ADDITIONAL CLASSIFICATIONS",
            "ADDITIONAL FEATURES",
        ):
            self.assertIn(f"\033[1;34m== {section} ==\033[0m", output)
        self.assertIn("  0) none", output)
        self.assertIn("== ADDITIONAL CLASSIFICATIONS ==", output)
        self.assertNotIn("expressions - blendshapes", output)
        self.assertIn("\033[1;33m  3) liveness - mediapipe\033[0m", output)
        self.assertIn("\033[1;32m  6) hair color - colorimetric\033[0m", output)
        self.assertIn("\033[1;33m  2) face landmarks - mediapipe\033[0m", output)

    def test_requirement_only_models_precede_additional_install_models(self):
        output = self._run_installer()

        for green, orange in (
            ("\033[1;32m  4) dex\033[0m", "\033[1;33m  5) ssrnet\033[0m"),
            ("\033[1;32m  3) fairface\033[0m", "\033[1;33m  4) deepface\033[0m"),
            ("\033[1;32m  3) hsemotion\033[0m", "\033[1;33m  4) mini_xception\033[0m"),
            ("\033[1;32m  2) facial hair - bisenet\033[0m", "\033[1;33m  3) liveness - mediapipe\033[0m"),
            ("\033[1;32m  3) body pose - mpi\033[0m", "\033[1;33m  4) 3d reconstruction - deep3d\033[0m"),
        ):
            self.assertLess(output.index(green), output.index(orange))

    def test_verbose_prompts_show_colored_dependency_details(self):
        for flag in ("-v", "--verbose"):
            with self.subTest(flag=flag):
                output = self._run_installer(flag)

                self.assertNotIn("\n      packages:", output)
                self.assertIn("\033[1;32m* 1) caffe\033[0m", output)
                self.assertIn("\033[1;33m  5) ssrnet\033[0m", output)
                self.assertIn("\033[1;32m* 0) ssd\033[0m", output)
                self.assertIn("\033[32mopencv-python-headless, numpy\033[0m", output)
                self.assertIn("\033[33mtorch, torchvision\033[0m", output)
                self.assertIn("      ├──", output)
                self.assertIn("      └──", output)

    def test_face_detector_selection_maps_to_its_original_build_variable(self):
        selections = "\n".join(["2"] + ["0"] * 7) + "\n"
        output = self._run_installer("--verbose", input_data=selections)

        self.assertIn("YOLO_FACE_MODEL=", output)
        self.assertIn("SCRFD_FACE_MODEL=scrfd", output)
        self.assertIn("RETINAFACE_MODEL=", output)

    def test_face_detector_selection_accepts_all_multi_option_separators(self):
        for selection in ("123", "1 2 3", "1,2,3"):
            with self.subTest(selection=selection):
                selections = "\n".join([selection] + ["0"] * 7) + "\n"
                output = self._run_installer(input_data=selections)

                self.assertIn("YOLO_FACE_MODEL=yolo", output)
                self.assertIn("SCRFD_FACE_MODEL=scrfd", output)
                self.assertIn("RETINAFACE_MODEL=retinaface", output)

    def test_reused_optional_packages_turn_later_options_green(self):
        selections = "\n".join(["0", "5", "0", "0", "5"] + ["0"] * 3) + "\n"
        output = self._run_installer("--verbose", input_data=selections)

        self.assertIn("\033[1;32m  5) dan\033[0m", output)
        self.assertIn("\033[32mopencv-python-headless, numpy, torch, torchvision\033[0m", output)

    def test_hidden_mode_omits_choices_with_new_optional_packages(self):
        output = self._run_installer("--hidden")

        self.assertIn("== AGE ==", output)
        self.assertIn("caffe", output)
        self.assertNotIn("ssrnet", output)
        self.assertNotIn("deepface", output)
        self.assertNotIn("yolo", output)
        self.assertNotIn("vggface", output)

    def test_hidden_mode_all_selects_only_base_package_models(self):
        output = self._run_installer("--hidden", input_data="9\n" * 8)

        self.assertIn("AGE_MODEL=caffe,insightface,fairface,dex", output)
        self.assertIn("GENDER_MODEL=caffe,insightface,fairface", output)
        self.assertIn("EMOTION_MODEL=efficientnet,ferplus,hsemotion", output)
        self.assertIn("RECOGNITION_MODEL=", output)
        self.assertNotIn("Installing: torch", output)
        self.assertNotIn("Installing: tensorflow-cpu", output)
        self.assertNotIn("Installing: onnxruntime", output)

    def test_package_marks_a_dependency_as_already_available(self):
        output = self._run_installer("-p", "onxruntime", input_data="0\n" * 8)

        self.assertIn("\033[1;32m  1) yolo\033[0m", output)

    def test_package_works_with_hidden_mode_and_comma_separated_packages(self):
        output = self._run_installer(
            "--hidden", "--package", "onxruntime,numpy", input_data="0\n" * 8
        )

        self.assertIn("\033[1;32m  1) yolo\033[0m", output)
        self.assertIn("scrfd", output)
        self.assertNotIn("deepface", output)

    def test_native_runtime_receives_selected_models(self):
        selections = "\n".join(["0", "1", "0", "0", "0", "0", "0", "1"]) + "\n"
        output = self._run_installer(input_data=selections, capture_env=True)

        self.assertIn("AGE_MODEL=caffe", output)
        self.assertIn("GENDER_MODEL=", output)
        self.assertIn("YOLO_FACE_MODEL=", output)
        self.assertIn("COLORIZATION_MODEL=eccv16", output)
        self.assertIn("AGE_PROGRESSION_MODEL=", output)

    def test_hair_color_selection_reaches_native_runtime(self):
        selections = "\n".join(["0", "0", "0", "0", "0", "0", "6", "0"]) + "\n"
        output = self._run_installer(input_data=selections, capture_env=True)

        self.assertIn("HAIR_COLOR_MODEL=colorimetric", output)


if __name__ == "__main__":
    unittest.main()
