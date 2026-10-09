"""Kaggle CPU kernel: record the app's face crops for FairFace TRAIN (see ../prep_faces.py).

Clones the repository at a pinned commit and runs its own detection, roll leveling and
landmark code in a Python 3.11 environment with the app's pins, so the crops match the app.
"""
import subprocess
import sys

REPO = "https://github.com/dpologdvinity/multimodal-face-analyzer.git"
COMMIT = "36dff9185c400410d343ddcd4395045e4e747fdb"
SRC = "/tmp/age/src"
ENV = "/tmp/age/env"
PY = f"{ENV}/bin/python"
PINS = ["numpy==2.4.6", "opencv-python-headless==4.14.0.94", "mediapipe==1.1.0", "pyarrow==25.0.1",
        "huggingface_hub==2.1.1"]


def run(*cmd: str) -> None:
    """Run a command, streaming its output into the kernel log."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


run(sys.executable, "-m", "pip", "install", "-q", "uv")
run("git", "clone", "-q", REPO, SRC)
run("git", "-C", SRC, "checkout", "-q", COMMIT)
run("uv", "venv", "-q", "--python", "3.11", ENV)
run("uv", "pip", "install", "-q", "--python", PY, *PINS)
# mediapipe pulls in another OpenCV build; keep only the pinned headless one (as docs/setup.md does).
run("uv", "pip", "uninstall", "-q", "--python", PY, "opencv-python", "opencv-contrib-python",
    "opencv-python-headless")
run("uv", "pip", "install", "-q", "--python", PY, "opencv-python-headless==4.14.0.94")
run("uv", "pip", "install", "-q", "--python", PY, "--no-deps", "-e", SRC)
run(PY, f"{SRC}/tools/fetch_models.py", "--keys", "FACE_LANDMARKS_MODEL=mediapipe",
    "--model-dir", "/tmp/age/models")
subprocess.run([PY, f"{SRC}/tools/kaggle/age_model/prep_faces.py", "--out", "/kaggle/working",
                "--workers", "4"], check=True, env={
    "PATH": f"{ENV}/bin:/usr/bin:/bin", "FACE_ANALYZER_MODEL_DIR": "/tmp/age/models",
    "OMP_NUM_THREADS": "1", "HOME": "/tmp/age"})
