"""Kaggle GPU kernel: fine-tune the age model on FairFace TRAIN (see train_age.py).

Clones the repository at a pinned commit so training rebuilds faces with the app's own code,
pins OpenCV to the app's version, and reads the crop records of the face-age-prep kernel.
"""
import glob
import os
import subprocess
import sys

REPO = "https://github.com/dpologdvinity/multimodal-face-analyzer.git"
COMMIT = "COMMIT_PLACEHOLDER"
SRC = "/tmp/age/src"
EXTRA_ARGS = []


def run(*cmd: str) -> None:
    """Run a command, streaming its output into the kernel log."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


pip = [sys.executable, "-m", "pip"]
run(*pip, "uninstall", "-q", "-y", "opencv-python", "opencv-contrib-python", "opencv-python-headless")
run(*pip, "install", "-q", "opencv-python-headless==4.14.0.94", "timm==1.0.30", "onnx==1.23.2")
run("git", "clone", "-q", REPO, SRC)
run("git", "-C", SRC, "checkout", "-q", COMMIT)
run(*pip, "install", "-q", "--no-deps", "-e", SRC)
prep = sorted(glob.glob("/kaggle/input/**/faces_train.npz", recursive=True))
if not prep:
    raise SystemExit("faces_train.npz not found under /kaggle/input")
os.environ["HF_HOME"] = "/tmp/age/hf"
run(sys.executable, f"{SRC}/tools/kaggle/age_model/train_age.py", "--prep", os.path.dirname(prep[0]),
    "--out", "/kaggle/working", *EXTRA_ARGS)
