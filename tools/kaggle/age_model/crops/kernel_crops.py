"""Kaggle CPU kernel: rebuild the app's aligned face crops for FairFace TRAIN (../build_crops.py).

Clones the repository at a pinned commit, pins OpenCV to the app's version, and reads the crop
records of the face-age-prep kernel. The GPU training kernel memory-maps the result.
"""
import glob
import os
import subprocess
import sys

REPO = "https://github.com/dpologdvinity/multimodal-face-analyzer.git"
COMMIT = "a7498aac6f2066f10c91e169ccd92669b3b1d529"
SRC = "/tmp/age/src"


def run(*cmd: str) -> None:
    """Run a command, streaming its output into the kernel log."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


pip = [sys.executable, "-m", "pip"]
run(*pip, "uninstall", "-q", "-y", "opencv-python", "opencv-contrib-python", "opencv-python-headless")
run(*pip, "install", "-q", "opencv-python-headless==4.14.0.94")
run("git", "clone", "-q", REPO, SRC)
run("git", "-C", SRC, "checkout", "-q", COMMIT)
run(*pip, "install", "-q", "--no-deps", "-e", SRC)
run("df", "-h", "/tmp", "/kaggle/working")
prep = sorted(glob.glob("/kaggle/input/**/faces_train.npz", recursive=True))
if not prep:
    raise SystemExit("faces_train.npz not found under /kaggle/input")
os.environ["HF_HOME"] = "/tmp/age/hf"
run(sys.executable, f"{SRC}/tools/kaggle/age_model/build_crops.py", "--prep", os.path.dirname(prep[0]),
    "--out", "/kaggle/working")
