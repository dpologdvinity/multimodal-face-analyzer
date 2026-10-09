"""Kaggle GPU kernel: fine-tune the age model on FairFace TRAIN (see train_age.py).

Clones the repository at a pinned commit, pins OpenCV to the app's version, and reads the labels
of the face-age-prep kernel and the aligned crops of the face-age-crops kernel.
"""
import glob
import os
import subprocess
import sys

REPO = "https://github.com/dpologdvinity/multimodal-face-analyzer.git"
COMMIT = "ecb8e824b6a42751ad1ae88c75359b9c58d66460"
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
crops = sorted(glob.glob("/kaggle/input/**/crops_train.npy", recursive=True))
if not prep or not crops:
    raise SystemExit("faces_train.npz or crops_train.npy not found under /kaggle/input")
os.environ["HF_HOME"] = "/tmp/age/hf"
run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
run("nproc")
run("free", "-g")
run(sys.executable, f"{SRC}/tools/kaggle/age_model/train_age.py", "--prep", os.path.dirname(prep[0]),
    "--crops", os.path.dirname(crops[0]), "--out", "/kaggle/working", *EXTRA_ARGS)
