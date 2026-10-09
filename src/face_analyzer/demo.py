"""Public demo mode: the stateless, permissively licensed subset a shared deployment exposes."""
from __future__ import annotations

import os
import sys
import threading
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager

DEMO_ENV_VAR = "FACE_ANALYZER_DEMO"
_TRUTHY = {"1", "true", "yes", "on"}

# Backends a public demo may load, per native-installer/Docker ARG name. Every weight file
# behind them is a "hf" (permissively licensed, mirrored) manifest entry, and none needs torch
# or TensorFlow, so the demo fits a 2.7 GB free host. mediapipe only loads where the package is
# installed (it is not in requirements.txt; see docs/deploy.md).
DEMO_MODELS: dict[str, frozenset[str]] = {
    "RETINAFACE_MODEL": frozenset({"retinaface"}),
    "AGE_MODEL": frozenset({"fairface"}),
    "GENDER_MODEL": frozenset({"fairface"}),
    "RACE_MODEL": frozenset({"fairface"}),
    "EMOTION_MODEL": frozenset({"ferplus"}),
    "FACE_LANDMARKS_MODEL": frozenset({"mediapipe"}),
    "COLORIZATION_MODEL": frozenset({"eccv16"}),
}
DEMO_FACE_DETECTOR = "retinaface"

DEMO_MAX_UPLOAD_MB = 10
DEMO_MAX_FILES = 3
DEMO_MAX_IMAGE_SIDE = 1600
# Rejects decompression bombs (a tiny PNG that decodes to gigabytes) before cv2 decodes them.
DEMO_MAX_IMAGE_PIXELS = 50_000_000
# One analysis at a time per process: a colorized run alone peaks near 1.6 GB of a 2.7 GB host.
DEMO_ANALYSIS_WAIT_SECONDS = 120.0
# The slot lives on a module outside the package, which streamlit_app.py's source reload never
# purges: a re-executed demo.py must pick up the same semaphore, or two analyses could run at once.
ANALYSIS_SLOT_HOLDER = "_face_analyzer_reload_state"


def _shared_analysis_slot() -> threading.BoundedSemaphore:
    """Return the process-wide analysis semaphore, creating it on first use."""
    holder = sys.modules.setdefault(ANALYSIS_SLOT_HOLDER, types.ModuleType(ANALYSIS_SLOT_HOLDER))
    # dict.setdefault is atomic, so concurrent imports still agree on one semaphore.
    return vars(holder).setdefault("analysis_slot", threading.BoundedSemaphore(1))


_ANALYSIS_SLOT = _shared_analysis_slot()

RESPONSIBLE_USE_NOTE = (
    "Public demo. Age, gender, race and emotion are guesses about how a face *appears* in one "
    "image, not who a person is. The models carry known demographic biases, so error rates "
    "differ across groups. Do not use the results to make decisions about people. Uploaded "
    "images are processed in memory and never stored."
)
BUSY_NOTE = "Another visitor's image is being analyzed — retrying…"
LIVE_MODE_NOTE = "Live webcam mode runs locally only: it needs WebRTC, which most hosts cannot relay without a TURN server."


def is_demo_mode() -> bool:
    """Return whether FACE_ANALYZER_DEMO is set to a truthy value."""
    return os.environ.get(DEMO_ENV_VAR, "").strip().lower() in _TRUTHY


def demo_model_allowed(environment_name: str, model_name: str) -> bool:
    """Return whether demo mode permits a backend (always true outside demo mode)."""
    return not is_demo_mode() or model_name in DEMO_MODELS.get(environment_name, frozenset())


def demo_manifest_entries(entries: list[dict], include_mediapipe: bool) -> list[dict]:
    """Return the manifest entries a demo needs: "hf" files that are required or demo-selected."""
    selected = {
        (arg, key) for arg, keys in DEMO_MODELS.items() for key in keys
        if include_mediapipe or key != "mediapipe"
    }
    return [
        entry for entry in entries
        if entry["source"] == "hf"
        and (entry.get("required") or any(item in selected for item in entry["models"].items()))
    ]


class DemoBusyError(RuntimeError):
    """Raised when the demo's analysis slot stays taken for longer than the wait timeout."""


@contextmanager
def analysis_slot(on_wait: Callable[[], object]) -> Iterator[None]:
    """Hold the process-wide analysis slot, calling on_wait once if another analysis holds it."""
    if not _ANALYSIS_SLOT.acquire(blocking=False):
        on_wait()
        if not _ANALYSIS_SLOT.acquire(timeout=DEMO_ANALYSIS_WAIT_SECONDS):
            raise DemoBusyError("The demo is busy. Try again in a minute.")
    try:
        yield
    finally:
        _ANALYSIS_SLOT.release()


def refuse_in_demo_mode(action: str) -> None:
    """Raise PermissionError for an action that stores faces, when demo mode is on."""
    if is_demo_mode():
        raise PermissionError(f"{action} is disabled in demo mode: the public demo stores no faces.")


def demo_attribution(include_mediapipe: bool) -> str:
    """Return the one-line model credit the demo shows (FairFace's CC BY 4.0 requires one)."""
    credits = [
        "RetinaFace (biubug6/Pytorch_Retinaface, MIT; ONNX export by AMD, Apache-2.0)",
        "FairFace (Kärkkäinen and Joo, CC BY 4.0)",
        "FER+ emotion (ONNX Model Zoo, MIT)",
        "Colorful Image Colorization (Zhang et al., BSD-2-Clause)",
        "OpenCV Haar eye cascade (Intel License)",
    ]
    if include_mediapipe:
        credits.append("MediaPipe Face Landmarker (Google, Apache-2.0)")
    return "Models: " + "; ".join(credits) + ". Details in MODEL_LICENSES.md."
