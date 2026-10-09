"""Public demo mode: the stateless, permissively licensed subset a shared deployment exposes."""
from __future__ import annotations

import os

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
    "HAIR_COLOR_MODEL": frozenset({"colorimetric"}),
}
DEMO_FACE_DETECTOR = "retinaface"

DEMO_MAX_UPLOAD_MB = 10
DEMO_MAX_IMAGE_SIDE = 1600
# Rejects decompression bombs (a tiny PNG that decodes to gigabytes) before cv2 decodes them.
DEMO_MAX_IMAGE_PIXELS = 50_000_000

RESPONSIBLE_USE_NOTE = (
    "Public demo. Age, gender, race and emotion are guesses about how a face *appears* in one "
    "image, not who a person is. The models carry known demographic biases, so error rates "
    "differ across groups. Do not use the results to make decisions about people. Uploaded "
    "images are processed in memory and never stored."
)
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
