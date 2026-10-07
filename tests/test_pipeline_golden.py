"""Golden characterization test: analyze_frame output on a fixed public-domain photo."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core import constants as C
from src.inference import analyze_frame, load_models
from tests._models import require_model

FIXTURE = Path(__file__).parent / "fixtures" / "crew_portrait.jpg"
GOLDEN = Path(__file__).parent / "fixtures" / "golden_analysis.json"

# Every model file the cv2.dnn/OpenCV-only backends load from (no torch/TF/onnxruntime needed).
REQUIRED_MODELS = [
    C.FACE_PROTO, C.FACE_MODEL,
    C.AGE_PROTO, C.AGE_MODEL,
    C.GENDER_PROTO, C.GENDER_MODEL,
    C.DEX_PROTO, C.DEX_MODEL,
    C.FAIRFACE_MODEL, C.FERPLUS_MODEL, C.HSEMOTION_MODEL,
]


def _config(models) -> dict:
    """Build the analyze_frame kwargs; the one place to change if its signature changes."""
    keys = lambda nets: set(nets)  # noqa: E731
    return dict(
        conf_threshold=0.5,
        active_age=keys(models.age_nets),
        active_gender=keys(models.gender_nets),
        active_emotion=keys(models.emotion_nets),
        active_race=keys(models.race_nets),
        active_recognition=set(),
        gallery={},
        active_glasses=set(),
        active_mask=set(),
        active_hair_color=keys(models.hair_color_nets),
        active_eye_color=keys(models.eye_color_nets),
        active_face_landmarks=set(),
        active_hands=set(),
        active_gaze=set(),
        global_adjustments={},
        face_adjustments={},
        face_detector="ssd",
    )


def _clean(value):
    """Recursively drop arrays, round floats to 3 dp, and turn tuples into lists."""
    if isinstance(value, dict):
        return {
            k: [int(c) for c in v] if k == "box" else _clean(v)
            for k, v in value.items() if not isinstance(v, np.ndarray)
        }
    if isinstance(value, (list, tuple)):
        items = [_clean(v) for v in value if not isinstance(v, np.ndarray)]
        # Model outputs follow set iteration order, which varies with string hash seeding.
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    if isinstance(value, np.integer):
        return int(value)
    return value


def normalize(faces: list[dict]) -> list[dict]:
    """Reduce per-face results to stable JSON: no images, rounded floats, sorted by box."""
    cleaned = [_clean(face) for face in faces]
    return sorted(cleaned, key=lambda f: (f["box"][0], f["box"][1]))


def test_analyze_frame_matches_golden():
    """Per-face output on the fixture photo must equal the committed golden snapshot."""
    for path in REQUIRED_MODELS:
        require_model(path)
    models = load_models()
    frame = cv2.imread(str(FIXTURE))
    _, faces, *_ = analyze_frame(models, frame, **_config(models))
    actual = normalize(faces)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip("golden updated")
    assert actual == json.loads(GOLDEN.read_text())
