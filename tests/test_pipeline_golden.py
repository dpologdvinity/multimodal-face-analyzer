"""Golden characterization test: analyze_frame output on a fixed public-domain photo."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_analyzer.core import constants as C
from face_analyzer.inference import AnalysisConfig, analyze_frame, load_models
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


# Pinned so the golden does not depend on which optional runtimes (torch/TF/mediapipe) are installed.
EXPECTED_ACTIVE = {
    "age": {"caffe", "dex", "fairface"},
    "gender": {"caffe", "fairface"},
    "emotion": {"ferplus", "hsemotion"},
    "race": {"fairface"},
    "hair_color": {"colorimetric"},
    "eye_color": {"colorimetric"},
}


def _config(models) -> AnalysisConfig:
    """Build the golden run's AnalysisConfig, checking the pinned backends loaded."""
    loaded = {
        "age": models.age_nets,
        "gender": models.gender_nets,
        "emotion": models.emotion_nets,
        "race": models.race_nets,
        "hair_color": models.hair_color_nets,
        "eye_color": models.eye_color_nets,
    }
    for feature, expected in EXPECTED_ACTIVE.items():
        missing = expected - set(loaded[feature])
        if missing:
            # Explicit raise: a bare assert is stripped under -O and would let a partial
            # backend set silently produce a different golden output.
            raise AssertionError(f"golden backends failed to load for {feature}: {sorted(missing)}")
    return AnalysisConfig(
        conf_threshold=0.5,
        active_age=set(EXPECTED_ACTIVE["age"]),
        active_gender=set(EXPECTED_ACTIVE["gender"]),
        active_emotion=set(EXPECTED_ACTIVE["emotion"]),
        active_race=set(EXPECTED_ACTIVE["race"]),
        active_hair_color=set(EXPECTED_ACTIVE["hair_color"]),
        active_eye_color=set(EXPECTED_ACTIVE["eye_color"]),
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
    _, faces, *_ = analyze_frame(models, frame, _config(models))
    actual = normalize(faces)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip("golden updated")
    assert actual == json.loads(GOLDEN.read_text())
