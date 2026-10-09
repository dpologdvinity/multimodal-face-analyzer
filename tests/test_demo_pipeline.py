"""Regression test: the public demo's pipeline (RetinaFace, no MediaPipe) on the crew photo."""
from pathlib import Path

import cv2
import pytest

from face_analyzer.core import constants as C
from tests._models import require_model

FIXTURE = Path(__file__).parent / "fixtures" / "crew_portrait.jpg"


def test_demo_path_gets_both_women_on_the_crew_photo_right(monkeypatch):
    """Align FairFace on RetinaFace's landmarks so McClain and Koch read Female, as with MediaPipe."""
    pytest.importorskip("onnxruntime")
    for path in (C.RETINAFACE_MODEL, C.FAIRFACE_MODEL):
        require_model(path)
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    # The demo host has no mediapipe package; keep its landmarker off where it is installed.
    monkeypatch.setenv("FACE_LANDMARKS_MODEL", "")
    monkeypatch.setenv("LIVENESS_MODEL", "")
    from face_analyzer.inference import AnalysisConfig, analyze_frame, load_models

    models = load_models()
    assert not models.face_landmarks_nets
    config = AnalysisConfig(
        face_detector="retinaface", active_age={"fairface"}, active_gender={"fairface"},
        active_race={"fairface"},
    )
    _, faces, _, _ = analyze_frame(models, cv2.imread(str(FIXTURE)), config)

    left_to_right = sorted(faces, key=lambda face: face["box"][0])
    assert len(left_to_right) == 6
    genders = [face["raw_columns"]["gender_fairface"] for face in left_to_right]
    # Anne McClain is second from the left and Christina Koch on the right; the four men stay Male.
    assert genders == ["Male", "Female", "Male", "Male", "Male", "Female"]
