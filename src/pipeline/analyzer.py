"""Pipeline analyzer orchestrating multi-face detection, attribute prediction, fusion, and drawing."""
from __future__ import annotations

import numpy as np

try:
    from src.core.types import Models
    from src.pipeline.config import AnalysisConfig
    from src.pipeline.stages import (
        _analyze_face,
        _annotate,
        _detect,
        _face_record,
        _frame_inputs,
        _prepare_frame,
        _run_whole_frame_features,
    )
except ImportError:
    from core.types import Models
    from pipeline.config import AnalysisConfig
    from pipeline.stages import (
        _analyze_face,
        _annotate,
        _detect,
        _face_record,
        _frame_inputs,
        _prepare_frame,
        _run_whole_frame_features,
    )


def analyze_frame(
    models: Models, frame: np.ndarray, config: AnalysisConfig
) -> tuple[np.ndarray, list[dict], bool, bool]:
    """Detect faces and run inference for whichever model keys are active per feature.

    Multiple active models for the same feature (e.g. config.active_age = {"caffe", "fairface"})
    all run and are shown together. No Streamlit calls (safe for background threads).
    Returns (annotated frame, one result dict per face, any face found, any hand found).

    config.global_adjustments apply to the whole frame first, so every output of this call sees
    the adjusted pixels; config.face_adjustments apply again per face, to that face's crop only.

    config.tracker is optional and stays None for single-image callers (upload/snapshot have no
    "next frame" for an ID to persist into). When a FaceTracker is passed -- video/webcam LIVE
    mode only -- each face's dict also carries a stable "track_id" (see FaceTracker), and the
    number burned into the annotated frame is that track_id instead of this frame's
    detection-order position, so tracking is visible, not just data the caller ignores.

    config.liveness_tracker is only ever passed by video/webcam LIVE mode -- a single static
    image has no blink transitions to observe, so liveness is unavailable there by design (not
    just unchecked): analyze_frame skips liveness entirely -- no "liveness" pairs, no
    LivenessResult -- whenever liveness_tracker is None, regardless of active_liveness.
    config.active_liveness additionally gates it off within LIVE mode itself (unchecked box =
    skipped); None defaults to every loaded backend.
    """
    frame = _prepare_frame(frame, config)
    annotated_frame = frame.copy()
    face_boxes = _detect(models, frame, config)
    tracker = config.tracker
    track_ids = tracker.update(face_boxes) if tracker is not None else [None] * len(face_boxes)
    hands_detected = _run_whole_frame_features(models, frame, annotated_frame, config)
    shared = _frame_inputs(models, config)

    cropped_faces = []
    for idx, (box, track_id) in enumerate(zip(face_boxes, track_ids, strict=False), 1):
        analysis = _analyze_face(models, frame, box, track_id, config, shared)
        if analysis is None:
            continue
        _annotate(models, annotated_frame, idx, analysis, config)
        cropped_faces.append(_face_record(idx, analysis))

    return annotated_frame, cropped_faces, bool(face_boxes), hands_detected


AGGREGATE_FEATURES = ("age", "gender", "race")


def aggregate_demographics(cropped_faces: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    """Whole-image demographic aggregate over already-computed per-face results."""
    totals: dict[str, dict[str, dict[str, int]]] = {feature: {} for feature in AGGREGATE_FEATURES}
    for face in cropped_faces:
        for column, value in face["raw_columns"].items():
            for feature in AGGREGATE_FEATURES:
                prefix = f"{feature}_"
                if not column.startswith(prefix) or not value:
                    continue
                model_key = column[len(prefix):]
                bucket = totals[feature].setdefault(model_key, {})
                bucket[value] = bucket.get(value, 0) + 1
    return {feature: models for feature, models in totals.items() if models}


__all__ = [
    "analyze_frame",
    "aggregate_demographics",
    "AGGREGATE_FEATURES",
]
