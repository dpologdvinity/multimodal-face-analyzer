"""Per-feature attribute tasks that run in parallel on one detected face."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..attributes import (
    _format_race_label,
    caffe_probabilities,
    crop_face_dex,
    deepface_probabilities,
    dex_age_estimate,
    fairface_age_label,
    fairface_gender_label,
    fairface_probabilities,
    fairface_race_label,
    format_dex_age,
    predict_emotion_dan,
    predict_emotion_ferplus,
    predict_emotion_hsemotion,
    predict_emotion_mini_xception,
    predict_eye_color_colorimetric,
    predict_glasses_mobilenet,
    predict_hair_color_colorimetric,
    predict_mask_mobilenetv2,
)
from ..core.constants import (
    AGE_LIST,
    AGE_LIST_RANGES,
    FAIRFACE_AGE_RANGES,
    GENDER_LIST,
    RACE_LABELS_DEEPFACE,
)
from ..core.image_utils import (
    apply_image_adjustments,
    crop_region,
)
from ..core.types import (
    Models,
)
from ..fusion import (
    fuse_emotion,
    fuse_gender,
    select_age,
    select_race,
)
from ..gallery import (
    compute_face_embedding,
    match_face_identity,
    predict_identity_lbph,
)
from ..liveness import (
    assess_static_liveness,
)
from .cache import (
    _cached_face_predict,
    _record_model_latency,
)
from .config import (
    AnalysisConfig,
)
from .landmarks import (
    predict_gaze_mediapipe,
    predict_head_pose_mediapipe,
)


@dataclass
class _FaceInputs:
    """One detected face's crops and landmarks, shared by every attribute task."""

    frame: np.ndarray
    box: tuple[int, int, int, int]
    crop_frame: np.ndarray
    crop_box: tuple[int, int, int, int]
    face: np.ndarray
    track_id: Any
    landmarker_result: Any
    points: list[tuple[float, float]] | None
    fairface_landmarks: np.ndarray | None
    run_liveness: bool
    texture_score: float
    blink_score: Any
    blob227: np.ndarray | None
    mivolo_result: Any
    mivolo_age_result: Any


def _age_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> tuple[list, Any]:
    """Run every active age backend; return (per-model pairs, most reliable model's answer)."""
    face, metrics, face_adjustments = inputs.face, config.metrics, config.face_adjustments
    pairs, estimates = [], {}
    for key in config.active_age:
        net = models.age_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        if key == "caffe":
            probs = _cached_face_predict("age_probs", key, face, caffe_probabilities, net, inputs.blob227)
            bucket = int(np.argmax(probs))
            value = AGE_LIST[bucket]
            estimates[key] = float(np.mean(AGE_LIST_RANGES[bucket]))
        elif key == "fairface":
            probs = _cached_face_predict(
                "age_probs", key, face, fairface_probabilities,
                net, inputs.crop_frame, inputs.crop_box, "age_output", inputs.fairface_landmarks,
            )
            value = fairface_age_label(probs)
            estimates[key] = float(np.mean(FAIRFACE_AGE_RANGES[int(np.argmax(probs))]))
        elif key == "dex":
            dex_face = crop_face_dex(inputs.frame, inputs.box)
            if face_adjustments and any(face_adjustments.values()):
                dex_face = apply_image_adjustments(dex_face, face_adjustments)
            estimate = _cached_face_predict("age_estimate", key, dex_face, dex_age_estimate, net, dex_face)
            value = format_dex_age(estimate)
            if estimate is not None:
                estimates[key] = estimate[0]
        elif key == "mivolo":
            mivolo_result, mivolo_age_result = inputs.mivolo_result, inputs.mivolo_age_result
            if mivolo_result is None and mivolo_age_result is None:
                continue
            age = mivolo_result[0] if mivolo_result is not None else mivolo_age_result
            value = f"{age:.0f}"
            estimates[key] = age
        pairs.append((key, value))
        _record_model_latency(metrics, "age", key, started)
    return pairs, select_age(estimates)


def _gender_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> tuple[list, Any]:
    """Run every active gender backend; return (per-model pairs, fused answer)."""
    face, metrics = inputs.face, config.metrics
    pairs, male_probabilities = [], {}
    for key in config.active_gender:
        net = models.gender_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        if key == "caffe":
            probs = _cached_face_predict("gender_probs", key, face, caffe_probabilities, net, inputs.blob227)
            value = GENDER_LIST[int(np.argmax(probs))]
            male_probabilities[key] = float(probs[0] / (probs.sum() or 1.0))
        elif key == "deepface":
            probs = _cached_face_predict("gender_probs", key, face, deepface_probabilities, net, face)
            value = "Male" if np.argmax(probs) == 1 else "Female"
            male_probabilities[key] = float(probs[1] / (probs.sum() or 1.0))
        elif key == "fairface":
            probs = _cached_face_predict(
                "gender_probs", key, face, fairface_probabilities,
                net, inputs.crop_frame, inputs.crop_box, "gender_output", inputs.fairface_landmarks,
            )
            value = fairface_gender_label(probs)
            male_probabilities[key] = float(probs[0])
        elif key == "mivolo":
            if inputs.mivolo_result is None:
                continue
            value = inputs.mivolo_result[1]
            male_probabilities[key] = 1.0 if value == "Male" else 0.0
        pairs.append((key, value))
        _record_model_latency(metrics, "gender", key, started)
    return pairs, fuse_gender(male_probabilities)


def _emotion_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> tuple[list, Any]:
    """Run every active emotion backend; return (per-model pairs, fused answer)."""
    face, metrics, face_adjustments = inputs.face, config.metrics, config.face_adjustments
    pairs = []
    for key in config.active_emotion:
        net = models.emotion_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        if key == "dan":
            value = _cached_face_predict("emotion", key, face, predict_emotion_dan, net, face)
        elif key == "mini_xception":
            value = _cached_face_predict("emotion", key, face, predict_emotion_mini_xception, net, face)
        elif key == "ferplus":
            ferplus_face = crop_region(inputs.frame, *inputs.box)
            if face_adjustments and any(face_adjustments.values()):
                ferplus_face = apply_image_adjustments(ferplus_face, face_adjustments)
            value = _cached_face_predict(
                "emotion", key, ferplus_face, predict_emotion_ferplus, net, ferplus_face,
            )
        else:
            value = _cached_face_predict("emotion", key, face, predict_emotion_hsemotion, net, face)
        pairs.append((key, value))
        _record_model_latency(metrics, "emotion", key, started)
    return pairs, fuse_emotion(dict(pairs))


def _race_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> tuple[list, Any]:
    """Run every active race backend; return (per-model pairs, best-model answer)."""
    face, metrics = inputs.face, config.metrics
    pairs = []
    for key in config.active_race:
        net = models.race_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        if key == "fairface":
            probs = _cached_face_predict(
                "race_probs", key, face, fairface_probabilities,
                net, inputs.crop_frame, inputs.crop_box, "race_output", inputs.fairface_landmarks,
            )
            value = fairface_race_label(probs)
        else:
            probs = _cached_face_predict("race_probs", key, face, deepface_probabilities, net, face)
            value = _format_race_label(probs, RACE_LABELS_DEEPFACE)
        pairs.append((key, value))
        _record_model_latency(metrics, "race", key, started)
    return pairs, select_race(dict(pairs))


def _gaze_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> list:
    """Run every active gaze backend on the face's shared landmarker result."""
    pairs = []
    for key in config.active_gaze:
        net = models.gaze_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        value = predict_gaze_mediapipe(net, inputs.face, inputs.landmarker_result)
        pairs.append((key, value))
        _record_model_latency(config.metrics, "gaze", key, started)
    return pairs


def _head_pose_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> list:
    """Estimate head pose with every active gaze backend (they share the landmarker)."""
    pairs = []
    for key in config.active_gaze:
        net = models.gaze_nets.get(key)
        if net is not None:
            pairs.append((key, predict_head_pose_mediapipe(net, inputs.face, inputs.landmarker_result)))
    return pairs


def _recognition_task(
    models: Models, config: AnalysisConfig, inputs: _FaceInputs, lbph_trained: Any,
) -> tuple[list, Any]:
    """Match the face against the gallery (embeddings) or the trained LBPH recognizer."""
    face = inputs.face
    pairs = []
    embedding = None
    for key in config.active_recognition:
        net = models.recognition_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        if key == "lbph":
            if lbph_trained is None:
                value = "UNKNOWN"
            else:
                recognizer, label_names = lbph_trained
                match = predict_identity_lbph(recognizer, label_names, face)
                value = f"{match[0]} ({match[1]:.0f})" if match else "UNKNOWN"
        else:
            embedding = _cached_face_predict("embedding", key, face, compute_face_embedding, net, face)
            match = match_face_identity(embedding, config.gallery)
            value = f"{match[0]} ({match[1] * 100:.0f}%)" if match else "UNKNOWN"
        pairs.append((key, value))
        _record_model_latency(config.metrics, "recognition", key, started)
    return pairs, embedding


def _glasses_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> list:
    """Run every active glasses backend."""
    pairs = []
    for key in config.active_glasses:
        net = models.glasses_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        value = _cached_face_predict("glasses", key, inputs.face, predict_glasses_mobilenet, net, inputs.face)
        pairs.append((key, value))
        _record_model_latency(config.metrics, "glasses", key, started)
    return pairs


def _mask_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> list:
    """Run every active mask backend."""
    pairs = []
    for key in config.active_mask:
        net = models.mask_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        value = _cached_face_predict("mask", key, inputs.face, predict_mask_mobilenetv2, net, inputs.face)
        pairs.append((key, value))
        _record_model_latency(config.metrics, "mask", key, started)
    return pairs


def _hair_color_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> list:
    """Classify hair color from the region above the (roll-aligned) face box."""
    pairs = []
    for key in config.active_hair_color:
        if key not in models.hair_color_nets:
            continue
        started = time.perf_counter()
        value = predict_hair_color_colorimetric(inputs.crop_frame, inputs.crop_box)
        pairs.append((key, value))
        _record_model_latency(config.metrics, "hair_color", key, started)
    return pairs


def _eye_color_task(models: Models, config: AnalysisConfig, inputs: _FaceInputs) -> list:
    """Classify eye color, using the face-mesh points when the landmarker found any."""
    face, points = inputs.face, inputs.points
    pairs = []
    for key in config.active_eye_color:
        net = models.eye_color_nets.get(key)
        if net is None:
            continue
        started = time.perf_counter()
        if points is not None:
            value = predict_eye_color_colorimetric(net, face, points)
        else:
            value = _cached_face_predict("eye_color", key, face, predict_eye_color_colorimetric, net, face)
        pairs.append((key, value))
        _record_model_latency(config.metrics, "eye_color", key, started)
    return pairs


def _liveness_task(config: AnalysisConfig, inputs: _FaceInputs) -> tuple[list, Any]:
    """Score liveness for LIVE mode only (a tracker is required to observe blinks over time)."""
    if not inputs.run_liveness:
        return [], None
    started = time.perf_counter()
    if inputs.track_id is not None:
        result = config.liveness_tracker.update(inputs.track_id, inputs.blink_score, inputs.texture_score)
    else:
        result = assess_static_liveness(inputs.texture_score)
    _record_model_latency(config.metrics, "liveness", "mediapipe", started)
    return [("mediapipe", result.summary)], result


__all__ = [
    "_FaceInputs",
    "_age_task",
    "_gender_task",
    "_emotion_task",
    "_race_task",
    "_gaze_task",
    "_head_pose_task",
    "_recognition_task",
    "_glasses_task",
    "_mask_task",
    "_hair_color_task",
    "_eye_color_task",
    "_liveness_task",
]
