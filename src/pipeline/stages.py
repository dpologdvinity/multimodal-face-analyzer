"""Stage helpers that analyze_frame orchestrates: prepare, detect, per-face inference, annotate."""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from src.attributes import (
        _estimate_roll_angle,
        _format_race_label,
        _rotate_region,
        caffe_probabilities,
        crop_face_dex,
        deepface_probabilities,
        dex_age_estimate,
        fairface_age_label,
        fairface_gender_label,
        fairface_landmarks_from_mediapipe,
        fairface_probabilities,
        fairface_race_label,
        format_dex_age,
        mivolo_age_estimate,
        mivolo_estimate,
        predict_emotion_dan,
        predict_emotion_ferplus,
        predict_emotion_hsemotion,
        predict_emotion_mini_xception,
        predict_eye_color_colorimetric,
        predict_glasses_mobilenet,
        predict_hair_color_colorimetric,
        predict_mask_mobilenetv2,
        predict_texture_artifact_score,
    )
    from src.core.constants import (
        AGE_LIST,
        AGE_LIST_RANGES,
        BEST_MODEL_KEY,
        FAIRFACE_AGE_RANGES,
        GENDER_LIST,
        MODEL_MEAN_VALUES,
        RACE_LABELS_DEEPFACE,
        RACE_LABELS_FAIRFACE,
    )
    from src.core.image_utils import (
        apply_image_adjustments,
        crop_region,
        face_crop_bounds,
    )
    from src.core.types import Models
    from src.detectors import (
        detect_faces,
        detect_faces_retinaface,
        detect_faces_scrfd,
        detect_faces_yolo,
    )
    from src.fusion import (
        _format_results,
        _gather_face_results,
        canonical_race_probabilities,
        fuse_emotion,
        fuse_gender,
        fuse_race,
        select_age,
        with_headline,
    )
    from src.gallery import (
        compute_face_embedding,
        match_face_identity,
        predict_identity_lbph,
        train_lbph_recognizer,
    )
    from src.liveness import (
        assess_static_liveness,
        blink_score_from_landmarker,
    )
    from src.pipeline.config import AnalysisConfig
    from src.pipeline.drawing import (
        draw_face_landmarks,
        draw_hand_landmarks,
        draw_outlined_text,
    )
    from src.pipeline.landmarks import (
        _detect_face_landmarker,
        detect_hand_landmarks_mediapipe,
        predict_face_landmarks_mediapipe,
        predict_gaze_mediapipe,
        predict_head_pose_mediapipe,
    )
except ImportError:
    from attributes import (
        _estimate_roll_angle,
        _format_race_label,
        _rotate_region,
        caffe_probabilities,
        crop_face_dex,
        deepface_probabilities,
        dex_age_estimate,
        fairface_age_label,
        fairface_gender_label,
        fairface_landmarks_from_mediapipe,
        fairface_probabilities,
        fairface_race_label,
        format_dex_age,
        mivolo_age_estimate,
        mivolo_estimate,
        predict_emotion_dan,
        predict_emotion_ferplus,
        predict_emotion_hsemotion,
        predict_emotion_mini_xception,
        predict_eye_color_colorimetric,
        predict_glasses_mobilenet,
        predict_hair_color_colorimetric,
        predict_mask_mobilenetv2,
        predict_texture_artifact_score,
    )
    from core.constants import (
        AGE_LIST,
        AGE_LIST_RANGES,
        BEST_MODEL_KEY,
        FAIRFACE_AGE_RANGES,
        GENDER_LIST,
        MODEL_MEAN_VALUES,
        RACE_LABELS_DEEPFACE,
        RACE_LABELS_FAIRFACE,
    )
    from core.image_utils import (
        apply_image_adjustments,
        crop_region,
        face_crop_bounds,
    )
    from core.types import Models
    from detectors import (
        detect_faces,
        detect_faces_retinaface,
        detect_faces_scrfd,
        detect_faces_yolo,
    )
    from fusion import (
        _format_results,
        _gather_face_results,
        canonical_race_probabilities,
        fuse_emotion,
        fuse_gender,
        fuse_race,
        select_age,
        with_headline,
    )
    from gallery import (
        compute_face_embedding,
        match_face_identity,
        predict_identity_lbph,
        train_lbph_recognizer,
    )
    from liveness import (
        assess_static_liveness,
        blink_score_from_landmarker,
    )
    from pipeline.config import AnalysisConfig
    from pipeline.drawing import (
        draw_face_landmarks,
        draw_hand_landmarks,
        draw_outlined_text,
    )
    from pipeline.landmarks import (
        _detect_face_landmarker,
        detect_hand_landmarks_mediapipe,
        predict_face_landmarks_mediapipe,
        predict_gaze_mediapipe,
        predict_head_pose_mediapipe,
    )


_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(4, (os.cpu_count() or 4)), thread_name_prefix="inference"
)

PREDICTION_CACHE_MAX_SIZE = 2048
_PREDICTION_CACHE: OrderedDict[tuple, object] = OrderedDict()


def _cached_face_predict(feature: str, model_key: str, face_bgr: np.ndarray, predict_fn: Any, *args: Any) -> Any:
    """Memoize a predict_*(net, face, ...) call on (feature, model_key, hash(image bytes))."""
    cache_key = (
        feature,
        model_key,
        face_bgr.shape,
        hashlib.blake2b(face_bgr.tobytes(), digest_size=16).digest(),
    )
    cached = _PREDICTION_CACHE.get(cache_key)
    if cached is not None or cache_key in _PREDICTION_CACHE:
        _PREDICTION_CACHE.move_to_end(cache_key)
        return cached
    value = predict_fn(*args)
    _PREDICTION_CACHE[cache_key] = value
    if len(_PREDICTION_CACHE) > PREDICTION_CACHE_MAX_SIZE:
        _PREDICTION_CACHE.popitem(last=False)
    return value


def _record_model_latency(metrics: dict | None, feature: str, model: str, started: float) -> None:
    """Append per-model inference latency (ms) to metrics dict for performance monitoring."""
    if metrics is None:
        return
    metrics.setdefault("model_latency_ms", {}).setdefault(f"{feature}/{model}", []).append(
        (time.perf_counter() - started) * 1000
    )


@dataclass
class _FrameInputs:
    """Per-frame setup shared by every detected face."""

    active_liveness: set
    need_blob227: bool
    eye_cascade: Any
    lbph_trained: Any


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


def _prepare_frame(frame: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    """Apply global_adjustments to the whole frame, before face detection even runs."""
    global_adjustments = config.global_adjustments
    if global_adjustments and any(global_adjustments.values()):
        frame = apply_image_adjustments(frame, global_adjustments)
    return frame


def _detect(models: Models, frame: np.ndarray, config: AnalysisConfig) -> list:
    """Run the one configured face detector; "yolo"/"scrfd"/"retinaface" fall back to SSD if not loaded.

    Unlike every other feature, exactly one detector runs per frame -- running two and merging
    their boxes would just produce duplicate/overlapping faces, not a meaningfully combined result.
    """
    conf_threshold = config.conf_threshold
    face_detector = config.face_detector
    yolo_net = models.yolo_face_nets.get("yolo")
    scrfd_net = models.scrfd_face_nets.get("scrfd")
    retinaface_net = models.retinaface_nets.get("retinaface")

    if face_detector == "yolo" and yolo_net is not None:
        face_boxes = _cached_face_predict(
            "face_detection", f"yolo:{conf_threshold}", frame, detect_faces_yolo, yolo_net, frame, conf_threshold
        )
    elif face_detector == "scrfd" and scrfd_net is not None:
        face_boxes = _cached_face_predict(
            "face_detection", f"scrfd:{conf_threshold}", frame, detect_faces_scrfd, scrfd_net, frame, conf_threshold
        )
    elif face_detector == "retinaface" and retinaface_net is not None:
        face_boxes = _cached_face_predict(
            "face_detection", f"retinaface:{conf_threshold}", frame, detect_faces_retinaface, retinaface_net, frame, conf_threshold
        )
    else:
        face_boxes = _cached_face_predict(
            "face_detection", f"ssd:{conf_threshold}", frame, detect_faces, models.face_net, frame, conf_threshold
        )
    return face_boxes


def _run_whole_frame_features(
    models: Models, frame: np.ndarray, annotated_frame: np.ndarray, config: AnalysisConfig
) -> bool:
    """Draw whole-frame hand landmarks onto annotated_frame; return whether any hand was found."""
    hands_detected = False
    hand_net = models.hand_nets.get("mediapipe")
    if hand_net is not None and "mediapipe" in config.active_hands:
        hands = _cached_face_predict("hand_landmarks", "mediapipe", frame, detect_hand_landmarks_mediapipe, hand_net, frame)
        if hands:
            hands_detected = True
            draw_hand_landmarks(annotated_frame, hands)
    return hands_detected


def _frame_inputs(models: Models, config: AnalysisConfig) -> _FrameInputs:
    """Resolve the per-frame setup (liveness default, shared Caffe blob, roll cascade, LBPH)."""
    active_liveness = config.active_liveness
    if active_liveness is None:
        active_liveness = set(models.liveness_nets)
    active_age, active_gender = config.active_age, config.active_gender
    need_blob227 = ("caffe" in active_age and "caffe" in models.age_nets) or \
                   ("caffe" in active_gender and "caffe" in models.gender_nets)
    eye_cascade = models.eye_color_nets.get("colorimetric")
    lbph_trained = train_lbph_recognizer() if "lbph" in config.active_recognition and models.recognition_nets.get("lbph") else None
    return _FrameInputs(active_liveness, need_blob227, eye_cascade, lbph_trained)


def _prepare_face(
    models: Models, frame: np.ndarray, box: tuple, track_id: Any, config: AnalysisConfig, shared: _FrameInputs,
) -> _FaceInputs | None:
    """Roll-align, crop and face-adjust one face, then compute its shared landmarks and blobs.

    face_adjustments apply to this face's own crop only, after detection but before
    classification -- they affect just this face's thumbnail/attributes, not the shared
    frame or other faces. Returns None when the crop is empty.
    """
    x1, y1, x2, y2 = box
    active_age, active_gender = config.active_age, config.active_gender
    face_adjustments = config.face_adjustments
    crop_frame, (cx1, cy1, cx2, cy2) = frame, (x1, y1, x2, y2)
    if shared.eye_cascade is not None:
        probe = frame[max(0, y1 - 20):min(y2 + 20, frame.shape[0]), max(0, x1 - 20):min(x2 + 20, frame.shape[1])]
        angle = (
            _cached_face_predict("roll_angle", "haarcascade", probe, _estimate_roll_angle, probe, shared.eye_cascade)
            if probe.size else None
        )
        if angle is not None and abs(angle) > 3:
            crop_frame, (cx1, cy1, cx2, cy2) = _rotate_region(frame, (x1, y1, x2, y2), angle)

    x1_crop, y1_crop, x2_crop, y2_crop = face_crop_bounds(
        (cx1, cy1, cx2, cy2), crop_frame.shape[:2],
    )

    face = crop_frame[y1_crop:y2_crop, x1_crop:x2_crop]
    if face.size == 0:
        return None

    if face_adjustments and any(face_adjustments.values()):
        face = apply_image_adjustments(face, face_adjustments)

    run_liveness = config.liveness_tracker is not None and "mediapipe" in shared.active_liveness
    liveness_net = models.liveness_nets.get("mediapipe") if run_liveness else None
    face_landmarker = (
        liveness_net if liveness_net is not None
        else models.face_landmarks_nets.get("mediapipe")
    )
    needs_face_landmarks = (
        "mediapipe" in config.active_gaze
        or "mediapipe" in config.active_face_landmarks
        or face_landmarker is not None
    )
    landmarker_result = (
        _detect_face_landmarker(face_landmarker, face)
        if face_landmarker is not None and needs_face_landmarks
        else None
    )
    fairface_landmarks = None
    points = None
    if landmarker_result is not None and face_landmarker is not None:
        points = predict_face_landmarks_mediapipe(face_landmarker, face, landmarker_result)
        if points is not None:
            local_landmarks = fairface_landmarks_from_mediapipe(points, face.shape[1], face.shape[0])
            if local_landmarks is not None:
                local_landmarks += np.array([x1_crop, y1_crop], dtype=np.float32)
                fairface_landmarks = local_landmarks
    texture_score = predict_texture_artifact_score(face) if run_liveness else 0.0
    blink_score = blink_score_from_landmarker(landmarker_result) if run_liveness else None

    blob227 = None
    if shared.need_blob227:
        blob227 = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    mivolo_net = models.age_nets.get("mivolo") or models.gender_nets.get("mivolo")
    mivolo_wanted = "mivolo" in active_age or "mivolo" in active_gender
    mivolo_result = None
    mivolo_age_result = None
    if mivolo_net is not None and mivolo_wanted:
        if "mivolo" in active_age and "mivolo" not in active_gender:
            mivolo_age_result = _cached_face_predict(
                "mivolo_age", "face", face, mivolo_age_estimate, mivolo_net, face,
            )
        else:
            mivolo_result = _cached_face_predict(
                "mivolo", "face", face, mivolo_estimate, mivolo_net, face,
            )

    return _FaceInputs(
        frame=frame, box=(x1, y1, x2, y2), crop_frame=crop_frame, crop_box=(cx1, cy1, cx2, cy2),
        face=face, track_id=track_id,
        landmarker_result=landmarker_result, points=points, fairface_landmarks=fairface_landmarks,
        run_liveness=run_liveness, texture_score=texture_score, blink_score=blink_score,
        blob227=blob227, mivolo_result=mivolo_result, mivolo_age_result=mivolo_age_result,
    )


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
    """Run every active race backend; return (per-model pairs, fused answer)."""
    face, metrics = inputs.face, config.metrics
    pairs, distributions = [], {}
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
            distributions[key] = canonical_race_probabilities(probs, RACE_LABELS_FAIRFACE)
        else:
            probs = _cached_face_predict("race_probs", key, face, deepface_probabilities, net, face)
            value = _format_race_label(probs, RACE_LABELS_DEEPFACE)
            distributions[key] = canonical_race_probabilities(probs, RACE_LABELS_DEEPFACE)
        pairs.append((key, value))
        _record_model_latency(metrics, "race", key, started)
    return pairs, fuse_race(distributions)


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


def _analyze_face(
    models: Models, frame: np.ndarray, box: tuple, track_id: Any, config: AnalysisConfig, shared: _FrameInputs,
) -> dict | None:
    """Run every attribute task for one face in parallel; None when the crop is empty.

    Returns {"inputs": _FaceInputs, <feature>: that task's result} for _annotate/_face_record.
    """
    inputs = _prepare_face(models, frame, box, track_id, config, shared)
    if inputs is None:
        return None
    submit = _INFERENCE_EXECUTOR.submit
    futures = {
        "age": submit(_age_task, models, config, inputs),
        "gender": submit(_gender_task, models, config, inputs),
        "emotion": submit(_emotion_task, models, config, inputs),
        "race": submit(_race_task, models, config, inputs),
        "gaze": submit(_gaze_task, models, config, inputs),
        "head_pose": submit(_head_pose_task, models, config, inputs),
        "recognition": submit(_recognition_task, models, config, inputs, shared.lbph_trained),
        "glasses": submit(_glasses_task, models, config, inputs),
        "mask": submit(_mask_task, models, config, inputs),
        "hair_color": submit(_hair_color_task, models, config, inputs),
        "eye_color": submit(_eye_color_task, models, config, inputs),
        "liveness": submit(_liveness_task, config, inputs),
    }
    # Collected in submission order, so the first failing task's exception is the one raised.
    results = {feature: future.result() for feature, future in futures.items()}

    emotion_pairs = results["emotion"][0]
    if config.metrics is not None and emotion_pairs:
        config.metrics.setdefault("emotion_samples", []).extend(
            {"model": key, "emotion": value} for key, value in emotion_pairs
        )
    return {"inputs": inputs, **results}


def _annotate(
    models: Models, annotated_frame: np.ndarray, idx: int, analysis: dict, config: AnalysisConfig,
) -> None:
    """Draw one face's box, its number and (if active) its face-mesh overlay.

    The number is the tracker's stable track_id in LIVE mode, else the detection-order index.
    """
    inputs: _FaceInputs = analysis["inputs"]
    x1, y1, x2, y2 = inputs.box
    track_id = inputs.track_id
    box_thickness = int(round(annotated_frame.shape[0] / 150)) or 1
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness, 8)
    display_id = track_id if track_id is not None else idx
    draw_outlined_text(annotated_frame, str(display_id), (x1, max(20, y1 - 10)), (0, 255, 255))

    landmarks_net = models.face_landmarks_nets.get("mediapipe")
    if landmarks_net is not None and "mediapipe" in config.active_face_landmarks:
        landmark_points = predict_face_landmarks_mediapipe(landmarks_net, inputs.face, inputs.landmarker_result)
        if landmark_points is not None:
            draw_face_landmarks(annotated_frame, landmark_points, (x1, y1, x2, y2))


def _face_record(idx: int, analysis: dict) -> dict:
    """Assemble one face's result dict: headlines, formatted outputs and table rows."""
    inputs: _FaceInputs = analysis["inputs"]
    age_pairs, best_age = analysis["age"]
    gender_pairs, fused_gender = analysis["gender"]
    emotion_pairs, fused_emotion = analysis["emotion"]
    race_pairs, fused_race = analysis["race"]
    gaze_pairs = analysis["gaze"]
    head_pose_pairs = analysis["head_pose"]
    recognition_pairs, face_embedding = analysis["recognition"]
    glasses_pairs = analysis["glasses"]
    mask_pairs = analysis["mask"]
    hair_color_pairs = analysis["hair_color"]
    eye_color_pairs = analysis["eye_color"]
    liveness_pairs, liveness_result = analysis["liveness"]

    eye_contact = [f"{key}=yes" if value.startswith("center/") else f"{key}=no" for key, value in gaze_pairs]

    age_pairs = with_headline(
        age_pairs,
        f"{best_age[0]} ({best_age[1]})" if best_age else None,
        BEST_MODEL_KEY,
    )
    gender_pairs = with_headline(gender_pairs, fused_gender)
    race_pairs = with_headline(race_pairs, fused_race)
    emotion_pairs = with_headline(emotion_pairs, fused_emotion)

    raw_columns = _gather_face_results({
        "age": age_pairs, "gender": gender_pairs, "race": race_pairs, "emotion": emotion_pairs,
        "gaze": gaze_pairs, "identity": recognition_pairs,
        "eye_contact": [("derived", value) for value in eye_contact], "head_pose": head_pose_pairs,
        "glasses": glasses_pairs, "mask": mask_pairs,
        "hair_color": hair_color_pairs, "eye_color": eye_color_pairs,
        "liveness": liveness_pairs,
    })
    model_results = [
        {"Feature": feature.replace("_", " ").upper(), "Model": model, "Output": str(value)}
        for feature, pairs in {
            "age": age_pairs,
            "gender": gender_pairs, "race": race_pairs, "emotion": emotion_pairs,
            "gaze": gaze_pairs, "identity": recognition_pairs,
            "eye contact": [("derived", value) for value in eye_contact],
            "head pose": head_pose_pairs,
            "glasses": glasses_pairs,
            "mask": mask_pairs, "hair color": hair_color_pairs, "eye color": eye_color_pairs,
            "liveness": liveness_pairs,
        }.items()
        for model, value in pairs
    ]

    return {
        "idx": idx,
        "track_id": inputs.track_id,
        "box": inputs.box,
        "image": cv2.cvtColor(inputs.face, cv2.COLOR_BGR2RGB),
        "age": _format_results(age_pairs),
        "headline": {
            "age": best_age[0] if best_age else None,
            "gender": fused_gender,
            "race": fused_race,
            "emotion": fused_emotion,
        },
        "gender": _format_results(gender_pairs),
        "race": _format_results(race_pairs),
        "emotion": _format_results(emotion_pairs),
        "gaze": _format_results(gaze_pairs),
        "eye_contact": eye_contact,
        "head_pose": _format_results(head_pose_pairs),
        "identity": _format_results(recognition_pairs),
        "glasses": _format_results(glasses_pairs),
        "mask": _format_results(mask_pairs),
        "hair_color": _format_results(hair_color_pairs),
        "eye_color": _format_results(eye_color_pairs),
        "liveness": [liveness_result.summary] if liveness_result is not None else [],
        "liveness_status": liveness_result.status if liveness_result is not None else None,
        "blink_count": liveness_result.blink_count if liveness_result is not None else None,
        "blink_rate": liveness_result.blink_rate if liveness_result is not None else None,
        "texture_score": liveness_result.texture_score if liveness_result is not None else None,
        "texture_artifact": liveness_result.texture_artifact if liveness_result is not None else None,
        "embedding": face_embedding.tolist() if face_embedding is not None else None,
        "raw_columns": raw_columns,
        "model_results": model_results,
    }


__all__ = [
    "_INFERENCE_EXECUTOR",
    "_PREDICTION_CACHE",
    "PREDICTION_CACHE_MAX_SIZE",
    "_analyze_face",
    "_annotate",
    "_cached_face_predict",
    "_detect",
    "_face_record",
    "_frame_inputs",
    "_prepare_frame",
    "_record_model_latency",
    "_run_whole_frame_features",
]
