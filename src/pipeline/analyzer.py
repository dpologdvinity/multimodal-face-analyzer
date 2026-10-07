"""Pipeline analyzer orchestrating multi-face detection, attribute prediction, fusion, and drawing."""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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


def predict_face_landmarks_mediapipe(
    landmarker: Any, face_bgr: np.ndarray, result: Any = None
) -> list[tuple[float, float]] | None:
    """MediaPipe FaceLandmarker face-mesh points used by the landmark and gaze features."""
    result = result if result is not None else _detect_face_landmarker(landmarker, face_bgr)
    if not result.face_landmarks:
        return None
    return [(lm.x, lm.y) for lm in result.face_landmarks[0]]


def predict_gaze_mediapipe(landmarker: Any, face_bgr: np.ndarray, result: Any = None) -> str:
    """Estimate coarse gaze direction from MediaPipe iris and eye landmarks."""
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr, result)
    if points is None or len(points) < 478:
        return "unknown"

    def center(indices: tuple[int, ...]) -> np.ndarray:
        return np.mean([points[index] for index in indices], axis=0)

    directions = []
    for iris, corners, vertical in (
        ((468, 469, 470, 471, 472), (33, 133), (159, 145)),
        ((473, 474, 475, 476, 477), (362, 263), (386, 374)),
    ):
        iris_center = center(iris)
        left_corner, right_corner = (points[index] for index in corners)
        eye_width = abs(right_corner[0] - left_corner[0])
        eye_height = abs(points[vertical[0]][1] - points[vertical[1]][1])
        if eye_width < 1e-6 or eye_height < 1e-6:
            continue
        horizontal = (iris_center[0] - min(left_corner[0], right_corner[0])) / eye_width
        vertical_position = (iris_center[1] - min(points[index][1] for index in vertical)) / eye_height
        directions.append((horizontal, vertical_position))

    if not directions:
        return "unknown"
    horizontal, vertical_position = np.mean(directions, axis=0)
    horizontal_label = "left" if horizontal < 0.38 else "right" if horizontal > 0.62 else "center"
    vertical_label = "up" if vertical_position < 0.35 else "down" if vertical_position > 0.65 else "level"
    return f"{horizontal_label}/{vertical_label}"


def predict_head_pose_mediapipe(landmarker: Any, face_bgr: np.ndarray, result: Any = None) -> str:
    """Estimate coarse yaw/pitch from stable MediaPipe face landmarks."""
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr, result)
    if points is None or len(points) < 264:
        return "unknown"
    image_points = np.float32([points[i] for i in (1, 152, 33, 263, 61, 291)])
    h, w = face_bgr.shape[:2]
    image_points[:, 0] *= w
    image_points[:, 1] *= h
    model_points = np.float32([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
    ])
    focal = float(w)
    camera = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return "unknown"
    matrix, _ = cv2.Rodrigues(rotation)
    pitch = np.degrees(np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2])))
    yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return f"yaw={yaw:.0f}°, pitch={pitch:.0f}°"


def _record_model_latency(metrics: dict | None, feature: str, model: str, started: float) -> None:
    """Append per-model inference latency (ms) to metrics dict for performance monitoring."""
    if metrics is None:
        return
    metrics.setdefault("model_latency_ms", {}).setdefault(f"{feature}/{model}", []).append(
        (time.perf_counter() - started) * 1000
    )


def analyze_frame(
    models: Models,
    frame: np.ndarray,
    conf_threshold: float,
    active_age: set,
    active_gender: set,
    active_emotion: set,
    active_race: set,
    active_recognition: set,
    gallery: dict,
    active_glasses: set,
    active_mask: set,
    active_hair_color: set,
    active_eye_color: set,
    active_face_landmarks: set,
    active_hands: set,
    active_gaze: set,
    global_adjustments: dict,
    face_adjustments: dict,
    face_detector: str = "yolo",
    metrics: dict | None = None,
    tracker: Any | None = None,
    liveness_tracker: Any | None = None,
    active_liveness: set | None = None,
) -> tuple[np.ndarray, list[dict], bool, bool]:
    """Detect faces and run inference for whichever model keys are active per feature.

    Multiple active models for the same feature (e.g. active_age = {"caffe", "fairface"})
    all run and are shown together. No Streamlit calls (safe for background threads).

    global_adjustments apply to the whole frame first, before face detection even runs --
    every output derived from this call (the annotated image, every face crop, every
    classification) sees the adjusted pixels. face_adjustments apply again, per detected
    face, to that face's own crop only, after detection but before classification -- they
    affect just that one face's thumbnail/attributes, not the shared frame or other faces.

    face_detector picks which face detection backend runs (unlike every other feature,
    exactly one runs per frame -- running two detectors and merging their boxes would just
    produce duplicate/overlapping faces, not a meaningfully combined result). "yolo"/"scrfd"/
    "retinaface" fall back to "ssd" (the always-required detector) if that model isn't loaded.

    tracker (#2) is optional and stays None for single-image callers (upload/snapshot have no
    "next frame" for an ID to persist into). When a FaceTracker is passed -- video/webcam LIVE
    mode only -- each face's dict also carries a stable "track_id" (see FaceTracker), and the
    number burned into the annotated frame is that track_id instead of this frame's
    detection-order position, so tracking is visible, not just data the caller ignores).

    liveness_tracker is only ever passed by video/webcam LIVE mode -- a single static image has
    no blink transitions to observe, so liveness is unavailable there by design (not just
    unchecked): static callers (upload/snapshot) never pass a tracker, and analyze_frame skips
    liveness entirely -- no "liveness" pairs, no LivenessResult -- whenever liveness_tracker is
    None, regardless of active_liveness. active_liveness additionally gates it off within LIVE
    mode itself (unchecked box = skipped); omitted callers default to every loaded backend.
    """
    if active_liveness is None:
        active_liveness = set(models.liveness_nets)
    if global_adjustments and any(global_adjustments.values()):
        frame = apply_image_adjustments(frame, global_adjustments)

    annotated_frame = frame.copy()
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

    track_ids = tracker.update(face_boxes) if tracker is not None else [None] * len(face_boxes)
    cropped_faces = []
    hands_detected = False
    hand_net = models.hand_nets.get("mediapipe")
    if hand_net is not None and "mediapipe" in active_hands:
        hands = _cached_face_predict("hand_landmarks", "mediapipe", frame, detect_hand_landmarks_mediapipe, hand_net, frame)
        if hands:
            hands_detected = True
            draw_hand_landmarks(annotated_frame, hands)

    need_blob227 = ("caffe" in active_age and "caffe" in models.age_nets) or \
                   ("caffe" in active_gender and "caffe" in models.gender_nets)

    eye_cascade = models.eye_color_nets.get("colorimetric")
    lbph_trained = train_lbph_recognizer() if "lbph" in active_recognition and models.recognition_nets.get("lbph") else None

    for idx, ((x1, y1, x2, y2), track_id) in enumerate(zip(face_boxes, track_ids, strict=False), 1):
        crop_frame, (cx1, cy1, cx2, cy2) = frame, (x1, y1, x2, y2)
        if eye_cascade is not None:
            probe = frame[max(0, y1 - 20):min(y2 + 20, frame.shape[0]), max(0, x1 - 20):min(x2 + 20, frame.shape[1])]
            angle = (
                _cached_face_predict("roll_angle", "haarcascade", probe, _estimate_roll_angle, probe, eye_cascade)
                if probe.size else None
            )
            if angle is not None and abs(angle) > 3:
                crop_frame, (cx1, cy1, cx2, cy2) = _rotate_region(frame, (x1, y1, x2, y2), angle)

        x1_crop, y1_crop, x2_crop, y2_crop = face_crop_bounds(
            (cx1, cy1, cx2, cy2), crop_frame.shape[:2],
        )

        face = crop_frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        if face_adjustments and any(face_adjustments.values()):
            face = apply_image_adjustments(face, face_adjustments)

        run_liveness = liveness_tracker is not None and "mediapipe" in active_liveness
        liveness_net = models.liveness_nets.get("mediapipe") if run_liveness else None
        face_landmarker = (
            liveness_net if liveness_net is not None
            else models.face_landmarks_nets.get("mediapipe")
        )
        needs_face_landmarks = (
            "mediapipe" in active_gaze
            or "mediapipe" in active_face_landmarks
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
        if need_blob227:
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

        def _age_task(blob227=blob227, crop_frame=crop_frame, cx1=cx1, cx2=cx2, cy1=cy1, cy2=cy2, face=face, fairface_landmarks=fairface_landmarks, mivolo_age_result=mivolo_age_result, mivolo_result=mivolo_result, x1=x1, x2=x2, y1=y1, y2=y2):
            pairs, estimates = [], {}
            for key in active_age:
                net = models.age_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "caffe":
                    probs = _cached_face_predict("age_probs", key, face, caffe_probabilities, net, blob227)
                    bucket = int(np.argmax(probs))
                    value = AGE_LIST[bucket]
                    estimates[key] = float(np.mean(AGE_LIST_RANGES[bucket]))
                elif key == "fairface":
                    probs = _cached_face_predict(
                        "age_probs", key, face, fairface_probabilities,
                        net, crop_frame, (cx1, cy1, cx2, cy2), "age_output", fairface_landmarks,
                    )
                    value = fairface_age_label(probs)
                    estimates[key] = float(np.mean(FAIRFACE_AGE_RANGES[int(np.argmax(probs))]))
                elif key == "dex":
                    dex_face = crop_face_dex(frame, (x1, y1, x2, y2))
                    if face_adjustments and any(face_adjustments.values()):
                        dex_face = apply_image_adjustments(dex_face, face_adjustments)
                    estimate = _cached_face_predict("age_estimate", key, dex_face, dex_age_estimate, net, dex_face)
                    value = format_dex_age(estimate)
                    if estimate is not None:
                        estimates[key] = estimate[0]
                elif key == "mivolo":
                    if mivolo_result is None and mivolo_age_result is None:
                        continue
                    age = mivolo_result[0] if mivolo_result is not None else mivolo_age_result
                    value = f"{age:.0f}"
                    estimates[key] = age
                pairs.append((key, value))
                _record_model_latency(metrics, "age", key, started)
            return pairs, select_age(estimates)

        def _gender_task(blob227=blob227, crop_frame=crop_frame, cx1=cx1, cx2=cx2, cy1=cy1, cy2=cy2, face=face, fairface_landmarks=fairface_landmarks, mivolo_result=mivolo_result):
            pairs, male_probabilities = [], {}
            for key in active_gender:
                net = models.gender_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "caffe":
                    probs = _cached_face_predict("gender_probs", key, face, caffe_probabilities, net, blob227)
                    value = GENDER_LIST[int(np.argmax(probs))]
                    male_probabilities[key] = float(probs[0] / (probs.sum() or 1.0))
                elif key == "deepface":
                    probs = _cached_face_predict("gender_probs", key, face, deepface_probabilities, net, face)
                    value = "Male" if np.argmax(probs) == 1 else "Female"
                    male_probabilities[key] = float(probs[1] / (probs.sum() or 1.0))
                elif key == "fairface":
                    probs = _cached_face_predict(
                        "gender_probs", key, face, fairface_probabilities,
                        net, crop_frame, (cx1, cy1, cx2, cy2), "gender_output", fairface_landmarks,
                    )
                    value = fairface_gender_label(probs)
                    male_probabilities[key] = float(probs[0])
                elif key == "mivolo":
                    if mivolo_result is None:
                        continue
                    value = mivolo_result[1]
                    male_probabilities[key] = 1.0 if value == "Male" else 0.0
                pairs.append((key, value))
                _record_model_latency(metrics, "gender", key, started)
            return pairs, fuse_gender(male_probabilities)

        def _emotion_task(face=face, x1=x1, x2=x2, y1=y1, y2=y2):
            pairs = []
            for key in active_emotion:
                net = models.emotion_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "dan":
                    value = _cached_face_predict("emotion", key, face, predict_emotion_dan, net, face)
                elif key == "mini_xception":
                    value = _cached_face_predict("emotion", key, face, predict_emotion_mini_xception, net, face)
                elif key == "ferplus":
                    ferplus_face = crop_region(frame, x1, y1, x2, y2)
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

        def _race_task(crop_frame=crop_frame, cx1=cx1, cx2=cx2, cy1=cy1, cy2=cy2, face=face, fairface_landmarks=fairface_landmarks):
            pairs, distributions = [], {}
            for key in active_race:
                net = models.race_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "fairface":
                    probs = _cached_face_predict(
                        "race_probs", key, face, fairface_probabilities,
                        net, crop_frame, (cx1, cy1, cx2, cy2), "race_output", fairface_landmarks,
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

        def _gaze_task(face=face, landmarker_result=landmarker_result):
            pairs = []
            for key in active_gaze:
                net = models.gaze_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = predict_gaze_mediapipe(net, face, landmarker_result)
                pairs.append((key, value))
                _record_model_latency(metrics, "gaze", key, started)
            return pairs

        def _head_pose_task(face=face, landmarker_result=landmarker_result):
            pairs = []
            for key in active_gaze:
                net = models.gaze_nets.get(key)
                if net is not None:
                    pairs.append((key, predict_head_pose_mediapipe(net, face, landmarker_result)))
            return pairs

        def _recognition_task(face=face):
            pairs = []
            embedding = None
            for key in active_recognition:
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
                    match = match_face_identity(embedding, gallery)
                    value = f"{match[0]} ({match[1] * 100:.0f}%)" if match else "UNKNOWN"
                pairs.append((key, value))
                _record_model_latency(metrics, "recognition", key, started)
            return pairs, embedding

        def _glasses_task(face=face):
            pairs = []
            for key in active_glasses:
                net = models.glasses_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict("glasses", key, face, predict_glasses_mobilenet, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "glasses", key, started)
            return pairs

        def _mask_task(face=face):
            pairs = []
            for key in active_mask:
                net = models.mask_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict("mask", key, face, predict_mask_mobilenetv2, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "mask", key, started)
            return pairs

        def _hair_color_task(crop_frame=crop_frame, cx1=cx1, cx2=cx2, cy1=cy1, cy2=cy2):
            pairs = []
            for key in active_hair_color:
                if key not in models.hair_color_nets:
                    continue
                started = time.perf_counter()
                value = predict_hair_color_colorimetric(crop_frame, (cx1, cy1, cx2, cy2))
                pairs.append((key, value))
                _record_model_latency(metrics, "hair_color", key, started)
            return pairs

        def _eye_color_task(face=face, points=points):
            pairs = []
            for key in active_eye_color:
                net = models.eye_color_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if points is not None:
                    value = predict_eye_color_colorimetric(net, face, points)
                else:
                    value = _cached_face_predict("eye_color", key, face, predict_eye_color_colorimetric, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "eye_color", key, started)
            return pairs

        def _liveness_task(blink_score=blink_score, run_liveness=run_liveness, texture_score=texture_score, track_id=track_id):
            if not run_liveness:
                return [], None
            started = time.perf_counter()
            if track_id is not None:
                result = liveness_tracker.update(track_id, blink_score, texture_score)
            else:
                result = assess_static_liveness(texture_score)
            _record_model_latency(metrics, "liveness", "mediapipe", started)
            return [("mediapipe", result.summary)], result

        futures = {
            "age": _INFERENCE_EXECUTOR.submit(_age_task),
            "gender": _INFERENCE_EXECUTOR.submit(_gender_task),
            "emotion": _INFERENCE_EXECUTOR.submit(_emotion_task),
            "race": _INFERENCE_EXECUTOR.submit(_race_task),
            "gaze": _INFERENCE_EXECUTOR.submit(_gaze_task),
            "head_pose": _INFERENCE_EXECUTOR.submit(_head_pose_task),
            "recognition": _INFERENCE_EXECUTOR.submit(_recognition_task),
            "glasses": _INFERENCE_EXECUTOR.submit(_glasses_task),
            "mask": _INFERENCE_EXECUTOR.submit(_mask_task),
            "hair_color": _INFERENCE_EXECUTOR.submit(_hair_color_task),
            "eye_color": _INFERENCE_EXECUTOR.submit(_eye_color_task),
            "liveness": _INFERENCE_EXECUTOR.submit(_liveness_task),
        }

        age_pairs, best_age = futures["age"].result()
        gender_pairs, fused_gender = futures["gender"].result()
        emotion_pairs, fused_emotion = futures["emotion"].result()
        race_pairs, fused_race = futures["race"].result()
        gaze_pairs = futures["gaze"].result()
        head_pose_pairs = futures["head_pose"].result()
        recognition_pairs, face_embedding = futures["recognition"].result()
        glasses_pairs = futures["glasses"].result()
        mask_pairs = futures["mask"].result()
        hair_color_pairs = futures["hair_color"].result()
        eye_color_pairs = futures["eye_color"].result()
        liveness_pairs, liveness_result = futures["liveness"].result()

        if metrics is not None and emotion_pairs:
            metrics.setdefault("emotion_samples", []).extend(
                {"model": key, "emotion": value} for key, value in emotion_pairs
            )

        box_thickness = int(round(frame.shape[0] / 150)) or 1
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness, 8)
        display_id = track_id if track_id is not None else idx
        draw_outlined_text(annotated_frame, str(display_id), (x1, max(20, y1 - 10)), (0, 255, 255))

        landmarks_net = models.face_landmarks_nets.get("mediapipe")
        if landmarks_net is not None and "mediapipe" in active_face_landmarks:
            landmark_points = predict_face_landmarks_mediapipe(landmarks_net, face, landmarker_result)
            if landmark_points is not None:
                draw_face_landmarks(annotated_frame, landmark_points, (x1, y1, x2, y2))

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

        cropped_faces.append({
            "idx": idx,
            "track_id": track_id,
            "box": (x1, y1, x2, y2),
            "image": cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
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
        })

    return annotated_frame, cropped_faces, bool(face_boxes), hands_detected


def analyze_frame_with_config(
    models: Models, frame: np.ndarray, config: AnalysisConfig
) -> tuple[np.ndarray, list[dict], bool, bool]:
    """Execute the facial analysis pipeline using a consolidated AnalysisConfig object."""
    return analyze_frame(
        models=models,
        frame=frame,
        conf_threshold=config.conf_threshold,
        active_age=config.active_age,
        active_gender=config.active_gender,
        active_emotion=config.active_emotion,
        active_race=config.active_race,
        active_recognition=config.active_recognition,
        gallery=config.gallery,
        active_glasses=config.active_glasses,
        active_mask=config.active_mask,
        active_hair_color=config.active_hair_color,
        active_eye_color=config.active_eye_color,
        active_face_landmarks=config.active_face_landmarks,
        active_hands=config.active_hands,
        active_gaze=config.active_gaze,
        global_adjustments=config.global_adjustments,
        face_adjustments=config.face_adjustments,
        face_detector=config.face_detector,
        metrics=config.metrics,
        tracker=config.tracker,
        liveness_tracker=config.liveness_tracker,
        active_liveness=config.active_liveness,
    )


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
    "analyze_frame_with_config",
    "aggregate_demographics",
    "_cached_face_predict",
    "predict_face_landmarks_mediapipe",
    "predict_gaze_mediapipe",
    "predict_head_pose_mediapipe",
    "_record_model_latency",
    "_PREDICTION_CACHE",
    "_INFERENCE_EXECUTOR",
    "AGGREGATE_FEATURES",
]
