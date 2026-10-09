"""Stage helpers that analyze_frame orchestrates: prepare, detect, per-face inference, annotate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..attributes import (
    _estimate_roll_angle,
    fairface_landmarks_from_detector,
    fairface_landmarks_from_mediapipe,
    level_face_points,
    level_face_region,
    mivolo_age_estimate,
    mivolo_estimate,
    predict_texture_artifact_score,
)
from ..core.constants import (
    BEST_MODEL_KEY,
    MODEL_MEAN_VALUES,
)
from ..core.image_utils import (
    apply_image_adjustments,
    face_crop_bounds,
)
from ..core.types import (
    Models,
)
from ..detectors import (
    detect_faces,
    detect_faces_retinaface_landmarks,
    detect_faces_scrfd_landmarks,
    detect_faces_yolo,
    face_detector_net,
    resolve_face_detector,
)
from ..fusion import (
    _format_results,
    _gather_face_results,
    with_headline,
)
from ..gallery import (
    train_lbph_recognizer,
)
from ..liveness import (
    blink_score_from_landmarker,
)
from .cache import (
    _INFERENCE_EXECUTOR,
    _cached_face_predict,
)
from .config import (
    AnalysisConfig,
)
from .drawing import (
    draw_face_landmarks,
    draw_hand_landmarks,
    draw_outlined_text,
)
from .face_tasks import (
    _age_task,
    _emotion_task,
    _eye_color_task,
    _FaceInputs,
    _gaze_task,
    _gender_task,
    _glasses_task,
    _head_pose_task,
    _liveness_task,
    _mask_task,
    _race_task,
    _recognition_task,
)
from .landmarks import (
    _detect_face_landmarker,
    detect_hand_landmarks_mediapipe,
    predict_face_landmarks_mediapipe,
)


@dataclass
class _FrameInputs:
    """Per-frame setup shared by every detected face."""

    active_liveness: set
    need_blob227: bool
    eye_cascade: Any
    lbph_trained: Any


def _prepare_frame(frame: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    """Apply global_adjustments to the whole frame, before face detection even runs."""
    global_adjustments = config.global_adjustments
    if global_adjustments and any(global_adjustments.values()):
        frame = apply_image_adjustments(frame, global_adjustments)
    return frame


def _detect(models: Models, frame: np.ndarray, config: AnalysisConfig) -> tuple[list, list]:
    """Run the one configured face detector, falling back to SSD (then any loaded detector) if absent.

    Unlike every other feature, exactly one detector runs per frame -- running two and merging
    their boxes would just produce duplicate/overlapping faces, not a meaningfully combined result.
    Returns (boxes, landmarks): each box's 5x2 detector landmarks, or None from SSD and YOLO.
    """
    conf_threshold = config.conf_threshold
    # With no detector loaded, the SSD path's bare-net call returns no faces.
    face_detector = resolve_face_detector(models, config.face_detector) or "ssd"
    detect_fn = {
        "yolo": detect_faces_yolo,
        "scrfd": detect_faces_scrfd_landmarks,
        "retinaface": detect_faces_retinaface_landmarks,
        # SSD keeps going through the factory's legacy bare-net path, as it always has.
        "ssd": detect_faces,
    }[face_detector]
    detections = _cached_face_predict(
        "face_detection", f"{face_detector}:{conf_threshold}", frame, detect_fn,
        face_detector_net(models, face_detector), frame, conf_threshold,
    )
    if face_detector in ("scrfd", "retinaface"):
        return detections
    return detections, [None] * len(detections)


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
    detector_landmarks: np.ndarray | None = None,
) -> _FaceInputs | None:
    """Roll-align, crop and face-adjust one face, then compute its shared landmarks and blobs.

    face_adjustments apply to this face's own crop only, after detection but before
    classification -- they affect just this face's thumbnail/attributes, not the shared
    frame or other faces. FairFace aligns on MediaPipe's landmarks, else on the detector's
    (frame coordinates, as _detect returns them), else on the box. Returns None when the
    crop is empty.
    """
    x1, y1, x2, y2 = box
    active_age, active_gender = config.active_age, config.active_gender
    face_adjustments = config.face_adjustments
    crop_frame, (cx1, cy1, cx2, cy2) = frame, (x1, y1, x2, y2)
    angle = None
    if shared.eye_cascade is not None:
        probe = frame[max(0, y1 - 20):min(y2 + 20, frame.shape[0]), max(0, x1 - 20):min(x2 + 20, frame.shape[1])]
        angle = (
            _cached_face_predict("roll_angle", "haarcascade", probe, _estimate_roll_angle, probe, shared.eye_cascade)
            if probe.size else None
        )
        crop_frame, (cx1, cy1, cx2, cy2) = level_face_region(frame, (x1, y1, x2, y2), angle)

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
    if fairface_landmarks is None and detector_landmarks is not None:
        # The FairFace forward reads crop_frame, which is the leveled region when the face was rotated.
        fairface_landmarks = fairface_landmarks_from_detector(
            level_face_points(frame.shape, (x1, y1, x2, y2), angle, detector_landmarks), x2 - x1,
        )
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


def _analyze_face(
    models: Models, frame: np.ndarray, box: tuple, track_id: Any, config: AnalysisConfig, shared: _FrameInputs,
    detector_landmarks: np.ndarray | None = None,
) -> dict | None:
    """Run every attribute task for one face in parallel; None when the crop is empty.

    Returns {"inputs": _FaceInputs, <feature>: that task's result} for _annotate/_face_record.
    """
    inputs = _prepare_face(models, frame, box, track_id, config, shared, detector_landmarks)
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
    gender_pairs, best_gender = analysis["gender"]
    emotion_pairs, fused_emotion = analysis["emotion"]
    race_pairs, best_race = analysis["race"]
    gaze_pairs = analysis["gaze"]
    head_pose_pairs = analysis["head_pose"]
    recognition_pairs, face_embedding = analysis["recognition"]
    glasses_pairs = analysis["glasses"]
    mask_pairs = analysis["mask"]
    eye_color_pairs = analysis["eye_color"]
    liveness_pairs, liveness_result = analysis["liveness"]

    eye_contact = [f"{key}=yes" if value.startswith("center/") else f"{key}=no" for key, value in gaze_pairs]

    age_pairs = with_headline(
        age_pairs,
        f"{best_age[0]} ({best_age[1]})" if best_age else None,
        BEST_MODEL_KEY,
    )
    gender_pairs = with_headline(
        gender_pairs,
        f"{best_gender[0]} ({best_gender[1]})" if best_gender else None,
        BEST_MODEL_KEY,
    )
    race_pairs = with_headline(
        race_pairs,
        f"{best_race[0]} ({best_race[1]})" if best_race else None,
        BEST_MODEL_KEY,
    )
    emotion_pairs = with_headline(emotion_pairs, fused_emotion)

    raw_columns = _gather_face_results({
        "age": age_pairs, "gender": gender_pairs, "race": race_pairs, "emotion": emotion_pairs,
        "gaze": gaze_pairs, "identity": recognition_pairs,
        "eye_contact": [("derived", value) for value in eye_contact], "head_pose": head_pose_pairs,
        "glasses": glasses_pairs, "mask": mask_pairs,
        "eye_color": eye_color_pairs,
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
            "mask": mask_pairs, "eye color": eye_color_pairs,
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
            "gender": best_gender[0] if best_gender else None,
            "race": best_race[0] if best_race else None,
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
    "_analyze_face",
    "_annotate",
    "_detect",
    "_face_record",
    "_frame_inputs",
    "_prepare_frame",
    "_run_whole_frame_features",
]
