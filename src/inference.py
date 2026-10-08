"""Facade interface for facial analysis pipeline, models, attributes, detectors, and gallery."""
from __future__ import annotations

import contextlib
import io
import os
import random
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

# Core types, constants, and image processing utilities
try:
    from src.core import (
        AGE_LIST,
        AGE_LIST_RANGES,
        AUDIO_AROUSAL_LOUD_RMS,
        AUDIO_AROUSAL_QUIET_RMS,
        AUDIO_AROUSAL_WINDOW_SECONDS,
        BASE_DIR,
        BEST_MODEL_KEY,
        BFM_DIR,
        BFM_LM3D_PATH,
        BFM_MODEL_PATH,
        COLORIZATION_MODEL,
        COLORIZATION_PROTO,
        COLORIZATION_PTS,
        DEEP3D_RECON_MODEL,
        DEEPFACE_GENDER_MODEL,
        DEEPFACE_RACE_MODEL,
        DEEPFACE_RECOGNITION_MODEL,
        DENOISE_METHODS,
        DEX_MEAN_VALUES,
        DEX_MODEL,
        DEX_PROTO,
        EIGEN_DIR,
        EMOTION_HIGH_AROUSAL_LABELS,
        EMOTION_LABELS_DAN,
        EMOTION_LABELS_FERPLUS,
        EMOTION_LABELS_HSEMOTION,
        EMOTION_LABELS_MINI_XCEPTION,
        EMOTION_LOW_AROUSAL_LABELS,
        EMOTION_MODEL,
        EYE_CASCADE_FILE,
        EYE_COLOR_LABELS,
        FACE_DETECTOR_OPTIONS,
        FACE_LANDMARKER_MODEL,
        FACE_MODEL,
        FACE_PROTO,
        FACE_REAGING_MODEL,
        FACES_DB_FILE,
        FACES_DIR,
        FAIRFACE_AGE_RANGES,
        FAIRFACE_MODEL,
        GENDER_LIST,
        GENDER_MODEL,
        GENDER_PROTO,
        GLASSES_MODEL,
        HAND_CONNECTIONS,
        HAND_LANDMARKER_MODEL,
        HEADLINE_MODEL_KEYS,
        HSEMOTION_MODEL,
        IMAGE_ADJUSTMENT_RANGES,
        IMAGE_OP_OPTIONS,
        INTENSITY_METHODS,
        KNOWN_PEOPLE_DIR,
        MASK_MODEL,
        MINI_XCEPTION_MODEL,
        MIVOLO_MODEL,
        MODEL_DIR,
        MODEL_MEAN_VALUES,
        RACE_CANONICAL_LABELS,
        RACE_LABELS_DEEPFACE,
        RACE_LABELS_FAIRFACE,
        RETINAFACE_MODEL,
        SCRFD_FACE_MODEL,
        SHARPEN_METHODS,
        YOLO_FACE_MODEL,
        BoundingBox,
        Detection,
        FaceResult,
        Models,
        _is_skin_hsv,
        apply_image_adjustments,
        crop_region,
        face_crop_bounds,
        is_grayscale_frame,
    )
except ImportError:
    from core import (
        AGE_LIST,
        AGE_LIST_RANGES,
        AUDIO_AROUSAL_LOUD_RMS,
        AUDIO_AROUSAL_QUIET_RMS,
        AUDIO_AROUSAL_WINDOW_SECONDS,
        BASE_DIR,
        BEST_MODEL_KEY,
        BFM_DIR,
        BFM_LM3D_PATH,
        BFM_MODEL_PATH,
        COLORIZATION_MODEL,
        COLORIZATION_PROTO,
        COLORIZATION_PTS,
        DEEP3D_RECON_MODEL,
        DEEPFACE_GENDER_MODEL,
        DEEPFACE_RACE_MODEL,
        DEEPFACE_RECOGNITION_MODEL,
        DENOISE_METHODS,
        DEX_MEAN_VALUES,
        DEX_MODEL,
        DEX_PROTO,
        EIGEN_DIR,
        EMOTION_HIGH_AROUSAL_LABELS,
        EMOTION_LABELS_DAN,
        EMOTION_LABELS_FERPLUS,
        EMOTION_LABELS_HSEMOTION,
        EMOTION_LABELS_MINI_XCEPTION,
        EMOTION_LOW_AROUSAL_LABELS,
        EMOTION_MODEL,
        EYE_CASCADE_FILE,
        EYE_COLOR_LABELS,
        FACE_DETECTOR_OPTIONS,
        FACE_LANDMARKER_MODEL,
        FACE_MODEL,
        FACE_PROTO,
        FACE_REAGING_MODEL,
        FACES_DB_FILE,
        FACES_DIR,
        FAIRFACE_AGE_RANGES,
        FAIRFACE_MODEL,
        GENDER_LIST,
        GENDER_MODEL,
        GENDER_PROTO,
        GLASSES_MODEL,
        HAND_CONNECTIONS,
        HAND_LANDMARKER_MODEL,
        HEADLINE_MODEL_KEYS,
        HSEMOTION_MODEL,
        IMAGE_ADJUSTMENT_RANGES,
        IMAGE_OP_OPTIONS,
        INTENSITY_METHODS,
        KNOWN_PEOPLE_DIR,
        MASK_MODEL,
        MINI_XCEPTION_MODEL,
        MIVOLO_MODEL,
        MODEL_DIR,
        MODEL_MEAN_VALUES,
        RACE_CANONICAL_LABELS,
        RACE_LABELS_DEEPFACE,
        RACE_LABELS_FAIRFACE,
        RETINAFACE_MODEL,
        SCRFD_FACE_MODEL,
        SHARPEN_METHODS,
        YOLO_FACE_MODEL,
        BoundingBox,
        Detection,
        FaceResult,
        Models,
        _is_skin_hsv,
        apply_image_adjustments,
        crop_region,
        face_crop_bounds,
        is_grayscale_frame,
    )

# Detectors
try:
    from src.detectors import (
        detect_faces,
        detect_faces_retinaface,
        detect_faces_scrfd,
        detect_faces_ssd,
        detect_faces_yolo,
    )
except ImportError:
    from detectors import (
        detect_faces,
        detect_faces_retinaface,
        detect_faces_scrfd,
        detect_faces_ssd,
        detect_faces_yolo,
    )

# Attributes
try:
    from src.attributes import (
        MiVOLOInference,
        VoiceFaceFusion,
        _emotion_arousal_category,
        _estimate_roll_angle,
        _fairface_forward,
        _format_race_label,
        _lock_for,
        _margin_align,
        _rotate_region,
        align_face_with_landmarks,
        apply_bilateral_filter,
        apply_color_correct,
        apply_denoise,
        apply_enhance,
        apply_geometric_transform,
        apply_image_op,
        apply_intensity_transform,
        apply_sharpen,
        apply_wavelet_denoise,
        audio_frame_to_mono_float,
        caffe_probabilities,
        classify_voice_arousal,
        colorize_frame,
        crop_face_dex,
        deepface_probabilities,
        dex_age_estimate,
        fairface_age_label,
        fairface_gender_label,
        fairface_landmarks_from_mediapipe,
        fairface_probabilities,
        fairface_race_label,
        format_dex_age,
        fuse_voice_and_emotion,
        maybe_colorize,
        mivolo_age_estimate,
        mivolo_estimate,
        predict_age_caffe,
        predict_age_dex,
        predict_age_fairface,
        predict_age_mivolo,
        predict_emotion_dan,
        predict_emotion_ferplus,
        predict_emotion_hsemotion,
        predict_emotion_mini_xception,
        predict_eye_color_colorimetric,
        predict_gender_caffe,
        predict_gender_deepface,
        predict_gender_fairface,
        predict_gender_mivolo,
        predict_glasses_mobilenet,
        predict_hair_color_colorimetric,
        predict_mask_mobilenetv2,
        predict_race_deepface,
        predict_race_fairface,
        predict_texture_artifact_score,
        run_3d_reconstruction,
        run_age_progression,
    )
except ImportError:
    from attributes import (
        MiVOLOInference,
        VoiceFaceFusion,
        _emotion_arousal_category,
        _estimate_roll_angle,
        _fairface_forward,
        _format_race_label,
        _lock_for,
        _margin_align,
        _rotate_region,
        align_face_with_landmarks,
        apply_bilateral_filter,
        apply_color_correct,
        apply_denoise,
        apply_enhance,
        apply_geometric_transform,
        apply_image_op,
        apply_intensity_transform,
        apply_sharpen,
        apply_wavelet_denoise,
        audio_frame_to_mono_float,
        caffe_probabilities,
        classify_voice_arousal,
        colorize_frame,
        crop_face_dex,
        deepface_probabilities,
        dex_age_estimate,
        fairface_age_label,
        fairface_gender_label,
        fairface_landmarks_from_mediapipe,
        fairface_probabilities,
        fairface_race_label,
        format_dex_age,
        fuse_voice_and_emotion,
        maybe_colorize,
        mivolo_age_estimate,
        mivolo_estimate,
        predict_age_caffe,
        predict_age_dex,
        predict_age_fairface,
        predict_age_mivolo,
        predict_emotion_dan,
        predict_emotion_ferplus,
        predict_emotion_hsemotion,
        predict_emotion_mini_xception,
        predict_eye_color_colorimetric,
        predict_gender_caffe,
        predict_gender_deepface,
        predict_gender_fairface,
        predict_gender_mivolo,
        predict_glasses_mobilenet,
        predict_hair_color_colorimetric,
        predict_mask_mobilenetv2,
        predict_race_deepface,
        predict_race_fairface,
        predict_texture_artifact_score,
        run_3d_reconstruction,
        run_age_progression,
    )

# Fusion
try:
    from src.fusion import (
        _format_results,
        _gather_face_results,
        _sanitize_column_name,
        _weighted_median,
        canonical_race_probabilities,
        fuse_emotion,
        fuse_gender,
        fuse_race,
        select_age,
        with_headline,
    )
except ImportError:
    from fusion import (
        _format_results,
        _gather_face_results,
        _sanitize_column_name,
        _weighted_median,
        canonical_race_probabilities,
        fuse_emotion,
        fuse_gender,
        fuse_race,
        select_age,
        with_headline,
    )

# Gallery
try:
    from src.gallery import (
        build_gallery_from_directory,
        compute_face_embedding,
        decode_image_bytes,
        enroll_lbph_face,
        load_gallery,
        match_face_eigenfaces,
        match_face_identity,
        match_faces_eigenfaces_batch,
        predict_identity_lbph,
        save_face,
        save_gallery,
        train_lbph_recognizer,
        validate_lbph_name,
    )
except ImportError:
    from gallery import (
        build_gallery_from_directory,
        compute_face_embedding,
        decode_image_bytes,
        enroll_lbph_face,
        load_gallery,
        match_face_eigenfaces,
        match_face_identity,
        match_faces_eigenfaces_batch,
        predict_identity_lbph,
        save_face,
        save_gallery,
        train_lbph_recognizer,
        validate_lbph_name,
    )

# Pipeline
try:
    from src.pipeline import (
        _INFERENCE_EXECUTOR,
        _PREDICTION_CACHE,
        AGGREGATE_FEATURES,
        AnalysisConfig,
        FaceTracker,
        _box_iou,
        _cached_face_predict,
        _record_model_latency,
        _silence_native_logs,
        aggregate_demographics,
        analyze_frame_with_config,
        draw_face_landmarks,
        draw_hand_landmarks,
        draw_outlined_text,
        draw_recognition_scan,
        predict_face_landmarks_mediapipe,
        predict_gaze_mediapipe,
        predict_head_pose_mediapipe,
    )
    from src.pipeline import (
        analyze_frame as _pipeline_analyze_frame,
    )
    from src.pipeline import (
        load_models as _pipeline_load_models,
    )
except ImportError:
    from pipeline import (
        _INFERENCE_EXECUTOR,
        _PREDICTION_CACHE,
        AGGREGATE_FEATURES,
        AnalysisConfig,
        FaceTracker,
        _box_iou,
        _cached_face_predict,
        _record_model_latency,
        _silence_native_logs,
        aggregate_demographics,
        analyze_frame_with_config,
        draw_face_landmarks,
        draw_hand_landmarks,
        draw_outlined_text,
        draw_recognition_scan,
        predict_face_landmarks_mediapipe,
        predict_gaze_mediapipe,
        predict_head_pose_mediapipe,
    )
    from pipeline import (
        analyze_frame as _pipeline_analyze_frame,
    )
    from pipeline import (
        load_models as _pipeline_load_models,
    )

try:
    from src.liveness import (
        LivenessTracker,
        assess_static_liveness,
        blink_score_from_landmarker,
        texture_artifact_score,
    )
except ImportError:
    from liveness import (
        LivenessTracker,
        assess_static_liveness,
        blink_score_from_landmarker,
        texture_artifact_score,
    )

try:
    from src.nets.deep3d_recon import (
        landmarks_5pt_from_mediapipe,
        mesh_to_obj_str,
        reconstruct_face_3d,
    )
except ImportError:
    try:
        from nets.deep3d_recon import (
            landmarks_5pt_from_mediapipe,
            mesh_to_obj_str,
            reconstruct_face_3d,
        )
    except ImportError:
        def mesh_to_obj_str(*args: Any, **kwargs: Any) -> str:
            return ""

        landmarks_5pt_from_mediapipe = None
        reconstruct_face_3d = None

try:
    import mediapipe as mp
except ImportError:
    mp = None


__all__ = [
    "AGE_LIST",
    "AGE_LIST_RANGES",
    "aggregate_demographics",
    "AGGREGATE_FEATURES",
    "align_face_with_landmarks",
    "AnalysisConfig",
    "analyze_frame",
    "analyze_frame_with_config",
    "apply_bilateral_filter",
    "apply_color_correct",
    "apply_denoise",
    "apply_enhance",
    "apply_geometric_transform",
    "apply_image_adjustments",
    "apply_image_op",
    "apply_intensity_transform",
    "apply_sharpen",
    "apply_wavelet_denoise",
    "assess_static_liveness",
    "AUDIO_AROUSAL_LOUD_RMS",
    "AUDIO_AROUSAL_QUIET_RMS",
    "AUDIO_AROUSAL_WINDOW_SECONDS",
    "audio_frame_to_mono_float",
    "BASE_DIR",
    "BEST_MODEL_KEY",
    "BFM_DIR",
    "BFM_LM3D_PATH",
    "BFM_MODEL_PATH",
    "blink_score_from_landmarker",
    "BoundingBox",
    "_box_iou",
    "build_gallery_from_directory",
    "_cached_face_predict",
    "caffe_probabilities",
    "canonical_race_probabilities",
    "classify_voice_arousal",
    "COLORIZATION_MODEL",
    "COLORIZATION_PROTO",
    "COLORIZATION_PTS",
    "colorize_frame",
    "compute_face_embedding",
    "contextlib",
    "crop_face_dex",
    "crop_region",
    "decode_image_bytes",
    "DEEP3D_RECON_MODEL",
    "DEEPFACE_GENDER_MODEL",
    "deepface_probabilities",
    "DEEPFACE_RACE_MODEL",
    "DEEPFACE_RECOGNITION_MODEL",
    "DENOISE_METHODS",
    "detect_faces",
    "detect_faces_retinaface",
    "detect_faces_scrfd",
    "detect_faces_ssd",
    "detect_faces_yolo",
    "detect_hand_landmarks_mediapipe",
    "Detection",
    "dex_age_estimate",
    "DEX_MEAN_VALUES",
    "DEX_MODEL",
    "DEX_PROTO",
    "draw_face_landmarks",
    "draw_hand_landmarks",
    "draw_outlined_text",
    "draw_recognition_scan",
    "EIGEN_DIR",
    "_emotion_arousal_category",
    "EMOTION_HIGH_AROUSAL_LABELS",
    "EMOTION_LABELS_DAN",
    "EMOTION_LABELS_FERPLUS",
    "EMOTION_LABELS_HSEMOTION",
    "EMOTION_LABELS_MINI_XCEPTION",
    "EMOTION_LOW_AROUSAL_LABELS",
    "EMOTION_MODEL",
    "enroll_lbph_face",
    "_estimate_roll_angle",
    "EYE_CASCADE_FILE",
    "EYE_COLOR_LABELS",
    "face_crop_bounds",
    "FACE_DETECTOR_OPTIONS",
    "FACE_LANDMARKER_MODEL",
    "FACE_MODEL",
    "FACE_PROTO",
    "FACE_REAGING_MODEL",
    "FaceResult",
    "FACES_DB_FILE",
    "FACES_DIR",
    "FaceTracker",
    "fairface_age_label",
    "FAIRFACE_AGE_RANGES",
    "_fairface_forward",
    "fairface_gender_label",
    "fairface_landmarks_from_mediapipe",
    "FAIRFACE_MODEL",
    "fairface_probabilities",
    "fairface_race_label",
    "format_dex_age",
    "_format_race_label",
    "_format_results",
    "fuse_emotion",
    "fuse_gender",
    "fuse_race",
    "fuse_voice_and_emotion",
    "_gather_face_results",
    "GENDER_LIST",
    "GENDER_MODEL",
    "GENDER_PROTO",
    "GLASSES_MODEL",
    "HAND_CONNECTIONS",
    "HAND_LANDMARKER_MODEL",
    "HEADLINE_MODEL_KEYS",
    "HSEMOTION_MODEL",
    "IMAGE_ADJUSTMENT_RANGES",
    "IMAGE_OP_OPTIONS",
    "_INFERENCE_EXECUTOR",
    "INTENSITY_METHODS",
    "io",
    "is_grayscale_frame",
    "_is_skin_hsv",
    "KNOWN_PEOPLE_DIR",
    "LivenessTracker",
    "load_gallery",
    "load_models",
    "_margin_align",
    "MASK_MODEL",
    "match_face_eigenfaces",
    "match_face_identity",
    "match_faces_eigenfaces_batch",
    "maybe_colorize",
    "MINI_XCEPTION_MODEL",
    "mivolo_age_estimate",
    "mivolo_estimate",
    "MIVOLO_MODEL",
    "MiVOLOInference",
    "MODEL_DIR",
    "MODEL_MEAN_VALUES",
    "os",
    "predict_age_caffe",
    "predict_age_dex",
    "predict_age_fairface",
    "predict_age_mivolo",
    "predict_emotion_dan",
    "predict_emotion_ferplus",
    "predict_emotion_hsemotion",
    "predict_emotion_mini_xception",
    "predict_eye_color_colorimetric",
    "predict_face_landmarks_mediapipe",
    "predict_gaze_mediapipe",
    "predict_gender_caffe",
    "predict_gender_deepface",
    "predict_gender_fairface",
    "predict_gender_mivolo",
    "predict_glasses_mobilenet",
    "predict_hair_color_colorimetric",
    "predict_head_pose_mediapipe",
    "predict_identity_lbph",
    "predict_mask_mobilenetv2",
    "predict_race_deepface",
    "predict_race_fairface",
    "predict_texture_artifact_score",
    "_PREDICTION_CACHE",
    "RACE_CANONICAL_LABELS",
    "RACE_LABELS_DEEPFACE",
    "RACE_LABELS_FAIRFACE",
    "random",
    "_record_model_latency",
    "RETINAFACE_MODEL",
    "_rotate_region",
    "run_3d_reconstruction",
    "run_age_progression",
    "_sanitize_column_name",
    "save_face",
    "save_gallery",
    "SCRFD_FACE_MODEL",
    "select_age",
    "SHARPEN_METHODS",
    "sys",
    "texture_artifact_score",
    "threading",
    "time",
    "train_lbph_recognizer",
    "validate_lbph_name",
    "VoiceFaceFusion",
    "_weighted_median",
    "with_headline",
    "YOLO_FACE_MODEL",
]


def load_models() -> Models:
    """Load every model whose file(s)/dependencies are present.

    Includes dictionary mapping for liveness_nets: dict and liveness_nets["mediapipe"].
    """
    return _pipeline_load_models()


def _detect_face_landmarker(landmarker: Any, face_bgr: np.ndarray) -> Any:
    """Run MediaPipe FaceLandmarker on one face crop with thread-safe locking."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    with _lock_for(landmarker):
        with _silence_native_logs():
            return landmarker.detect(mp_image)


def detect_hand_landmarks_mediapipe(landmarker: Any, frame_bgr: np.ndarray) -> list[list[tuple[int, int]]]:
    """MediaPipe HandLandmarker, whole-frame (hands aren't tied to a detected face box)."""
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    with _lock_for(landmarker):
        with _silence_native_logs():
            result = landmarker.detect(mp_image)
    return [
        [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand]
        for hand in result.hand_landmarks
    ]


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
    return _pipeline_analyze_frame(
        models=models,
        frame=frame,
        conf_threshold=conf_threshold,
        active_age=active_age,
        active_gender=active_gender,
        active_emotion=active_emotion,
        active_race=active_race,
        active_recognition=active_recognition,
        gallery=gallery,
        active_glasses=active_glasses,
        active_mask=active_mask,
        active_hair_color=active_hair_color,
        active_eye_color=active_eye_color,
        active_face_landmarks=active_face_landmarks,
        active_hands=active_hands,
        active_gaze=active_gaze,
        global_adjustments=global_adjustments,
        face_adjustments=face_adjustments,
        face_detector=face_detector,
        metrics=metrics,
        tracker=tracker,
        liveness_tracker=liveness_tracker,
        active_liveness=active_liveness,
    )
