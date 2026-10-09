"""Facade interface for facial analysis pipeline, models, attributes, detectors, and gallery."""
from __future__ import annotations

from typing import Any

# Attributes
from .attributes import (
    VoiceFaceFusion,
    align_face_with_landmarks,
    apply_image_op,
    audio_frame_to_mono_float,
    classify_voice_arousal,
    crop_face_dex,
    dex_age_estimate,
    fairface_landmarks_from_mediapipe,
    format_dex_age,
    fuse_voice_and_emotion,
    maybe_colorize,
    mivolo_estimate,
    predict_emotion_dan,
    predict_emotion_ferplus,
    predict_emotion_hsemotion,
    predict_emotion_mini_xception,
    predict_eye_color_colorimetric,
    predict_hair_color_colorimetric,
    run_3d_reconstruction,
    run_age_progression,
)

# Core types, constants, and image processing utilities
from .core import (
    AGE_LIST,
    AGE_LIST_RANGES,
    AUDIO_AROUSAL_LOUD_RMS,
    AUDIO_AROUSAL_QUIET_RMS,
    BEST_MODEL_KEY,
    DENOISE_METHODS,
    DEX_MEAN_VALUES,
    FACE_DETECTOR_OPTIONS,
    FAIRFACE_AGE_RANGES,
    GENDER_LIST,
    HEADLINE_MODEL_KEYS,
    IMAGE_ADJUSTMENT_RANGES,
    IMAGE_FILE_EXTENSIONS,
    IMAGE_OP_OPTIONS,
    INTENSITY_METHODS,
    KNOWN_PEOPLE_DIR,
    MODEL_MEAN_VALUES,
    RACE_CANONICAL_LABELS,
    RACE_LABELS_DEEPFACE,
    RACE_LABELS_FAIRFACE,
    SHARPEN_METHODS,
    Models,
    apply_image_adjustments,
    face_crop_bounds,
)

# Detectors
from .detectors import (
    available_face_detectors,
    detect_faces,
    detect_faces_retinaface,
    detect_faces_scrfd,
    detect_faces_ssd,
    detect_faces_yolo,
    resolve_face_detector,
)

# Fusion
from .fusion import (
    canonical_race_probabilities,
    fuse_emotion,
    select_age,
    select_gender,
    select_race,
    with_headline,
)

# Gallery
from .gallery import (
    build_gallery_from_directory,
    decode_image_bytes,
    enroll_lbph_face,
    load_gallery,
    match_face_eigenfaces,
    match_face_identity,
    match_faces_eigenfaces_batch,
    save_face,
    save_gallery,
    validate_lbph_name,
)
from .liveness import LivenessTracker

# Pipeline
from .pipeline import (
    AGGREGATE_FEATURES,
    AnalysisConfig,
    FaceTracker,
    aggregate_demographics,
    analyze_frame,
    draw_recognition_scan,
    load_models,
    predict_face_landmarks_mediapipe,
)

try:
    from .nets.deep3d_recon import mesh_to_obj_str
except ImportError:
    def mesh_to_obj_str(*args: Any, **kwargs: Any) -> str:
        """Return an empty OBJ string when the deep3d module is unavailable."""
        return ""

__all__ = [
    "AGE_LIST",
    "AGE_LIST_RANGES",
    "aggregate_demographics",
    "AGGREGATE_FEATURES",
    "align_face_with_landmarks",
    "AnalysisConfig",
    "analyze_frame",
    "apply_image_adjustments",
    "apply_image_op",
    "AUDIO_AROUSAL_LOUD_RMS",
    "AUDIO_AROUSAL_QUIET_RMS",
    "audio_frame_to_mono_float",
    "BEST_MODEL_KEY",
    "build_gallery_from_directory",
    "canonical_race_probabilities",
    "classify_voice_arousal",
    "crop_face_dex",
    "decode_image_bytes",
    "DENOISE_METHODS",
    "available_face_detectors",
    "detect_faces",
    "detect_faces_retinaface",
    "detect_faces_scrfd",
    "detect_faces_ssd",
    "detect_faces_yolo",
    "resolve_face_detector",
    "dex_age_estimate",
    "DEX_MEAN_VALUES",
    "draw_recognition_scan",
    "enroll_lbph_face",
    "face_crop_bounds",
    "FACE_DETECTOR_OPTIONS",
    "FaceTracker",
    "FAIRFACE_AGE_RANGES",
    "fairface_landmarks_from_mediapipe",
    "format_dex_age",
    "fuse_emotion",
    "fuse_voice_and_emotion",
    "GENDER_LIST",
    "HEADLINE_MODEL_KEYS",
    "IMAGE_ADJUSTMENT_RANGES",
    "IMAGE_FILE_EXTENSIONS",
    "IMAGE_OP_OPTIONS",
    "INTENSITY_METHODS",
    "KNOWN_PEOPLE_DIR",
    "LivenessTracker",
    "load_gallery",
    "load_models",
    "match_face_eigenfaces",
    "match_face_identity",
    "match_faces_eigenfaces_batch",
    "maybe_colorize",
    "mesh_to_obj_str",
    "mivolo_estimate",
    "Models",
    "MODEL_MEAN_VALUES",
    "predict_emotion_dan",
    "predict_emotion_ferplus",
    "predict_emotion_hsemotion",
    "predict_emotion_mini_xception",
    "predict_eye_color_colorimetric",
    "predict_face_landmarks_mediapipe",
    "predict_hair_color_colorimetric",
    "RACE_CANONICAL_LABELS",
    "RACE_LABELS_DEEPFACE",
    "RACE_LABELS_FAIRFACE",
    "run_3d_reconstruction",
    "run_age_progression",
    "save_face",
    "save_gallery",
    "select_age",
    "select_gender",
    "select_race",
    "SHARPEN_METHODS",
    "validate_lbph_name",
    "VoiceFaceFusion",
    "with_headline",
]
