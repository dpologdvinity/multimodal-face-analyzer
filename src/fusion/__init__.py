"""Fusion subpackage: model ranking and ensemble fusion for multi-model predictions."""
from __future__ import annotations

from .ranking import (
    _weighted_median,
    select_age,
    with_headline,
    _format_results,
    _sanitize_column_name,
    _gather_face_results,
)

from .ensembles import (
    fuse_gender,
    canonical_race_probabilities,
    fuse_race,
    fuse_emotion,
    _softmax,
)

# Re-export constants used by callers (e.g., tests, inference.py) via src.fusion
try:
    from src.core.constants import (
        RACE_CANONICAL_LABELS,
        RACE_LABEL_TO_CANONICAL,
        RACE_LABELS_FAIRFACE,
        RACE_LABELS_DEEPFACE,
        FUSED_MODEL_KEY,
        BEST_MODEL_KEY,
        HEADLINE_MODEL_KEYS,
        AGE_MODEL_RELIABILITY,
        GENDER_FUSION_WEIGHTS,
        RACE_FUSION_WEIGHTS,
        EMOTION_FUSION_WEIGHTS,
        EMOTION_CANONICAL,
    )
except ImportError:
    from core.constants import (
        RACE_CANONICAL_LABELS,
        RACE_LABEL_TO_CANONICAL,
        RACE_LABELS_FAIRFACE,
        RACE_LABELS_DEEPFACE,
        FUSED_MODEL_KEY,
        BEST_MODEL_KEY,
        HEADLINE_MODEL_KEYS,
        AGE_MODEL_RELIABILITY,
        GENDER_FUSION_WEIGHTS,
        RACE_FUSION_WEIGHTS,
        EMOTION_FUSION_WEIGHTS,
        EMOTION_CANONICAL,
    )

__all__ = [
    # ranking
    "_weighted_median",
    "select_age",
    "with_headline",
    "_format_results",
    "_sanitize_column_name",
    "_gather_face_results",
    # ensembles
    "fuse_gender",
    "canonical_race_probabilities",
    "fuse_race",
    "fuse_emotion",
    "_softmax",
    # constants re-exported for backward compatibility
    "RACE_CANONICAL_LABELS",
    "RACE_LABEL_TO_CANONICAL",
    "RACE_LABELS_FAIRFACE",
    "RACE_LABELS_DEEPFACE",
    "FUSED_MODEL_KEY",
    "BEST_MODEL_KEY",
    "HEADLINE_MODEL_KEYS",
    "AGE_MODEL_RELIABILITY",
    "GENDER_FUSION_WEIGHTS",
    "RACE_FUSION_WEIGHTS",
    "EMOTION_FUSION_WEIGHTS",
    "EMOTION_CANONICAL",
]
