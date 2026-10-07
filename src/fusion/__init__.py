"""Fusion subpackage: model ranking and ensemble fusion for multi-model predictions."""
from __future__ import annotations

from .ensembles import (
    canonical_race_probabilities,
    fuse_emotion,
    fuse_gender,
    fuse_race,
)
from .ranking import (
    _format_results,
    _gather_face_results,
    _sanitize_column_name,
    _weighted_median,
    select_age,
    with_headline,
)

# Re-export constants used by callers (e.g., tests, inference.py) via src.fusion
try:
    from src.core.constants import (
        AGE_MODEL_RELIABILITY,
        BEST_MODEL_KEY,
        EMOTION_CANONICAL,
        EMOTION_FUSION_WEIGHTS,
        FUSED_MODEL_KEY,
        GENDER_FUSION_WEIGHTS,
        HEADLINE_MODEL_KEYS,
        RACE_CANONICAL_LABELS,
        RACE_FUSION_WEIGHTS,
        RACE_LABEL_TO_CANONICAL,
        RACE_LABELS_DEEPFACE,
        RACE_LABELS_FAIRFACE,
    )
except ImportError:
    from core.constants import (
        AGE_MODEL_RELIABILITY,
        BEST_MODEL_KEY,
        EMOTION_CANONICAL,
        EMOTION_FUSION_WEIGHTS,
        FUSED_MODEL_KEY,
        GENDER_FUSION_WEIGHTS,
        HEADLINE_MODEL_KEYS,
        RACE_CANONICAL_LABELS,
        RACE_FUSION_WEIGHTS,
        RACE_LABEL_TO_CANONICAL,
        RACE_LABELS_DEEPFACE,
        RACE_LABELS_FAIRFACE,
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
