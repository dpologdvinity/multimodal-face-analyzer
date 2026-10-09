"""Fusion subpackage: model ranking and ensemble fusion for multi-model predictions."""
from __future__ import annotations

# Re-export constants used by callers (e.g., tests, inference.py) via face_analyzer.fusion
from ..core.constants import (
    BEST_MODEL_KEY,
    FUSED_MODEL_KEY,
    RACE_CANONICAL_LABELS,
    RACE_LABELS_FAIRFACE,
)
from .ensembles import (
    canonical_race_probabilities,
    fuse_emotion,
)
from .ranking import (
    _format_results,
    _gather_face_results,
    _sanitize_column_name,
    select_age,
    select_gender,
    select_race,
    with_headline,
)

__all__ = [
    # ranking
    "select_age",
    "select_gender",
    "select_race",
    "with_headline",
    "_format_results",
    "_sanitize_column_name",
    "_gather_face_results",
    # ensembles
    "canonical_race_probabilities",
    "fuse_emotion",
    # constants re-exported for backward compatibility
    "RACE_CANONICAL_LABELS",
    "RACE_LABELS_FAIRFACE",
    "FUSED_MODEL_KEY",
    "BEST_MODEL_KEY",
]
