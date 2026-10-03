"""Ranking utilities for multi-model age selection and result formatting."""
from __future__ import annotations

import numpy as np

try:
    from src.core.constants import (
        AGE_MODEL_RELIABILITY,
        FUSED_MODEL_KEY,
        RACE_CANONICAL_LABELS,
        RACE_LABEL_TO_CANONICAL,
        RACE_LABELS_FAIRFACE,
        RACE_LABELS_DEEPFACE,
        RACE_CLOSE_MARGIN,
        GENDER_FUSION_WEIGHTS,
        RACE_FUSION_WEIGHTS,
        EMOTION_FUSION_WEIGHTS,
        EMOTION_CANONICAL,
    )
except ImportError:
    from core.constants import (
        AGE_MODEL_RELIABILITY,
        FUSED_MODEL_KEY,
        RACE_CANONICAL_LABELS,
        RACE_LABEL_TO_CANONICAL,
        RACE_LABELS_FAIRFACE,
        RACE_LABELS_DEEPFACE,
        RACE_CLOSE_MARGIN,
        GENDER_FUSION_WEIGHTS,
        RACE_FUSION_WEIGHTS,
        EMOTION_FUSION_WEIGHTS,
        EMOTION_CANONICAL,
    )


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Return the value where cumulative weight first reaches half the total."""
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    return float(sorted_values[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def select_age(estimates: dict[str, float]) -> tuple[str, str] | None:
    """Pick the headline age as the most reliable model present, returning (label, model_key)."""
    usable = {key: value for key, value in estimates.items()
              if value is not None and np.isfinite(value) and 0 <= value <= 122}
    if len(usable) < 2:
        return None
    ranked = sorted(usable, key=lambda key: AGE_MODEL_RELIABILITY.index(key)
                    if key in AGE_MODEL_RELIABILITY else len(AGE_MODEL_RELIABILITY))
    chosen = ranked[0]
    return f"{usable[chosen]:.0f}", chosen


def with_headline(
    pairs: list[tuple[str, str]], headline: str | None, key: str = FUSED_MODEL_KEY,
) -> list[tuple[str, str]]:
    """Prepend a combined answer to a feature's model pairs so it leads every display."""
    return ([(key, headline)] if headline else []) + pairs


def _format_results(pairs: list[tuple[str, str]]) -> list[str]:
    """Return plain values only, without model-name prefix."""
    return [value for _, value in pairs]


def _sanitize_column_name(feature: str, model_key: str) -> str:
    """Convert feature/model names into a valid SQLite column identifier."""
    return f"{feature}_{model_key}".lower().replace(" ", "_").replace("-", "_")


def _gather_face_results(pairs_by_feature: dict[str, list[tuple[str, str]]]) -> dict[str, str]:
    """Flatten per-feature (model_key, value) pairs into {column_name: value}."""
    results = {}
    for feature, pairs in pairs_by_feature.items():
        for model_key, value in pairs:
            results[_sanitize_column_name(feature, model_key)] = value
    return results
