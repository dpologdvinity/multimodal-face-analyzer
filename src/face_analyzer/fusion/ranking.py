"""Ranking utilities for multi-model age/gender/race selection and result formatting."""
from __future__ import annotations

import numpy as np

from ..core.constants import (
    AGE_MODEL_RELIABILITY,
    FUSED_MODEL_KEY,
    GENDER_MODEL_RELIABILITY,
    RACE_MODEL_RELIABILITY,
)


def _most_reliable(usable: dict, ranking: tuple[str, ...]) -> str:
    """Return the key of usable that ranks first in ranking (unranked keys come last)."""
    return min(usable, key=lambda key: ranking.index(key) if key in ranking else len(ranking))


def select_age(estimates: dict[str, float]) -> tuple[str, str] | None:
    """Pick the headline age as the most reliable model present, returning (label, model_key)."""
    usable = {key: value for key, value in estimates.items()
              if value is not None and np.isfinite(value) and 0 <= value <= 122}
    if len(usable) < 2:
        return None
    chosen = _most_reliable(usable, AGE_MODEL_RELIABILITY)
    return f"{usable[chosen]:.0f}", chosen


def select_gender(labels: dict[str, str]) -> tuple[str, str] | None:
    """Pick the headline gender as the most reliable model present, returning (label, model_key)."""
    usable = {key: label for key, label in labels.items() if label}
    if len(usable) < 2:
        return None
    chosen = _most_reliable(usable, GENDER_MODEL_RELIABILITY)
    return usable[chosen], chosen


def select_race(labels: dict[str, str]) -> tuple[str, str] | None:
    """Pick the headline race as the most reliable model present, returning (label, model_key)."""
    usable = {key: label for key, label in labels.items() if label}
    if len(usable) < 2:
        return None
    chosen = _most_reliable(usable, RACE_MODEL_RELIABILITY)
    return usable[chosen], chosen


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
