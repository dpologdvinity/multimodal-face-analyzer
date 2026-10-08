"""Ensemble fusion functions for combining predictions from multiple models."""
from __future__ import annotations

import numpy as np

from ..core.constants import (
    EMOTION_CANONICAL,
    EMOTION_FUSION_WEIGHTS,
    GENDER_FUSION_WEIGHTS,
    RACE_CANONICAL_LABELS,
    RACE_CLOSE_MARGIN,
    RACE_LABEL_TO_CANONICAL,
)


def _format_race_label(probs: np.ndarray, labels: list[str]) -> str:
    """Format the top race prediction, showing top-2 together when probabilities are close."""
    order = np.argsort(probs)[::-1]
    top1, top2 = order[0], order[1]
    if probs[top1] - probs[top2] < RACE_CLOSE_MARGIN:
        return f"{labels[top1]} ({probs[top1] * 100:.0f}%)/{labels[top2]} ({probs[top2] * 100:.0f}%)"
    return labels[top1]


def fuse_gender(male_probabilities: dict[str, float]) -> str | None:
    """Combine per-model P(Male) into one weighted-mean gender label."""
    usable = {key: value for key, value in male_probabilities.items()
              if value is not None and np.isfinite(value)}
    if len(usable) < 2:
        return None
    weights = np.array([GENDER_FUSION_WEIGHTS.get(key, 1.0) for key in usable])
    probability = float(np.array(list(usable.values())) @ weights / weights.sum())
    return "Male" if probability >= 0.5 else "Female"


def canonical_race_probabilities(probs: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Re-express one backend's class probabilities over the shared canonical race keys."""
    total = float(np.sum(probs)) or 1.0
    combined = dict.fromkeys(RACE_CANONICAL_LABELS, 0.0)
    for label, probability in zip(labels, probs, strict=False):
        key = RACE_LABEL_TO_CANONICAL.get(label.lower())
        if key is not None:
            combined[key] += float(probability) / total
    return combined


def fuse_emotion(labels: dict[str, str]) -> str | None:
    """Combine per-model emotion labels into one weighted-vote label."""
    if len(labels) < 2:
        return None
    votes: dict[str, float] = {}
    for model, label in labels.items():
        canonical = EMOTION_CANONICAL.get(str(label).lower())
        if canonical is None:
            continue
        votes[canonical] = votes.get(canonical, 0.0) + EMOTION_FUSION_WEIGHTS.get(model, 1.0)
    if not votes:
        return None
    return max(votes.items(), key=lambda item: item[1])[0]
