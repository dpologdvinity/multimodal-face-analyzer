"""Transparent face-geometry proxy for exploratory body-composition research.

This is deliberately not a clinical BMI or body-fat estimator.  It turns the relative
width and aspect ratio of a detected face's lower geometry into a repeatable ordinal index
for within-dataset exploration.  A calibrated model and subject metadata would be required
before interpreting the index as a physical measurement.
"""
from __future__ import annotations

BODY_COMPOSITION_MODEL_OPTIONS = ["face_geometry"]
MIN_LANDMARKS = 6


def _landmark_xy(result) -> list[tuple[float, float]]:
    if result is None or not getattr(result, "face_landmarks", None):
        return []
    return [
        (float(point.x), float(point.y))
        for point in result.face_landmarks[0]
        if hasattr(point, "x") and hasattr(point, "y")
    ]


def _width(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    xs = [point[0] for point in points]
    return max(xs) - min(xs)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def estimate_face_composition(result) -> str:
    """Return a coarse facial-adiposity proxy from normalized face landmarks.

    The index is intentionally relative (0..100), not a BMI or body-fat percentage.  The
    lower-face width and overall face aspect ratio are scale-invariant and make this useful
    for exploratory ranking or data-quality checks when capture conditions are controlled.
    """
    points = _landmark_xy(result)
    if len(points) < MIN_LANDMARKS:
        return "insufficient landmarks"

    ys = [point[1] for point in points]
    face_height = max(ys) - min(ys)
    face_width = _width(points)
    lower_points = [point for point in points if point[1] >= min(ys) + face_height * 0.5]
    if face_height <= 0.0 or face_width <= 0.0 or len(lower_points) < 3:
        return "insufficient landmarks"

    lower_width_ratio = _clamp(_width(lower_points) / face_width, 0.0, 1.0)
    face_ratio = _clamp(face_width / face_height, 0.0, 2.0)
    # The range is a transparent normalization for display, not a learned medical model.
    geometry_score = 0.55 * lower_width_ratio + 0.45 * (face_ratio / 2.0)
    index = round(_clamp((geometry_score - 0.25) / 0.55 * 100.0, 0.0, 100.0))

    if index < 33:
        band = "lower relative fullness"
    elif index < 66:
        band = "mid relative fullness"
    else:
        band = "higher relative fullness"
    return f"BMI/body-fat proxy: facial adiposity proxy, index={index}/100 ({band}; research only)"
