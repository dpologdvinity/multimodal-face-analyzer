"""Small, model-free liveness signals used by the face analysis pipeline."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence


BLINK_THRESHOLD = 0.55
TEXTURE_ARTIFACT_THRESHOLD = 0.72


@dataclass(frozen=True)
class LivenessResult:
    status: str
    blink_count: int
    blink_rate: float | None
    texture_score: float
    texture_artifact: bool

    @property
    def summary(self) -> str:
        rate = "n/a" if self.blink_rate is None else f"{self.blink_rate:.1f}/min"
        return f"{self.status} (blinks={self.blink_count}, rate={rate}, texture={self.texture_score:.2f})"


@dataclass
class _TrackState:
    first_seen: float
    last_seen: float
    eye_closed: bool = False
    eye_seen_open: bool = False
    blink_count: int = 0


def _axis_differences(gray: Sequence[Sequence[float]], step: int) -> tuple[list[float], list[float]]:
    adjacent: list[float] = []
    repeated: list[float] = []
    for row in gray:
        for index in range(step, len(row)):
            adjacent.append(abs(float(row[index]) - float(row[index - 1])))
            if step == 2:
                repeated.append(abs(float(row[index]) - float(row[index - step])))

    height = len(gray)
    width = len(gray[0]) if height else 0
    for y in range(step, height):
        for x in range(width):
            adjacent.append(abs(float(gray[y][x]) - float(gray[y - 1][x])))
            if step == 2:
                repeated.append(abs(float(gray[y][x]) - float(gray[y - step][x])))
    return adjacent, repeated


def texture_artifact_score(gray: Sequence[Sequence[float]]) -> float:
    """Score regular high-frequency texture, a replay/screen artifact cue.

    This is a heuristic. It detects strong alternating or periodic texture, not every spoof.
    Input must be a rectangular grayscale sample with values in the 0..255 range.
    """
    if not gray or not gray[0] or any(len(row) != len(gray[0]) for row in gray):
        return 0.0
    if len(gray) < 3 or len(gray[0]) < 3:
        return 0.0

    adjacent, repeated = _axis_differences(gray, 2)
    if not adjacent:
        return 0.0
    adjacent_mean = fmean(adjacent)
    high_frequency = min(1.0, adjacent_mean / 85.0)
    repeated_mean = fmean(repeated) if repeated else adjacent_mean
    periodicity = max(0.0, 1.0 - repeated_mean / (adjacent_mean + 1e-6))
    return max(0.0, min(1.0, high_frequency * periodicity))


def blink_score_from_landmarker(result) -> float | None:
    """Return mean left/right eye-blink blendshape score, or None when unavailable."""
    if result is None or not getattr(result, "face_blendshapes", None):
        return None
    blendshapes = result.face_blendshapes[0] if result.face_blendshapes else []
    scores = {
        item.category_name: float(item.score)
        for item in blendshapes
        if getattr(item, "category_name", None) in {"eyeBlinkLeft", "eyeBlinkRight"}
    }
    values = [scores[name] for name in ("eyeBlinkLeft", "eyeBlinkRight") if name in scores]
    return fmean(values) if values else None


def _make_result(
    blink_count: int,
    elapsed: float | None,
    texture_score: float,
    texture_threshold: float,
) -> LivenessResult:
    texture_artifact = texture_score >= texture_threshold
    if texture_artifact:
        status = "SUSPECTED SPOOF"
    elif blink_count:
        status = "LIVE"
    else:
        status = "INCONCLUSIVE"
    blink_rate = blink_count * 60.0 / elapsed if elapsed and elapsed > 0 else None
    return LivenessResult(status, blink_count, blink_rate, texture_score, texture_artifact)


def assess_static_liveness(texture_score: float) -> LivenessResult:
    """Assess one image without temporal evidence."""
    return _make_result(0, None, texture_score, TEXTURE_ARTIFACT_THRESHOLD)


class LivenessTracker:
    """Track blink transitions per stable face ID across webcam frames."""

    def __init__(
        self,
        blink_threshold: float = BLINK_THRESHOLD,
        texture_threshold: float = TEXTURE_ARTIFACT_THRESHOLD,
    ) -> None:
        self._blink_threshold = blink_threshold
        self._texture_threshold = texture_threshold
        self._states: dict[int, _TrackState] = {}
        self._lock = threading.Lock()

    def update(
        self,
        track_id: int,
        blink_score: float | None,
        texture_score: float,
        now: float | None = None,
    ) -> LivenessResult:
        """Update one face and return current liveness state."""
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            state = self._states.get(track_id)
            if state is None:
                state = _TrackState(timestamp, timestamp)
                self._states[track_id] = state
            timestamp = max(timestamp, state.last_seen)
            is_closed = blink_score is not None and blink_score >= self._blink_threshold
            if blink_score is not None:
                if not is_closed:
                    state.eye_seen_open = True
                elif state.eye_seen_open and not state.eye_closed:
                    state.blink_count += 1
                state.eye_closed = is_closed
            state.last_seen = timestamp
            elapsed = timestamp - state.first_seen
            return _make_result(state.blink_count, elapsed, texture_score, self._texture_threshold)

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
