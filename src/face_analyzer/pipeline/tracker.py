"""Multi-face tracking using greedy IoU matching across video frames."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from ..core.constants import IOU_TRACKING_THRESHOLD, TRACKING_MAX_MISSED_FRAMES


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Standard intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Track:
    """Internal state for a single tracked face."""

    box: tuple[int, int, int, int]
    missed_frames: int = 0


class FaceTracker:
    """Multi-face tracker that assigns stable IDs across video frames via greedy IoU matching."""

    def __init__(
        self,
        iou_threshold: float = IOU_TRACKING_THRESHOLD,
        max_missed_frames: int = TRACKING_MAX_MISSED_FRAMES,
    ):
        """Initialize tracker with IoU matching threshold and maximum missed frames."""
        self._lock = threading.Lock()
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, boxes: list[tuple[int, int, int, int]]) -> list[int]:
        """Match this frame's detections against existing tracks and return stable track IDs."""
        with self._lock:
            candidates = []
            for track_id, track in self._tracks.items():
                for det_index, box in enumerate(boxes):
                    iou = _box_iou(track.box, tuple(box))
                    if iou >= self._iou_threshold:
                        candidates.append((iou, track_id, det_index))
            candidates.sort(key=lambda c: c[0], reverse=True)

            track_for_detection: dict[int, int] = {}
            used_tracks: set[int] = set()
            for _iou, track_id, det_index in candidates:
                if track_id in used_tracks or det_index in track_for_detection:
                    continue
                track_for_detection[det_index] = track_id
                used_tracks.add(track_id)

            result_ids = []
            for det_index, box in enumerate(boxes):
                track_id = track_for_detection.get(det_index)
                if track_id is None:
                    track_id = self._next_id
                    self._next_id += 1
                self._tracks[track_id] = _Track(box=tuple(box), missed_frames=0)
                result_ids.append(track_id)

            handled_this_frame = set(result_ids)
            for track_id in list(self._tracks):
                if track_id in handled_this_frame:
                    continue
                track = self._tracks[track_id]
                track.missed_frames += 1
                if track.missed_frames > self._max_missed_frames:
                    del self._tracks[track_id]

            return result_ids

    def reset(self) -> None:
        """Drop every tracked face and restart ID numbering from 1."""
        with self._lock:
            self._tracks.clear()
            self._next_id = 1


__all__ = ["FaceTracker", "_box_iou", "_Track"]
