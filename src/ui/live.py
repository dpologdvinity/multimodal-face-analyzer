"""Live webcam stream state and the per-frame callback streamlit-webrtc invokes."""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import av

try:
    from src.inference import analyze_frame, fuse_voice_and_emotion, maybe_colorize
    from src.pipeline.config import AnalysisConfig
except ImportError:
    from inference import analyze_frame, fuse_voice_and_emotion, maybe_colorize
    from pipeline.config import AnalysisConfig

# Slow per-face classifiers that frame skipping throttles; detection, the landmark and hand
# overlays, and liveness are cheap enough to keep running every frame.
_THROTTLED_FIELDS = (
    "active_age", "active_gender", "active_emotion", "active_race", "active_recognition",
    "active_glasses", "active_mask", "active_hair_color", "active_eye_color", "active_gaze",
)


def make_video_frame_callback(
    models_provider: Callable[[], Any],
    config_provider: Callable[[], AnalysisConfig],
    *,
    live_state: dict[str, Any],
    live_state_lock: threading.Lock,
    live_metrics: deque,
    live_metrics_lock: threading.Lock,
    frame_skip: int = 1,
    active_colorization: set | None = None,
    voice_fusion: Any | None = None,
) -> Callable[[av.VideoFrame], av.VideoFrame]:
    """Build the streamlit-webrtc callback that analyzes frames and publishes live state.

    Classifiers run on every ``frame_skip``-th frame; detection and overlays run on all of them.
    The live_* state belongs to the caller (one set per Streamlit session and rerun) and is only
    touched under its lock.
    """
    colorization = active_colorization if active_colorization is not None else set()
    frame_counter = 0

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        """Run detection/inference on one frame, update live state, handle voice fusion."""
        nonlocal frame_counter
        try:
            frame_started = time.perf_counter()
            metrics: dict = {}
            models = models_provider()
            img = frame.to_ndarray(format="bgr24")
            img, _ = maybe_colorize(models, img, colorization)
            frame_counter += 1
            run_classifiers = frame_counter % frame_skip == 0
            config = config_provider()
            # replace() copies the shared config so per-frame metrics and throttling never leak
            # into the next frame or the other thread reading it.
            throttled = {} if run_classifiers else {name: set() for name in _THROTTLED_FIELDS}
            annotated_frame, cropped_faces, _, _ = analyze_frame(
                models, img, replace(config, metrics=metrics, **throttled),
            )
            metrics["frame_ms"] = (time.perf_counter() - frame_started) * 1000
            metrics["timestamp"] = time.monotonic()
            live_faces = [
                {key: face[key] for key in ("idx", "model_results")}
                for face in cropped_faces
            ]
            with live_metrics_lock:
                live_metrics.append(metrics)
            with live_state_lock:
                live_state.update(faces=live_faces, error=None, updated=time.monotonic())
            if voice_fusion is not None and cropped_faces:
                # v1 scope: fuse against the single largest detected face.
                largest = max(cropped_faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
                emotion_label = largest["emotion"][0] if largest["emotion"] else None
                voice_arousal = voice_fusion.current_arousal()
                voice_fusion.set_latest_status({
                    "voice_arousal": voice_arousal,
                    "emotion": emotion_label,
                    "consistency": fuse_voice_and_emotion(voice_arousal, emotion_label) if emotion_label else None,
                })
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as exc:
            # Inference failure must not crash the WebRTC video stream. Log the error to
            # live_state so the UI thread can display it, but always return a frame
            # (raw passthrough) to keep the stream alive and the camera usable.
            with live_state_lock:
                live_state["error"] = f"{type(exc).__name__}: {exc}"
            return frame

    return callback
