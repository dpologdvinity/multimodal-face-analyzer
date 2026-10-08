"""Live webcam tab: per-session stream state, the streamlit-webrtc frame callback, and its view."""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import av
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from .results import _target_card_html
from .sidebar import SidebarState

try:
    from src.inference import (
        FaceTracker,
        LivenessTracker,
        VoiceFaceFusion,
        analyze_frame,
        audio_frame_to_mono_float,
        fuse_voice_and_emotion,
        maybe_colorize,
    )
    from src.pipeline.config import AnalysisConfig
except ImportError:
    from inference import (
        FaceTracker,
        LivenessTracker,
        VoiceFaceFusion,
        analyze_frame,
        audio_frame_to_mono_float,
        fuse_voice_and_emotion,
        maybe_colorize,
    )
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


def new_live_session() -> dict[str, Any]:
    """Create one script run's live stream state, keyed as make_video_frame_callback expects."""
    # Called by app.py on every script run: module-level state here would outlive the run and be
    # shared by every browser session, since imported modules persist across sessions.
    return {
        "live_state": {"faces": [], "error": None, "updated": 0.0},
        "live_state_lock": threading.Lock(),
        "live_metrics": deque(maxlen=120),
        "live_metrics_lock": threading.Lock(),
    }


@st.cache_resource
def _get_face_tracker() -> FaceTracker:
    """#2: one FaceTracker instance for the live webcam stream, cached (not session_state) so
    it's the same object across Streamlit reruns and reachable from streamlit-webrtc's own
    callback thread -- same reasoning as app.py's load_models(), see FaceTracker's docstring."""
    return FaceTracker()


@st.cache_resource
def _get_liveness_tracker() -> LivenessTracker:
    """Keep blink history stable across Streamlit reruns for the LIVE webcam stream."""
    return LivenessTracker()


@st.cache_resource
def _get_voice_fusion() -> VoiceFaceFusion:
    """#10: same caching reasoning as _get_face_tracker() above -- the audio callback and the
    video callback are different threads and need to share the SAME buffer instance."""
    return VoiceFaceFusion()


def render_live_tab(
    models: Any, sidebar: SidebarState, live_session: dict[str, Any], *,
    global_adjustments: dict, face_adjustments: dict,
) -> None:
    """Render the Live capture mode: stream controls, the WebRTC feed, and live/after-run stats."""
    state, state_lock = live_session["live_state"], live_session["live_state_lock"]
    metrics_buffer, metrics_lock = live_session["live_metrics"], live_session["live_metrics_lock"]
    st.caption("Live analysis updates as people enter or leave view. Each face keeps a stable ID as it moves.")
    frame_skip = st.slider(
        "Classifier frame skip", 1, 10, 1,
        help="Run age/gender/emotion/race/recognition/etc. classifiers every Nth frame "
        "instead of every frame. Face detection and the hand/face-landmark overlays "
        "still run every frame, so the video stays smooth. These classifiers' outputs "
        "aren't otherwise drawn onto the LIVE video (see target cards in Image upload / "
        "SNAPSHOT for that), so skipping them here only reduces CPU load, with no visible "
        "staleness to interpolate around.",
    )
    face_tracker = _get_face_tracker()
    liveness_tracker = _get_liveness_tracker()
    reset_col, voice_col = st.columns([1, 2])
    if reset_col.button("Reset tracking IDs", key="reset_tracking_ids"):
        face_tracker.reset()
        liveness_tracker.reset()

    # #10: off by default -- requesting the microphone is a permission prompt the user
    # didn't ask for just by opening the webcam tab, so it needs its own explicit opt-in
    # rather than riding along with LIVE mode's existing camera request.
    enable_voice_fusion = voice_col.checkbox(
        "Enable microphone-assisted fusion (experimental)", value=False, key="enable_voice_fusion",
        help="Heuristic only: cross-checks mic loudness against the largest face's emotion "
             "label. Not a trained speech-emotion model -- see src/inference.py's "
             "VoiceFaceFusion docstring for why.",
    )
    voice_fusion = _get_voice_fusion() if enable_voice_fusion else None
    if voice_fusion is not None and voice_col.button("Reset voice buffer", key="reset_voice_buffer"):
        voice_fusion.reset()
    gallery_snapshot = dict(st.session_state.get("gallery", {}))

    live_config = sidebar.to_config(
        gallery=gallery_snapshot,
        global_adjustments=global_adjustments, face_adjustments=face_adjustments,
        tracker=face_tracker, liveness_tracker=liveness_tracker,
    )
    _video_frame_callback = make_video_frame_callback(
        lambda: models, lambda: live_config,
        **live_session,
        frame_skip=frame_skip, active_colorization=sidebar.active_colorization, voice_fusion=voice_fusion,
    )

    def _audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        """Ingest audio samples into voice fusion tracker if enabled."""
        if voice_fusion is not None:
            samples = audio_frame_to_mono_float(frame.to_ndarray())
            voice_fusion.ingest_audio(samples, frame.sample_rate)
        return frame

    webrtc_ctx = webrtc_streamer(
        key="live-face-feed",
        video_frame_callback=_video_frame_callback,
        audio_frame_callback=_audio_frame_callback if enable_voice_fusion else None,
        media_stream_constraints={"video": True, "audio": enable_voice_fusion},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    if voice_fusion is not None:
        status = voice_fusion.get_latest_status()
        if status is None:
            st.caption("Microphone fusion is waiting for audio and a detected face…")
        else:
            consistency_text = status["consistency"] or "n/a (emotion label not categorized)"
            st.caption(
                f"Microphone fusion: {status['voice_arousal']}; largest face emotion: "
                f"{status['emotion']}; consistency: {consistency_text}."
            )

    live_info = st.empty()
    live_fps = st.empty()
    while webrtc_ctx.state.playing:
        with state_lock:
            live_snapshot = dict(state)
        if live_snapshot["error"]:
            live_info.error(f"Live frame error: {live_snapshot['error']}")
        elif live_snapshot["faces"]:
            with live_info.container():
                st.markdown("#### Live face details")
                for face in live_snapshot["faces"]:
                    st.markdown(_target_card_html(face), unsafe_allow_html=True)
        else:
            live_info.caption("Waiting for a detected face…")
        with metrics_lock:
            live_metrics = list(metrics_buffer)
        intervals = np.diff([item["timestamp"] for item in live_metrics[-30:]])
        fps = 1.0 / float(np.mean(intervals)) if len(intervals) and np.mean(intervals) > 0 else 0.0
        live_fps.metric("Live FPS", f"{fps:.1f}")
        time.sleep(0.25)
    with metrics_lock:
        live_metrics = list(metrics_buffer)
    if live_metrics:
        latency_rows = []
        for item in live_metrics:
            for model_name, values in item.get("model_latency_ms", {}).items():
                latency_rows.extend({"Model": model_name, "Latency (ms)": value} for value in values)
        if latency_rows:
            latency_frame = pd.DataFrame(latency_rows)
            summary = latency_frame.groupby("Model", as_index=False)["Latency (ms)"].mean()
            summary["Latency (ms)"] = summary["Latency (ms)"].round(1)
            st.dataframe(summary, hide_index=True, width="stretch")
        emotion_rows = []
        start_time = live_metrics[0]["timestamp"]
        for item in live_metrics:
            for sample in item.get("emotion_samples", []):
                emotion_rows.append({
                    "Seconds": item["timestamp"] - start_time,
                    "Model": sample["model"],
                    "Emotion": sample["emotion"],
                })
        if emotion_rows:
            st.markdown("#### Emotion over time")
            emotion_frame = pd.DataFrame(emotion_rows)
            for model_name, model_frame in emotion_frame.groupby("Model"):
                labels = sorted(model_frame["Emotion"].unique())
                chart_rows = []
                for _, row in model_frame.iterrows():
                    chart_rows.append({
                        "Seconds": row["Seconds"],
                        **{label: float(label == row["Emotion"]) for label in labels},
                    })
                chart = pd.DataFrame(chart_rows).groupby("Seconds").max().sort_index()
                st.caption(f"{model_name}: dominant emotion (1 = active, 0 = inactive)")
                st.line_chart(chart, height=220)
