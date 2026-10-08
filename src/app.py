import threading
import time
from collections import deque

import av
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

import inference
from ui.live import make_video_frame_callback
from ui.results import _target_card_html, process_and_display
from ui.sidebar import render_sidebar
from ui.theme import apply_theme, inject_css, select_theme

# Live webcam state, guarded by locks for thread-safe access from streamlit-webrtc callbacks
# running in separate threads. Streamlit re-executes this script per session and rerun, so these
# must stay here rather than in an imported module, which would share them across sessions.
LIVE_METRICS = deque(maxlen=120)
LIVE_METRICS_LOCK = threading.Lock()
LIVE_STATE = {"faces": [], "error": None, "updated": 0.0}
LIVE_STATE_LOCK = threading.Lock()

# Page setup and visual system
st.set_page_config(
    page_title="Multimodal Face Analyzer",
    page_icon=":material/face:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme = select_theme()
inject_css()

# Cache load_models so models persist across Streamlit reruns (avoids reloading expensive
# neural network weights for each interaction with sliders, buttons, tabs, etc.).
load_models = st.cache_resource(inference.load_models)

# Same "avoid redoing pure work on an unrelated rerun" reasoning as load_models above: any
# widget interaction (a model checkbox, an export button) reruns this whole script, which
# would otherwise re-decode (and re-downscale) the same uploaded/captured image bytes every
# time. max_entries bounds memory since each cached entry holds a full decoded frame.
decode_image_bytes = st.cache_data(max_entries=16)(inference.decode_image_bytes)


@st.cache_resource
def _get_face_tracker() -> inference.FaceTracker:
    """#2: one FaceTracker instance for the live webcam stream, cached (not session_state) so
    it's the same object across Streamlit reruns and reachable from streamlit-webrtc's own
    callback thread -- same reasoning as load_models() above, see FaceTracker's docstring."""
    return inference.FaceTracker()


@st.cache_resource
def _get_liveness_tracker() -> inference.LivenessTracker:
    """Keep blink history stable across Streamlit reruns for the LIVE webcam stream."""
    return inference.LivenessTracker()


@st.cache_resource
def _get_voice_fusion() -> inference.VoiceFaceFusion:
    """#10: same caching reasoning as _get_face_tracker() above -- the audio callback and the
    video callback are different threads and need to share the SAME buffer instance."""
    return inference.VoiceFaceFusion()

try:
    models = load_models()
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

st.markdown(
    '<div class="app-hero">'
    '<div class="app-hero-heading"><h1>Multimodal Face Analyzer</h1>'
    '<p>Upload a photo or open your webcam, turn on the detectors you want, '
    'then read each face\'s results below.</p></div>'
    f'<div class="app-hero-readout">DETECTORS READY<br/>'
    f'<strong>{models.loaded_feature_count} / {models.total_feature_count}</strong></div>'
    '</div>',
    unsafe_allow_html=True,
)

sidebar = render_sidebar(models)

apply_theme(theme)


global_adjustments = {
    name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()
}
face_adjustments = {
    name: values[2] for name, values in inference.IMAGE_ADJUSTMENT_RANGES.items()
}

tab_upload, tab_webcam = st.tabs(["Image upload", "Webcam"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Choose images to analyze",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                frame = decode_image_bytes(uploaded_file.read())
            except ValueError as exc:
                st.error(f"Could not read {uploaded_file.name}: {exc}")
                continue
            process_and_display(
                models, sidebar, frame, uploaded_file.name,
                theme=theme, face_adjustments=face_adjustments,
            )

with tab_webcam:
    capture_mode = st.segmented_control("Capture mode", ["Snapshot", "Live"], default="Snapshot")

    if capture_mode == "Snapshot":
        webcam_image = st.camera_input("Take a snapshot")

        if webcam_image:
            try:
                frame = decode_image_bytes(webcam_image.read())
            except ValueError as exc:
                st.error(f"Could not read camera image: {exc}")
                frame = None
            if frame is not None:
                process_and_display(
                    models, sidebar, frame, "WEBCAM_CAPTURE",
                    theme=theme, face_adjustments=face_adjustments,
                )
    else:
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
            live_state=LIVE_STATE, live_state_lock=LIVE_STATE_LOCK,
            live_metrics=LIVE_METRICS, live_metrics_lock=LIVE_METRICS_LOCK,
            frame_skip=frame_skip, active_colorization=sidebar.active_colorization, voice_fusion=voice_fusion,
        )

        def _audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
            """Ingest audio samples into voice fusion tracker if enabled."""
            if voice_fusion is not None:
                samples = inference.audio_frame_to_mono_float(frame.to_ndarray())
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
            with LIVE_STATE_LOCK:
                live_snapshot = dict(LIVE_STATE)
            if live_snapshot["error"]:
                live_info.error(f"Live frame error: {live_snapshot['error']}")
            elif live_snapshot["faces"]:
                with live_info.container():
                    st.markdown("#### Live face details")
                    for face in live_snapshot["faces"]:
                        st.markdown(_target_card_html(face), unsafe_allow_html=True)
            else:
                live_info.caption("Waiting for a detected face…")
            with LIVE_METRICS_LOCK:
                live_metrics = list(LIVE_METRICS)
            intervals = np.diff([item["timestamp"] for item in live_metrics[-30:]])
            fps = 1.0 / float(np.mean(intervals)) if len(intervals) and np.mean(intervals) > 0 else 0.0
            live_fps.metric("Live FPS", f"{fps:.1f}")
            time.sleep(0.25)
        with LIVE_METRICS_LOCK:
            live_metrics = list(LIVE_METRICS)
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
