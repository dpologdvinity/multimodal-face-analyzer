"""Streamlit entry point: page setup, theme, model loading, sidebar, and the upload/webcam tabs."""

import streamlit as st

# Absolute imports: Streamlit runs this file as a script, so it has no parent package.
from face_analyzer import inference
from face_analyzer.demo import (
    DEMO_MAX_FILES,
    DEMO_MAX_IMAGE_PIXELS,
    DEMO_MAX_IMAGE_SIDE,
    DEMO_MAX_UPLOAD_MB,
    LIVE_MODE_NOTE,
    RESPONSIBLE_USE_NOTE,
    demo_attribution,
    is_demo_mode,
)
from face_analyzer.ui import SAMPLE_PHOTO, SAMPLE_PHOTO_CREDIT, SAMPLE_PHOTO_NAME
from face_analyzer.ui.live import new_live_session, render_live_tab
from face_analyzer.ui.results import process_and_display
from face_analyzer.ui.sidebar import render_sidebar
from face_analyzer.ui.theme import apply_theme, inject_css, select_theme

demo = is_demo_mode()
if demo:
    # Visitors see a generic error instead of tracebacks with server paths; the log keeps details.
    st.set_option("client.showErrorDetails", "none")

# Live webcam state, guarded by locks for thread-safe access from streamlit-webrtc callbacks
# running in separate threads. Streamlit re-executes this script per session and rerun, so it is
# created here on every run rather than held in an imported module, which would share it across
# sessions.
live_session = new_live_session()

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
_decode_image_bytes = st.cache_data(max_entries=16)(inference.decode_image_bytes)


def decode_image_bytes(file_bytes: bytes):
    """Decode an upload or snapshot; demo mode bounds its size and keeps it out of the cache.

    The demo shares one 2.7 GB process across visitors and promises that uploads are never
    stored, so it decodes afresh on every run instead of caching decoded frames.
    """
    if demo:
        return inference.decode_image_bytes(file_bytes, DEMO_MAX_IMAGE_SIDE, DEMO_MAX_IMAGE_PIXELS)
    return _decode_image_bytes(file_bytes)


try:
    models = load_models()
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

st.markdown(
    '<div class="wrap-hero">'
    '<div class="wrap-hero-intro"><h1>Every face,<br/>read back.</h1>'
    '<p><strong>Multimodal Face Analyzer.</strong> Upload a photo. Each face gets its own card, '
    'and every answer names the model that gave '
    'it. Age, gender and race come from the model that scored best on 1,002 held-out faces.</p></div>'
    f'<div class="wrap-hero-stat"><span class="wrap-big">{models.loaded_feature_count}</span>'
    f'<span class="wrap-caption">of {models.total_feature_count} analyses ready</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

if demo:
    st.info(RESPONSIBLE_USE_NOTE, icon=":material/info:")

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
        max_upload_size=DEMO_MAX_UPLOAD_MB if demo else None,
    )

    # The sample stays on across reruns (theme changes, editing) until hidden or replaced by an upload.
    if uploaded_files:
        st.session_state["show_sample"] = False
    else:
        showing = st.session_state.get("show_sample", False)
        if st.button("Hide the sample photo" if showing else "No photo? Try a sample",
                     icon=":material/photo_library:", key="sample_photo_btn"):
            st.session_state["show_sample"] = not showing
            st.rerun()
        if showing:
            st.caption(SAMPLE_PHOTO_CREDIT)
            process_and_display(
                models, sidebar, decode_image_bytes(SAMPLE_PHOTO.read_bytes()), SAMPLE_PHOTO_NAME,
                theme=theme, face_adjustments=face_adjustments,
            )

    if uploaded_files:
        if demo and len(uploaded_files) > DEMO_MAX_FILES:
            st.warning(f"The demo analyzes the first {DEMO_MAX_FILES} images of each upload.")
            uploaded_files = uploaded_files[:DEMO_MAX_FILES]
        for uploaded_file in uploaded_files:
            if demo and uploaded_file.size > DEMO_MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"{uploaded_file.name} is larger than the demo's {DEMO_MAX_UPLOAD_MB} MB limit.")
                continue
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
    if demo:
        st.caption(LIVE_MODE_NOTE)
        capture_mode = "Snapshot"
    else:
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
        render_live_tab(
            models, sidebar, live_session,
            global_adjustments=global_adjustments, face_adjustments=face_adjustments, theme=theme,
        )

if demo:
    st.divider()
    st.caption(demo_attribution(include_mediapipe=bool(models.face_landmarks_nets)))
