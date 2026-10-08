"""Streamlit entry point: page setup, theme, model loading, sidebar, and the upload/webcam tabs."""

import streamlit as st

# Absolute imports: Streamlit runs this file as a script, so it has no parent package.
from face_analyzer import inference
from face_analyzer.ui.live import new_live_session, render_live_tab
from face_analyzer.ui.results import process_and_display
from face_analyzer.ui.sidebar import render_sidebar
from face_analyzer.ui.theme import apply_theme, inject_css, select_theme

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
decode_image_bytes = st.cache_data(max_entries=16)(inference.decode_image_bytes)

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
        render_live_tab(
            models, sidebar, live_session,
            global_adjustments=global_adjustments, face_adjustments=face_adjustments,
        )
