import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

import inference

# Page setup & surveillance-terminal style injection
st.set_page_config(page_title="MULTIMODAL_FACE_ANALYZER", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    /* Main app background and font */
    .stApp {
        background-color: #0d1117;
        color: #00ff66;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff66 !important;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* File uploader custom styling */
    div[data-testid="stFileUploader"] {
        border: 1px dashed #00ff66;
        border-radius: 4px;
        background-color: #010409;
        padding: 10px;
    }

    /* Warning and info alerts */
    .stAlert {
        background-color: #161b22;
        color: #ffcc00;
        border: 1px solid #ffcc00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MULTIMODAL FACE ANALYZER")
st.caption("[ STATUS: ONLINE ] -- Multi-Model Face Analysis")

load_models = st.cache_resource(inference.load_models)

try:
    models = load_models()
except Exception as e:
    st.error(f"[SYSTEM ERROR] Failed to load models: {e}")
    st.stop()

if models.offline_features:
    st.sidebar.caption(f"[ OFFLINE: {', '.join(models.offline_features)} ] -- image built without these model file(s)")


def _model_checkboxes(label: str, nets: dict) -> set:
    """Render one checkbox per loaded model for a feature; return the set of checked keys."""
    active = set()
    if not nets:
        return active
    st.sidebar.markdown(f"**{label}**")
    for key in nets:
        if st.sidebar.checkbox(key.upper(), value=True, key=f"chk_{label}_{key}"):
            active.add(key)
    return active


st.sidebar.markdown("### MODEL SELECTION")
active_age = _model_checkboxes("AGE", models.age_nets)
active_gender = _model_checkboxes("GENDER", models.gender_nets)
active_race = _model_checkboxes("RACE", models.race_nets)
active_emotion = _model_checkboxes("EMOTION", models.emotion_nets)
active_drowsiness = _model_checkboxes("DROWSINESS", models.drowsiness_nets)

# Sidebar Interface Controls
st.sidebar.markdown("### CONTROL PANEL")
crop_toggle = st.sidebar.toggle("CROP FACE TARGETS ONLY", value=False)
conf_threshold = st.sidebar.slider("CONFIDENCE THRESHOLD", 0.1, 1.0, 0.7)


def process_and_display(frame: np.ndarray, identifier: str, crop_toggle: bool, conf_threshold: float) -> None:
    """Run detection/inference on frame and render result in Streamlit."""
    annotated_frame, cropped_faces, any_drowsy, has_faces = inference.analyze_frame(
        models, frame, conf_threshold, active_age, active_gender, active_emotion, active_drowsiness, active_race
    )

    if not has_faces:
        st.warning(f"[TARGET MISSING] Zero targets detected in file: {identifier}")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=identifier, use_container_width=True)
        return

    st.markdown(f"#### ANALYSIS RESULT: `{identifier}`")

    if any_drowsy:
        st.error("[ALERT] DROWSINESS DETECTED -- SUBJECT EYES CLOSED")

    # Toggle Display Output Mode
    if crop_toggle:
        cols = st.columns(min(len(cropped_faces), 4))
        for idx, (label, crop_img) in enumerate(cropped_faces):
            with cols[idx % 4]:
                st.image(crop_img, caption=label, use_container_width=True)
    else:
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)


tab_upload, tab_webcam = st.tabs(["[ FILE UPLOAD ]", "[ LIVE WEBCAM ]"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "SELECT OR DROP IMAGE FILES FOR INFERENCE...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            process_and_display(frame, uploaded_file.name, crop_toggle, conf_threshold)

with tab_webcam:
    capture_mode = st.radio("CAPTURE MODE", ["SNAPSHOT", "LIVE"], horizontal=True)

    if capture_mode == "SNAPSHOT":
        webcam_image = st.camera_input("CAPTURE LIVE TARGET...")

        if webcam_image:
            file_bytes = np.asarray(bytearray(webcam_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            process_and_display(frame, "WEBCAM_CAPTURE", crop_toggle, conf_threshold)
    else:
        st.caption("[ CONTINUOUS FEED ] -- Every frame is re-scanned, so the analysis auto-updates as targets enter/leave view.")

        def _video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            annotated_frame, _, _, _ = inference.analyze_frame(
                models, img, conf_threshold, active_age, active_gender, active_emotion, active_drowsiness, active_race
            )
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        webrtc_streamer(
            key="live-drowsiness-feed",
            video_frame_callback=_video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
