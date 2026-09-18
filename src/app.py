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

    /* Target dossier cards -- one per detected face, HUD-style corner brackets
       instead of a generic rounded shadow card. */
    .target-card {
        position: relative;
        border: 1px solid #1f6b3d;
        background-color: #0a0f0a;
        padding: 14px 16px 12px;
        margin-bottom: 18px;
    }
    .target-card::before, .target-card::after,
    .target-card .corner-br, .target-card .corner-bl {
        content: "";
        position: absolute;
        width: 14px;
        height: 14px;
        border-color: #00ff66;
        border-style: solid;
    }
    .target-card::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
    .target-card::after { top: -1px; right: -1px; border-width: 2px 2px 0 0; }
    .target-card .corner-bl { bottom: -1px; left: -1px; border-width: 0 0 2px 2px; }
    .target-card .corner-br { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }
    .target-card-id {
        color: #6bd68f;
        font-size: 0.75rem;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .target-card-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        padding: 3px 0;
        border-bottom: 1px dashed #1f2d1f;
        font-size: 0.9rem;
    }
    .target-card-row:last-child { border-bottom: none; }
    .target-card-row .k { color: #4d9c6b; }
    .target-card-row .v { color: #eafff0; text-align: right; }
    .target-card-status-alert { color: #00ff66; }
    .target-card-status-drowsy { color: #ff4d4d; }
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
active_expression = _model_checkboxes("EXPRESSION", models.expression_nets)
active_recognition = _model_checkboxes("RECOGNITION", models.recognition_nets)
active_facial_hair = _model_checkboxes("FACIAL HAIR", models.facial_hair_nets)
active_skin_tone = _model_checkboxes("SKIN TONE", models.skin_tone_nets)
active_glasses = _model_checkboxes("GLASSES", models.glasses_nets)
active_mask = _model_checkboxes("MASK", models.mask_nets)
active_hair_color = _model_checkboxes("HAIR COLOR", models.hair_color_nets)
active_eye_color = _model_checkboxes("EYE COLOR", models.eye_color_nets)
active_colorization = _model_checkboxes("AUTO-COLORIZE B&W", models.colorization_nets)
active_pose = _model_checkboxes("POSE ESTIMATION", models.pose_nets)
active_face_landmarks = _model_checkboxes("FACE LANDMARKS", models.face_landmarks_nets)
active_hands = _model_checkboxes("HAND LANDMARKS", models.hand_nets)


def _adjustment_sliders(section_label: str, caption: str, key_prefix: str) -> dict:
    st.sidebar.markdown(f"### {section_label}")
    st.sidebar.caption(caption)
    values = {}
    for adj_key, (adj_min, adj_max, adj_default) in inference.IMAGE_ADJUSTMENT_RANGES.items():
        values[adj_key] = st.sidebar.slider(
            adj_key.replace("_", " ").upper(), adj_min, adj_max, adj_default, key=f"{key_prefix}_{adj_key}"
        )
    return values


global_adjustments = _adjustment_sliders(
    "GLOBAL IMAGE ADJUSTMENTS",
    "Applied to the whole image first, before face detection. Visible in every output.",
    "global_adj",
)
face_adjustments = _adjustment_sliders(
    "PER-FACE IMAGE ADJUSTMENTS",
    "Applied to each detected face individually, after detection, before classification.",
    "face_adj",
)

st.session_state.setdefault("gallery", inference.load_gallery())

search_gallery = {}
if models.recognition_nets:
    st.sidebar.markdown("### GALLERY")
    gallery = st.session_state["gallery"]
    if not gallery:
        st.sidebar.caption("[ EMPTY ] -- no enrolled identities")
    for name in list(gallery):
        col_name, col_del = st.sidebar.columns([3, 1])
        col_name.text(name)
        if col_del.button("X", key=f"del_gallery_{name}"):
            del st.session_state["gallery"][name]
            inference.save_gallery(st.session_state["gallery"])
            st.rerun()

    st.sidebar.markdown("### IDENTITY SEARCH")
    st.sidebar.caption("Local directory matching only -- no live internet search.")
    custom_search_dir = st.sidebar.text_input(
        "SEARCH DIRECTORY (optional)", value="", placeholder="/path/to/reference/photos",
        help="Extra directory of named reference photos to search, in addition to the bundled known_people/.",
    )

    @st.cache_resource
    def _load_known_people_gallery():
        net = models.recognition_nets.get("vggface")
        return inference.build_gallery_from_directory(models.face_net, net, inference.KNOWN_PEOPLE_DIR) if net else {}

    search_gallery = dict(_load_known_people_gallery())
    if custom_search_dir:
        net = models.recognition_nets.get("vggface")
        if net is not None:
            search_gallery.update(inference.build_gallery_from_directory(models.face_net, net, custom_search_dir))

# Sidebar Interface Controls
st.sidebar.markdown("### CONTROL PANEL")
conf_threshold = st.sidebar.slider("CONFIDENCE THRESHOLD", 0.1, 1.0, 0.7)


def _target_card_html(face: dict) -> str:
    """Render one face's results as a HUD-style dossier card (native markup, not pixel text --
    keeps results legible no matter how many faces are packed into one image)."""
    rows = ""
    for label, values in (
        ("AGE", face["age"]), ("GENDER", face["gender"]), ("RACE", face["race"]), ("MOOD", face["emotion"]),
        ("EXPR", face["expression"]), ("IDENTITY", face["identity"]), ("FACIAL HAIR", face["facial_hair"]),
        ("SKIN TONE", face["skin_tone"]), ("GLASSES", face["glasses"]), ("MASK", face["mask"]),
        ("HAIR COLOR", face["hair_color"]), ("EYE COLOR", face["eye_color"]),
    ):
        if values:
            rows += f'<div class="target-card-row"><span class="k">{label}</span><span class="v">{" / ".join(values)}</span></div>'
    if face["status"] is not None:
        status_class = "target-card-status-drowsy" if face["drowsy"] else "target-card-status-alert"
        dot = "●"
        rows += f'<div class="target-card-row"><span class="k">STATUS</span><span class="v {status_class}">{dot} {face["status"]}</span></div>'
    if not rows:
        rows = '<div class="target-card-row"><span class="k">STATUS</span><span class="v">no model output</span></div>'
    return f'<div class="target-card"><div class="target-card-id">TARGET_{face["idx"]:02d}</div>{rows}</div>'


def process_and_display(frame: np.ndarray, identifier: str, conf_threshold: float) -> None:
    """Run detection/inference on frame and render result in Streamlit."""
    frame, was_colorized = inference.maybe_colorize(models, frame, active_colorization)

    annotated_frame, cropped_faces, any_drowsy, has_faces, pose_detected, hands_detected = inference.analyze_frame(
        models, frame, conf_threshold, active_age, active_gender, active_emotion, active_drowsiness, active_race, active_expression,
        active_recognition, st.session_state.get("gallery", {}),
        active_facial_hair, active_skin_tone, active_glasses, active_mask, active_hair_color, active_eye_color,
        active_pose, active_face_landmarks, active_hands, global_adjustments, face_adjustments,
    )

    if was_colorized:
        st.caption("[ AUTO-COLORIZED ] -- source detected as grayscale")
    if pose_detected:
        st.caption("[ POSE DETECTED ] -- skeleton overlay drawn")
    if hands_detected:
        st.caption("[ HANDS DETECTED ] -- landmark overlay drawn")

    if not has_faces:
        st.warning(f"[TARGET MISSING] Zero targets detected in file: {identifier}")
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=identifier, use_container_width=True)
        return

    st.markdown(f"#### ANALYSIS RESULT: `{identifier}`")

    if any_drowsy:
        st.error("[ALERT] DROWSINESS DETECTED -- SUBJECT EYES CLOSED")

    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cols = st.columns(min(len(cropped_faces), 4))
    for i, face in enumerate(cropped_faces):
        with cols[i % 4]:
            st.image(face["image"], use_container_width=True)
            st.markdown(_target_card_html(face), unsafe_allow_html=True)
            if face["embedding"] is not None:
                enroll_name = st.text_input("ENROLL AS", key=f"enroll_name_{identifier}_{face['idx']}", label_visibility="collapsed", placeholder="ENROLL AS...")
                if st.button("ENROLL", key=f"enroll_btn_{identifier}_{face['idx']}") and enroll_name:
                    st.session_state["gallery"][enroll_name] = np.array(face["embedding"], dtype=np.float32)
                    inference.save_gallery(st.session_state["gallery"])
                    st.rerun()

                if st.button("SEARCH", key=f"search_btn_{identifier}_{face['idx']}"):
                    if not search_gallery:
                        st.warning("[ NO MATCH ] -- search gallery is empty (no bundled or directory photos with a detected face)")
                    else:
                        match = inference.match_face_identity(np.array(face["embedding"], dtype=np.float32), search_gallery)
                        if match:
                            st.success(f"[ MATCH ] {match[0]} ({match[1] * 100:.0f}%)")
                        else:
                            st.warning("[ NO MATCH ] -- no known face cleared the similarity threshold")


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
            process_and_display(frame, uploaded_file.name, conf_threshold)

with tab_webcam:
    capture_mode = st.radio("CAPTURE MODE", ["SNAPSHOT", "LIVE"], horizontal=True)

    if capture_mode == "SNAPSHOT":
        webcam_image = st.camera_input("CAPTURE LIVE TARGET...")

        if webcam_image:
            file_bytes = np.asarray(bytearray(webcam_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            process_and_display(frame, "WEBCAM_CAPTURE", conf_threshold)
    else:
        st.caption("[ CONTINUOUS FEED ] -- Every frame is re-scanned, so the analysis auto-updates as targets enter/leave view.")

        def _video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img, _ = inference.maybe_colorize(models, img, active_colorization)
            annotated_frame, _, _, _, _, _ = inference.analyze_frame(
                models, img, conf_threshold, active_age, active_gender, active_emotion, active_drowsiness, active_race, active_expression,
                active_recognition, st.session_state.get("gallery", {}),
                active_facial_hair, active_skin_tone, active_glasses, active_mask, active_hair_color, active_eye_color,
                active_pose, active_face_landmarks, active_hands, global_adjustments, face_adjustments,
            )
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        webrtc_streamer(
            key="live-drowsiness-feed",
            video_frame_callback=_video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
