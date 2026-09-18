import base64
import threading
import time
from collections import deque
import csv
import io
import json
import re
from html import escape

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

import inference

LIVE_METRICS = deque(maxlen=120)
LIVE_METRICS_LOCK = threading.Lock()

# Page setup and visual system
st.set_page_config(page_title="MULTIMODAL_FACE_ANALYZER", layout="wide")

theme = st.sidebar.selectbox("THEME", ["Dark cyberpunk", "Light cyberpunk"], key="theme")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --base: #0b1217;
        --surface: #121f26;
        --surface-raised: #192a31;
        --line: #2a4248;
        --text: #e8f2ef;
        --muted: #a3bdb9;
        --accent: #76dfb1;
        --alert: #ff8d83;
    }

    body:has(.light-theme) {
        --base: #f3f8f6;
        --surface: #ffffff;
        --surface-raised: #e4f0eb;
        --line: #a9c5ba;
        --text: #16312a;
        --muted: #4f6d63;
        --accent: #087a52;
        --alert: #b42318;
    }

    .stApp {
        background: radial-gradient(circle at 85% 0%, #17332f 0, var(--base) 34rem);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
        letter-spacing: -0.025em;
    }
    .stApp p, .stApp label, .stApp span { color: var(--text); }
    .stApp [data-testid="stCaptionContainer"] p { color: var(--muted); }
    .app-hero {
        border-left: 3px solid var(--accent);
        padding: 0.2rem 0 0.25rem 1.5rem;
        margin: 0 0 2.2rem;
    }
    .app-hero h1 {
        font-size: clamp(2.1rem, 4vw, 3.5rem);
        line-height: 1.08;
        margin: 0 0 0.65rem;
        font-weight: 700;
    }
    .app-hero p { color: var(--muted); margin: 0; font-size: 1.05rem; }
    section[data-testid="stSidebar"] {
        background: #101c22;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--accent);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        margin-top: 1.7rem;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p { line-height: 1.4; }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.75rem;
        border-bottom: 1px solid var(--line);
    }
    div[data-testid="stTabs"] button[role="tab"] {
        color: var(--muted);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--accent);
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #4e8174;
        border-radius: 12px;
        background: var(--surface);
        padding: 1rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--surface);
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border: 1px solid #467a6c;
        border-radius: 8px;
        background: var(--surface-raised);
        color: var(--text);
        font-weight: 600;
        transition: background 120ms ease, border-color 120ms ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: #244339;
        border-color: var(--accent);
        color: #fff;
    }
    div[data-testid="stButton"] > button:focus-visible,
    div[data-testid="stDownloadButton"] > button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--line);
    }
    .target-card {
        border: 1px solid var(--line);
        border-top: 2px solid var(--accent);
        border-radius: 10px;
        background: var(--surface);
        padding: 1.1rem 1.25rem;
        margin: 0.75rem 0 1.25rem;
    }
    .target-card-id {
        color: var(--accent);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        margin-bottom: 0.7rem;
    }
    .target-card-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.4rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.88rem;
    }
    .target-card-row:last-child { border-bottom: none; }
    .target-card-row .k { color: var(--muted); }
    .target-card-row .v { color: var(--text); text-align: right; overflow-wrap: anywhere; }
    .target-card-status-alert { color: var(--accent) !important; }
    .target-card-status-drowsy { color: var(--alert) !important; }
    .face-hover-image {
        position: relative;
        width: 100%;
        margin-bottom: 1rem;
    }
    .face-hover-image > img {
        display: block;
        width: 100%;
        height: 100%;
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    .face-hover-target {
        position: absolute;
        z-index: 1;
        cursor: help;
        border-radius: 4px;
    }
    .face-hover-target:hover,
    .face-hover-target:focus-visible {
        z-index: 3;
        outline: 2px solid var(--accent);
        background: rgba(118, 223, 177, 0.12);
    }
    .face-hover-info {
        display: none;
        position: absolute;
        top: calc(100% + 0.5rem);
        left: 0;
        width: min(18rem, 75vw);
        z-index: 4;
    }
    .face-hover-target.place-right .face-hover-info { left: auto; right: 0; }
    .face-hover-target.place-up .face-hover-info { top: auto; bottom: calc(100% + 0.5rem); }
    .face-hover-target:hover .face-hover-info,
    .face-hover-target:focus .face-hover-info { display: block; }
    .face-hover-info .target-card {
        margin: 0;
        padding: 0.8rem;
        max-height: min(70vh, 22rem);
        overflow: auto;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
    }
    @media (max-width: 640px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        .app-hero { padding-left: 1rem; margin-bottom: 1.5rem; }
        .target-card-row { display: block; }
        .target-card-row .v { display: block; text-align: left; margin-top: 0.15rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button { transition: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-hero"><h1>Multimodal Face Analyzer</h1>'
    '<p>Analyze faces, compare models, and inspect each result.</p></div>',
    unsafe_allow_html=True,
)

load_models = st.cache_resource(inference.load_models)


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

active_face_detector = "yolo" if models.yolo_face_nets else "ssd"
_face_detector_options = (
    (["yolo"] if models.yolo_face_nets else [])
    + ["ssd"]
    + (["scrfd"] if models.scrfd_face_nets else [])
    + (["retinaface"] if models.retinaface_nets else [])
)
if len(_face_detector_options) > 1:
    active_face_detector = st.sidebar.selectbox(
        "FACE DETECTOR", _face_detector_options, index=_face_detector_options.index(active_face_detector),
        help="Exactly one detector runs per frame -- yolo is the default YOLOv8-Face detector when loaded; ssd is the always-available TensorFlow SSD/ResNet-10 fallback; scrfd and retinaface are alternatives.",
    )

active_age = _model_checkboxes("AGE", models.age_nets)
active_gender = _model_checkboxes("GENDER", models.gender_nets)
active_race = _model_checkboxes("RACE", models.race_nets)
active_emotion = _model_checkboxes("EMOTION", models.emotion_nets)
active_drowsiness = _model_checkboxes("DROWSINESS", models.drowsiness_nets)
active_expression = _model_checkboxes("EXPRESSION", models.expression_nets)
active_liveness = _model_checkboxes("LIVENESS", models.liveness_nets)
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
active_gaze = _model_checkboxes("GAZE", models.gaze_nets)
active_body_composition = _model_checkboxes("BMI / BODY-FAT ESTIMATE", models.body_composition_nets)
if models.body_composition_nets:
    st.sidebar.caption("Experimental relative facial-adiposity proxy for research/data triage only; not clinical BMI or body-fat measurement.")


def _reset_adjustments(prefixes: tuple[str, ...]) -> None:
    for state_key in list(st.session_state):
        if not any(state_key.startswith(f"{prefix}_") for prefix in prefixes):
            continue
        for adj_key, (_, _, default) in inference.IMAGE_ADJUSTMENT_RANGES.items():
            if state_key.endswith(f"_{adj_key}"):
                st.session_state[state_key] = default
                break


def _adjustment_sliders(caption: str, key_prefix: str, column_count: int = 2) -> dict:
    st.caption(caption)
    if st.button("Reset these sliders", key=f"{key_prefix}_reset"):
        _reset_adjustments((key_prefix,))
    values = {}
    columns = st.columns(column_count)
    for index, (adj_key, (adj_min, adj_max, adj_default)) in enumerate(inference.IMAGE_ADJUSTMENT_RANGES.items()):
        with columns[index % column_count]:
            values[adj_key] = st.slider(
                adj_key.replace("_", " ").title(), adj_min, adj_max, adj_default, key=f"{key_prefix}_{adj_key}"
            )
    return values


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

enable_crowd_count = st.sidebar.checkbox("CROWD COUNT / DEMOGRAPHICS", value=False, key="crowd_count_enabled")
if enable_crowd_count:
    st.sidebar.caption(
        "Aggregates age/gender/race across every face detected in an image into a total count "
        "plus a breakdown per active model. Off by default -- confirm this complies with local "
        "policy before using it on images of people who haven't consented to aggregate analysis."
    )

if theme == "Light cyberpunk":
    st.markdown('<div class="light-theme"></div>', unsafe_allow_html=True)


def _target_card_html(face: dict) -> str:
    """Render one face's results as a HUD-style dossier card (native markup, not pixel text --
    keeps results legible no matter how many faces are packed into one image)."""
    rows = ""
    for label, values in (
        ("AGE", face["age"]), ("GENDER", face["gender"]), ("RACE", face["race"]), ("MOOD", face["emotion"]),
        ("EXPR", face["expression"]), ("GAZE", face["gaze"]), ("EYE CONTACT", face["eye_contact"]), ("HEAD POSE", face["head_pose"]), ("IDENTITY", face["identity"]), ("FACIAL HAIR", face["facial_hair"]),
        ("SKIN TONE", face["skin_tone"]), ("GLASSES", face["glasses"]), ("MASK", face["mask"]),
        ("HAIR COLOR", face["hair_color"]), ("EYE COLOR", face["eye_color"]),
        ("BMI / BODY-FAT", face["body_composition"]),
        ("LIVENESS", face["liveness"]),
    ):
        if values:
            rows += f'<div class="target-card-row"><span class="k">{label}</span><span class="v">{escape(" / ".join(values))}</span></div>'
    if face["status"] is not None:
        status_class = "target-card-status-drowsy" if face["drowsy"] else "target-card-status-alert"
        dot = "●"
        rows += f'<div class="target-card-row"><span class="k">STATUS</span><span class="v {status_class}">{dot} {escape(face["status"])}</span></div>'
    if not rows:
        rows = '<div class="target-card-row"><span class="k">STATUS</span><span class="v">no model output</span></div>'
    return f'<div class="target-card"><div class="target-card-id">FACE {face["idx"]:02d}</div>{rows}</div>'


def _comparison_rows(face: dict) -> list[dict[str, str]]:
    """Turn the sparse per-model result columns into rows for one face's comparison table."""
    return sorted(face["model_results"], key=lambda row: (row["Feature"], row["Model"]))


def _render_confidence(rows: list[dict[str, str]]) -> None:
    """Render explicit percentage scores only; labels are not confidence."""
    scored = []
    for row in rows:
        match = re.search(r"(\d+(?:\.\d+)?)%", row["Output"])
        if match:
            scored.append((row, min(100.0, float(match.group(1)))))
    if not scored:
        st.caption("Confidence unavailable: active backend returned labels without calibrated scores.")
        return
    for row, score in scored:
        st.progress(score / 100, text=f"{row['Feature']} / {row['Model']}: {score:.0f}%")


def _hoverable_face_image(frame_bgr: np.ndarray, faces: list[dict]) -> str:
    """Render the annotated frame with focusable hover regions over detected boxes."""
    height, width = frame_bgr.shape[:2]
    success, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Could not encode annotated image")
    source = base64.b64encode(encoded).decode("ascii")
    targets = []
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
        y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
        placement = (" place-right" if x1 + x2 > width else "") + (" place-up" if y1 + y2 > height else "")
        style = f"left:{x1 / width * 100:.4f}%;top:{y1 / height * 100:.4f}%;width:{(x2 - x1) / width * 100:.4f}%;height:{(y2 - y1) / height * 100:.4f}%"
        targets.append(
            f'<div class="face-hover-target{placement}" style="{style}" tabindex="0" '
            f'aria-label="Face {face["idx"]}: hover or focus for details">'
            f'<div class="face-hover-info">{_target_card_html(face)}</div></div>'
        )
    return (
        f'<div class="face-hover-image" style="aspect-ratio:{width}/{height}">'
        f'<img src="data:image/jpeg;base64,{source}" alt="Annotated image with detected faces">'
        f'{"".join(targets)}</div>'
    )


def process_and_display(frame: np.ndarray, identifier: str, conf_threshold: float) -> None:
    """Run detection/inference on frame and render result in Streamlit."""
    frame, was_colorized = inference.maybe_colorize(models, frame, active_colorization)

    annotated_frame, cropped_faces, any_drowsy, has_faces, pose_detected, hands_detected = inference.analyze_frame(
        models, frame, conf_threshold, active_age, active_gender, active_emotion, active_drowsiness, active_race, active_expression,
        active_recognition, st.session_state.get("gallery", {}),
        active_facial_hair, active_skin_tone, active_glasses, active_mask, active_hair_color, active_eye_color,
        active_pose, active_face_landmarks, active_hands, active_gaze, global_adjustments, face_adjustments, face_detector=active_face_detector,
        active_liveness=active_liveness,
        active_body_composition=active_body_composition,
    )

    if was_colorized:
        st.caption("[ AUTO-COLORIZED ] -- source detected as grayscale")
    if pose_detected:
        st.caption("[ POSE DETECTED ] -- skeleton overlay drawn")
    if hands_detected:
        st.caption("[ HANDS DETECTED ] -- landmark overlay drawn")

    height, width = frame.shape[:2]
    region_key = f"region_result_{identifier}"
    with st.expander("SELECT REGION & TRANSFORM"):
        x_col, y_col = st.columns(2)
        x1 = x_col.number_input("X1", min_value=0, max_value=width, value=0, key=f"x1_{identifier}")
        x2 = x_col.number_input("X2", min_value=0, max_value=width, value=width, key=f"x2_{identifier}")
        y1 = y_col.number_input("Y1", min_value=0, max_value=height, value=0, key=f"y1_{identifier}")
        y2 = y_col.number_input("Y2", min_value=0, max_value=height, value=height, key=f"y2_{identifier}")
        transform = st.selectbox("TRANSFORM", inference.GEOMETRIC_TRANSFORM_OPTIONS, key=f"transform_{identifier}")
        params = {}
        if transform == "translate":
            params = {"dx": st.number_input("DX", value=0, key=f"dx_{identifier}"),
                      "dy": st.number_input("DY", value=0, key=f"dy_{identifier}")}
        elif transform == "reflect":
            params = {"axis": st.selectbox("AXIS", ["horizontal", "vertical"], key=f"reflect_axis_{identifier}")}
        elif transform == "rotate":
            params = {"angle": st.number_input("ANGLE (DEGREES)", value=0.0, key=f"angle_{identifier}"),
                      "scale": st.number_input("ROTATION SCALE", min_value=0.01, value=1.0, key=f"rotation_scale_{identifier}")}
        elif transform == "scale":
            params = {"fx": st.number_input("X SCALE", min_value=0.01, value=1.0, key=f"fx_{identifier}"),
                      "fy": st.number_input("Y SCALE", min_value=0.01, value=1.0, key=f"fy_{identifier}")}
        elif transform == "shear":
            params = {"axis": st.selectbox("AXIS", ["x", "y"], key=f"shear_axis_{identifier}"),
                      "factor": st.number_input("SHEAR FACTOR", value=0.0, key=f"shear_factor_{identifier}")}
        if st.button("APPLY TRANSFORM", key=f"transform_btn_{identifier}"):
            region = inference.crop_region(frame, x1, y1, x2, y2)
            if region.size == 0:
                st.warning("[ EMPTY REGION ] -- select a rectangle with nonzero width and height")
            else:
                result = inference.apply_geometric_transform(region, transform, **params)
                st.session_state[region_key] = result
        if region_key in st.session_state:
            result = st.session_state[region_key]
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Transformed region")
            st.download_button("DOWNLOAD TRANSFORMED PNG", cv2.imencode(".png", result)[1].tobytes(),
                               file_name="transformed_region.png", mime="image/png", key=f"transform_dl_{identifier}")

    if not has_faces:
        st.warning(f"[TARGET MISSING] Zero targets detected in file: {identifier}")
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=identifier, use_container_width=True)
        return

    st.markdown(f"#### Results for `{identifier}`")

    export_rows = []
    for face in cropped_faces:
        export_rows.append({
            key: value for key, value in face.items()
            if key not in {"image", "embedding", "raw_columns"} and isinstance(value, (str, int, float, bool, list, type(None)))
        })
    export_json = json.dumps(export_rows, indent=2, default=str)
    csv_buffer = io.StringIO()
    if export_rows:
        fieldnames = sorted({key for row in export_rows for key in row})
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()} for row in export_rows)
    export_col_json, export_col_csv = st.columns(2)
    export_col_json.download_button("DOWNLOAD RESULTS JSON", export_json, f"{identifier}_results.json", "application/json", key=f"json_dl_{identifier}")
    export_col_csv.download_button("DOWNLOAD RESULTS CSV", csv_buffer.getvalue(), f"{identifier}_results.csv", "text/csv", key=f"csv_dl_{identifier}")

    if any_drowsy:
        st.error("[ALERT] DROWSINESS DETECTED -- SUBJECT EYES CLOSED")

    if enable_crowd_count:
        with st.expander(f"CROWD COUNT: {len(cropped_faces)} face(s) detected", expanded=False):
            aggregate = inference.aggregate_demographics(cropped_faces)
            if not aggregate:
                st.caption("No age/gender/race model is active -- enable one to see a breakdown.")
            for feature in inference.AGGREGATE_FEATURES:
                for model_key, counts in aggregate.get(feature, {}).items():
                    st.caption(f"{feature.upper()} ({model_key})")
                    st.bar_chart(counts)

    st.caption("Hover or tap a face box to see its details.")
    st.markdown(_hoverable_face_image(annotated_frame, cropped_faces), unsafe_allow_html=True)

    st.markdown("#### Model comparison")
    for face in cropped_faces:
        rows = _comparison_rows(face)
        if rows:
            st.caption(f"Face {face['idx']:02d}")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            _render_confidence(rows)

    if st.button("SCAN ALL FACES: RECOGNIZED / UNRECOGNIZED", key=f"scan_btn_{identifier}"):
        faces_bgr = [cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR) for face in cropped_faces]
        matches = inference.match_faces_eigenfaces_batch(faces_bgr)
        scan_frame = frame.copy()
        inference.draw_recognition_scan(scan_frame, [(face["box"], match is not None) for face, match in zip(cropped_faces, matches)])
        st.image(cv2.cvtColor(scan_frame, cv2.COLOR_BGR2RGB), caption="Recognition scan", use_container_width=True)

        recognized_count = sum(match is not None for match in matches)
        st.caption(f"[ SCAN COMPLETE ] {recognized_count}/{len(matches)} face(s) recognized against saved faces (eigen/)")
        for face, match in zip(cropped_faces, matches):
            if match:
                st.text(f"#{face['idx']}: Recognized -- saved face ID {match[0]} (distance {match[1]:.0f})")
            else:
                st.text(f"#{face['idx']}: Unrecognized")

    cols = st.columns(min(len(cropped_faces), 2))
    for i, face in enumerate(cropped_faces):
        with cols[i % len(cols)]:
            with st.expander(f"Edit face {face['idx']}"):
                individual_adjustments = _adjustment_sliders(
                    "Edit this crop only. Analysis labels use the detected crop.",
                    f"individual_adj_{identifier}_{face['idx']}",
                    column_count=1,
                )
                face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                edited_face_bgr = (
                    inference.apply_image_adjustments(face_bgr, individual_adjustments)
                    if any(individual_adjustments.values()) else face_bgr
                )
                st.download_button(
                    "DOWNLOAD EDITED FACE PNG", cv2.imencode(".png", edited_face_bgr)[1].tobytes(),
                    file_name=f"face_{face['idx']}_edited.png", mime="image/png",
                    key=f"face_edit_dl_{identifier}_{face['idx']}",
                )

                op_key = f"image_op_result_{identifier}_{face['idx']}"
                op = st.selectbox("IMAGE OP", inference.IMAGE_OP_OPTIONS, key=f"image_op_{identifier}_{face['idx']}")
                op_params = {}
                if op == "intensity":
                    op_params["method"] = st.selectbox("INTENSITY METHOD", inference.INTENSITY_METHODS,
                                                        key=f"intensity_method_{identifier}_{face['idx']}")
                elif op == "sharpen":
                    op_params["method"] = st.selectbox("SHARPEN METHOD", inference.SHARPEN_METHODS,
                                                        key=f"sharpen_method_{identifier}_{face['idx']}")
                elif op == "denoise":
                    op_params["method"] = st.selectbox("DENOISE METHOD", inference.DENOISE_METHODS,
                                                        key=f"denoise_method_{identifier}_{face['idx']}")
                if st.button("APPLY IMAGE OP", key=f"image_op_btn_{identifier}_{face['idx']}"):
                    st.session_state[op_key] = inference.apply_image_op(edited_face_bgr, op, **op_params)
                if op_key in st.session_state:
                    result = st.session_state[op_key]
                    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Processed face")
                    st.download_button("DOWNLOAD FACE PNG", cv2.imencode(".png", result)[1].tobytes(),
                                       file_name=f"face_{face['idx']}_processed.png", mime="image/png",
                                       key=f"image_op_dl_{identifier}_{face['idx']}")
            st.image(cv2.cvtColor(edited_face_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(_target_card_html(face), unsafe_allow_html=True)
            lbph_available = models.recognition_nets.get("lbph") is not None
            if face["embedding"] is not None or lbph_available:
                enroll_name = st.text_input("ENROLL AS", key=f"enroll_name_{identifier}_{face['idx']}", label_visibility="collapsed", placeholder="ENROLL AS...")
                if st.button("ENROLL", key=f"enroll_btn_{identifier}_{face['idx']}") and enroll_name:
                    try:
                        safe_name = inference.validate_lbph_name(enroll_name) if lbph_available else enroll_name.strip()
                        if not safe_name:
                            raise ValueError("Enrollment name cannot be empty.")
                        if face["embedding"] is not None:
                            st.session_state["gallery"][safe_name] = np.array(face["embedding"], dtype=np.float32)
                            inference.save_gallery(st.session_state["gallery"])
                        if lbph_available:
                            inference.enroll_lbph_face(safe_name, cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR))
                        st.rerun()
                    except ValueError as exc:
                        st.error(f"[INVALID ENROLLMENT] {exc}")

            col_search, col_save = st.columns(2)
            if col_search.button("SEARCH", key=f"search_btn_{identifier}_{face['idx']}"):
                face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                found = False
                if face["embedding"] is not None and search_gallery:
                    match = inference.match_face_identity(np.array(face["embedding"], dtype=np.float32), search_gallery)
                    if match:
                        st.success(f"[ PHOTO MATCH ] {match[0]} ({match[1] * 100:.0f}%)")
                        found = True
                eigen_match = inference.match_face_eigenfaces(face_bgr)
                if eigen_match:
                    st.success(f"[ EIGENFACE MATCH ] saved face ID {eigen_match[0]} (distance {eigen_match[1]:.0f})")
                    found = True
                if not found:
                    st.warning("[ NO MATCH ] -- no known/saved face matched")

            if col_save.button("SAVE", key=f"save_btn_{identifier}_{face['idx']}"):
                face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                saved_id = inference.save_face(face_bgr, face["raw_columns"])
                st.info(f"[ SAVED ] ID {saved_id}")

            if models.reconstruction_3d_nets:
                if st.button("3D RECON", key=f"recon3d_btn_{identifier}_{face['idx']}"):
                    face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                    result = inference.run_3d_reconstruction(models, face_bgr)
                    if result is None:
                        st.warning("[ NO RECONSTRUCTION ] -- no face landmarks found in this crop")
                    else:
                        vertices, faces, colors = result
                        obj_text = inference.mesh_to_obj_str(vertices, faces, colors)
                        st.download_button(
                            "DOWNLOAD .OBJ", data=obj_text, file_name=f"face_{identifier}_{face['idx']}.obj",
                            mime="text/plain", key=f"recon3d_dl_{identifier}_{face['idx']}",
                        )

            if models.age_progression_nets:
                st.caption("AGE PROGRESSION (non-commercial use only -- see README)")
                col_src_age, col_tgt_age = st.columns(2)
                source_age = col_src_age.number_input(
                    "Source age", min_value=0, max_value=100, value=30,
                    key=f"reage_src_{identifier}_{face['idx']}",
                )
                target_age = col_tgt_age.number_input(
                    "Target age", min_value=0, max_value=100, value=60,
                    key=f"reage_tgt_{identifier}_{face['idx']}",
                )
                if st.button("AGE PROGRESSION", key=f"reage_btn_{identifier}_{face['idx']}"):
                    face_bgr = cv2.cvtColor(face["image"], cv2.COLOR_RGB2BGR)
                    aged_bgr = inference.run_age_progression(models, face_bgr, source_age, target_age)
                    st.session_state[f"reage_result_{identifier}_{face['idx']}"] = aged_bgr
                result_key = f"reage_result_{identifier}_{face['idx']}"
                if result_key in st.session_state:
                    aged_bgr = st.session_state[result_key]
                    col_before, col_after = st.columns(2)
                    col_before.image(face["image"], caption="Before")
                    col_after.image(cv2.cvtColor(aged_bgr, cv2.COLOR_BGR2RGB), caption="After")
                    st.download_button(
                        "DOWNLOAD AGED PNG", cv2.imencode(".png", aged_bgr)[1].tobytes(),
                        file_name=f"face_{identifier}_{face['idx']}_aged.png", mime="image/png",
                        key=f"reage_dl_{identifier}_{face['idx']}",
                    )


with st.expander("Image editing", expanded=True):
    if st.button("Reset all adjustments", key="reset_all_adjustments"):
        _reset_adjustments(("global_adj", "face_adj", "individual_adj"))
    whole_image_tab, each_face_tab = st.tabs(["Whole image", "Each face"])
    with whole_image_tab:
        global_adjustments = _adjustment_sliders(
            "Adjust the full image before face detection.", "global_adj"
        )
    with each_face_tab:
        face_adjustments = _adjustment_sliders(
            "Adjust each detected face before classification.", "face_adj"
        )

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
                frame = inference.decode_image_bytes(uploaded_file.read())
            except ValueError as exc:
                st.error(f"[INVALID IMAGE] {uploaded_file.name}: {exc}")
                continue
            process_and_display(frame, uploaded_file.name, conf_threshold)

with tab_webcam:
    capture_mode = st.radio("Capture mode", ["SNAPSHOT", "LIVE"], horizontal=True, format_func=str.title)

    if capture_mode == "SNAPSHOT":
        webcam_image = st.camera_input("Take a snapshot")

        if webcam_image:
            try:
                frame = inference.decode_image_bytes(webcam_image.read())
            except ValueError as exc:
                st.error(f"[INVALID IMAGE] WEBCAM_CAPTURE: {exc}")
                frame = None
            if frame is not None:
                process_and_display(frame, "WEBCAM_CAPTURE", conf_threshold)
    else:
        st.caption("Live analysis updates as people enter or leave view. Each face keeps a stable ID as it moves.")
        frame_skip = st.slider(
            "CLASSIFIER FRAME SKIP", 1, 10, 1,
            help="Run age/gender/emotion/race/recognition/etc. classifiers every Nth frame "
            "instead of every frame. Face detection and the pose/hand/face-landmark overlays "
            "still run every frame, so the video stays smooth. These classifiers' outputs "
            "aren't otherwise drawn onto the LIVE video (see target cards in Image upload / "
            "SNAPSHOT for that), so skipping them here only reduces CPU load, with no visible "
            "staleness to interpolate around.",
        )
        frame_counter = {"n": 0}
        _NO_MODELS: set = set()
        face_tracker = _get_face_tracker()
        liveness_tracker = _get_liveness_tracker()
        reset_col, voice_col = st.columns([1, 2])
        if reset_col.button("RESET TRACKING IDS", key="reset_tracking_ids"):
            face_tracker.reset()
            liveness_tracker.reset()

        # #10: off by default -- requesting the microphone is a permission prompt the user
        # didn't ask for just by opening the webcam tab, so it needs its own explicit opt-in
        # rather than riding along with LIVE mode's existing camera request.
        enable_voice_fusion = voice_col.checkbox(
            "Enable voice+face fusion (uses microphone)", value=False, key="enable_voice_fusion",
            help="Heuristic only: cross-checks mic loudness against the largest face's emotion "
                 "label. Not a trained speech-emotion model -- see src/inference.py's "
                 "VoiceFaceFusion docstring for why.",
        )
        voice_fusion = _get_voice_fusion() if enable_voice_fusion else None
        if voice_fusion is not None and voice_col.button("RESET VOICE BUFFER", key="reset_voice_buffer"):
            voice_fusion.reset()

        def _video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            frame_started = time.perf_counter()
            metrics = {}
            img = frame.to_ndarray(format="bgr24")
            img, _ = inference.maybe_colorize(models, img, active_colorization)
            frame_counter["n"] += 1
            run_classifiers = frame_counter["n"] % frame_skip == 0
            annotated_frame, cropped_faces, _, _, _, _ = inference.analyze_frame(
                models, img, conf_threshold,
                active_age if run_classifiers else _NO_MODELS,
                active_gender if run_classifiers else _NO_MODELS,
                active_emotion if run_classifiers else _NO_MODELS,
                active_drowsiness if run_classifiers else _NO_MODELS,
                active_race if run_classifiers else _NO_MODELS,
                active_expression if run_classifiers else _NO_MODELS,
                active_recognition if run_classifiers else _NO_MODELS, dict(st.session_state.get("gallery", {})),
                active_facial_hair if run_classifiers else _NO_MODELS,
                active_skin_tone if run_classifiers else _NO_MODELS,
                active_glasses if run_classifiers else _NO_MODELS,
                active_mask if run_classifiers else _NO_MODELS,
                active_hair_color if run_classifiers else _NO_MODELS,
                active_eye_color if run_classifiers else _NO_MODELS,
                active_pose, active_face_landmarks, active_hands,
                active_gaze if run_classifiers else _NO_MODELS,
                global_adjustments, face_adjustments,
                face_detector=active_face_detector, metrics=metrics, tracker=face_tracker,
                liveness_tracker=liveness_tracker,
                active_liveness=active_liveness,
                active_body_composition=active_body_composition if run_classifiers else _NO_MODELS,
            )
            metrics["frame_ms"] = (time.perf_counter() - frame_started) * 1000
            metrics["timestamp"] = time.monotonic()
            with LIVE_METRICS_LOCK:
                LIVE_METRICS.append(metrics)
            if voice_fusion is not None and cropped_faces:
                # v1 scope (matches #9's own "largest face only" precedent): fuse against the
                # single largest detected face, not a per-face history -- multi-face voice
                # attribution would need knowing WHICH face is speaking, which this app has no
                # signal for (that's a lip-sync/diarization problem, out of scope here). If this
                # is a skipped (non-classifier) frame, emotion is simply absent this frame --
                # same "no interpolation" tradeoff CLASSIFIER FRAME SKIP already documents.
                largest = max(cropped_faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
                emotion_label = largest["emotion"][0] if largest["emotion"] else None
                voice_arousal = voice_fusion.current_arousal()
                voice_fusion.set_latest_status({
                    "voice_arousal": voice_arousal,
                    "emotion": emotion_label,
                    "consistency": inference.fuse_voice_and_emotion(voice_arousal, emotion_label) if emotion_label else None,
                })
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        def _audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
            if voice_fusion is not None:
                samples = inference.audio_frame_to_mono_float(frame.to_ndarray())
                voice_fusion.ingest_audio(samples, frame.sample_rate)
            return frame

        webrtc_streamer(
            key="live-drowsiness-feed",
            video_frame_callback=_video_frame_callback,
            audio_frame_callback=_audio_frame_callback if enable_voice_fusion else None,
            media_stream_constraints={"video": True, "audio": enable_voice_fusion},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

        if voice_fusion is not None:
            status = voice_fusion.get_latest_status()
            if status is None:
                st.caption("[ VOICE FUSION ] waiting for audio + a detected face...")
            else:
                consistency_text = status["consistency"] or "n/a (emotion label not categorized)"
                st.caption(
                    f"[ VOICE FUSION ] mic: {status['voice_arousal']} | largest face emotion: "
                    f"{status['emotion']} | {consistency_text}"
                )

        with LIVE_METRICS_LOCK:
            live_metrics = list(LIVE_METRICS)
        if live_metrics:
            intervals = np.diff([item["timestamp"] for item in live_metrics[-30:]])
            fps = 1.0 / float(np.mean(intervals)) if len(intervals) and np.mean(intervals) > 0 else 0.0
            st.metric("LIVE FPS", f"{fps:.1f}")
            latency_rows = []
            for item in live_metrics:
                for model_name, values in item.get("model_latency_ms", {}).items():
                    latency_rows.extend({"Model": model_name, "Latency (ms)": value} for value in values)
            if latency_rows:
                latency_frame = pd.DataFrame(latency_rows)
                summary = latency_frame.groupby("Model", as_index=False)["Latency (ms)"].mean()
                summary["Latency (ms)"] = summary["Latency (ms)"].round(1)
                st.dataframe(summary, hide_index=True, use_container_width=True)
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
