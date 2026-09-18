import os
import sys
from pathlib import Path
import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer

# Page setup & surveillance-terminal style injection
st.set_page_config(page_title="AGE_GENDER_DETECTOR", layout="wide")

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

st.title("AGE & GENDER INFERENCE")
st.caption("[ STATUS: ONLINE ] -- Deep Neural Network Image Analysis")

# Resolve model paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))
from dan_model import DAN

FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"
GENDER_PROTO = MODEL_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"
EYE_CASCADE_FILE = MODEL_DIR / "haarcascade_eye.xml"
EMOTION_MODEL = MODEL_DIR / "dan_affecnet7.pth"

MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
EMOTION_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
EMOTION_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_EYES_OPEN = 2


@st.cache_resource
def load_models():
    """Load neural network files into memory."""
    required_files = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL, EYE_CASCADE_FILE, EMOTION_MODEL]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing weight/config file: {file_path}")

    face_net = cv2.dnn.readNet(str(FACE_MODEL), str(FACE_PROTO))
    age_net = cv2.dnn.readNet(str(AGE_MODEL), str(AGE_PROTO))
    gender_net = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))
    eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_FILE))

    emotion_net = DAN(num_class=7, num_head=4, pretrained=False)
    checkpoint = torch.load(str(EMOTION_MODEL), map_location="cpu")
    emotion_net.load_state_dict(checkpoint["model_state_dict"])
    emotion_net.eval()

    return face_net, age_net, gender_net, eye_cascade, emotion_net


try:
    face_net, age_net, gender_net, eye_cascade, emotion_net = load_models()
except Exception as e:
    st.error(f"[SYSTEM ERROR] Failed to load models: {e}")
    st.stop()


def detect_faces(net: cv2.dnn.Net, frame: np.ndarray, conf_threshold: float = 0.7) -> list[list[int]]:
    """Detect faces and return bounding box limits."""
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes


def predict_gender(blob: np.ndarray) -> str:
    gender_net.setInput(blob)
    return GENDER_LIST[gender_net.forward()[0].argmax()]


def predict_age(blob: np.ndarray) -> str:
    age_net.setInput(blob)
    return AGE_LIST[age_net.forward()[0].argmax()]


def predict_emotion(face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - EMOTION_MEAN) / EMOTION_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits, _, _ = emotion_net(tensor)
    return EMOTION_LABELS[logits[0].argmax().item()]


def _draw_outlined_text(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Draw text with a black outline so it stays readable over any background. Clamps origin so text stays inside the frame."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    frame_h, frame_w = frame.shape[:2]

    x, y = org
    x = max(0, min(x, frame_w - text_w))
    y = max(text_h, min(y, frame_h - baseline))
    org = (x, y)

    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def detect_drowsiness(face_bgr: np.ndarray) -> bool:
    """Return True if fewer than MIN_EYES_OPEN eyes are visible (eyes likely closed)."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    return len(eyes) < MIN_EYES_OPEN


# Sidebar Interface Controls
st.sidebar.markdown("### CONTROL PANEL")
crop_toggle = st.sidebar.toggle("CROP FACE TARGETS ONLY", value=False)
conf_threshold = st.sidebar.slider("CONFIDENCE THRESHOLD", 0.1, 1.0, 0.7)

def analyze_frame(frame: np.ndarray, conf_threshold: float):
    """Detect faces and run age/gender/drowsiness inference. No Streamlit calls (safe for background threads)."""
    annotated_frame = frame.copy()
    face_boxes = detect_faces(face_net, frame, conf_threshold)
    cropped_faces = []
    any_drowsy = False

    for idx, (x1, y1, x2, y2) in enumerate(face_boxes, 1):
        y1_crop = max(0, y1 - 20)
        y2_crop = min(y2 + 20, frame.shape[0] - 1)
        x1_crop = max(0, x1 - 20)
        x2_crop = min(x2 + 20, frame.shape[1] - 1)

        face = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender = predict_gender(blob)
        age = predict_age(blob)
        emotion = predict_emotion(face)
        drowsy = detect_drowsiness(face)
        any_drowsy = any_drowsy or drowsy

        label = f"{gender}, {age}, {emotion}"
        status_label = "DROWSY" if drowsy else "ALERT"
        status_color = (0, 0, 255) if drowsy else (0, 255, 0)

        # Draw green box and cyan overlay text with black outline for readability
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame.shape[0] / 150)), 8)
        _draw_outlined_text(annotated_frame, label, (x1, y1 - 10), (0, 255, 255))
        _draw_outlined_text(annotated_frame, status_label, (x1, y1 + (y2 - y1) + 30), status_color)

        cropped_faces.append((f"TARGET_{idx}: {label} | {status_label}", cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))

    return annotated_frame, cropped_faces, any_drowsy, bool(face_boxes)


def process_and_display(frame: np.ndarray, identifier: str, crop_toggle: bool, conf_threshold: float) -> None:
    """Run detection/inference on frame and render result in Streamlit."""
    annotated_frame, cropped_faces, any_drowsy, has_faces = analyze_frame(frame, conf_threshold)

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
            annotated_frame, _, _, _ = analyze_frame(img, conf_threshold)
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        webrtc_streamer(
            key="live-drowsiness-feed",
            video_frame_callback=_video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )