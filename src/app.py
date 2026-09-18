import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

# Page setup & Cyberpunk style injection
st.set_page_config(page_title="SYS // AGE_GENDER_DETECTOR", layout="wide")

st.markdown(
    """
    <style>
    /* Main app background and font */
    .stApp {
        background-color: #0d1117;
        color: #00ff66;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff66 !important;
        font-family: 'Courier New', Courier, monospace;
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

st.title("SYS // AGE & GENDER INFERENCE")
st.caption("[ STATUS: ONLINE ] -- Deep Neural Network Image Analysis")

# Resolve model paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"
GENDER_PROTO = MODEL_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


@st.cache_resource
def load_models():
    """Load neural network files into memory."""
    required_files = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing weight/config file: {file_path}")

    face_net = cv2.dnn.readNet(str(FACE_MODEL), str(FACE_PROTO))
    age_net = cv2.dnn.readNet(str(AGE_MODEL), str(AGE_PROTO))
    gender_net = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))
    return face_net, age_net, gender_net


try:
    face_net, age_net, gender_net = load_models()
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


# Sidebar Interface Controls
st.sidebar.markdown("### // CONTROL PANEL")
crop_toggle = st.sidebar.toggle("CROP FACE TARGETS ONLY", value=False)
conf_threshold = st.sidebar.slider("CONFIDENCE THRESHOLD", 0.1, 1.0, 0.7)

# File Upload Dropzone
uploaded_files = st.file_uploader(
    "SELECT OR DROP IMAGE FILES FOR INFERENCE...",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        annotated_frame = frame.copy()

        face_boxes = detect_faces(face_net, frame, conf_threshold)

        if not face_boxes:
            st.warning(f"[TARGET MISSING] Zero targets detected in file: {uploaded_file.name}")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=uploaded_file.name, use_container_width=True)
            continue

        cropped_faces = []

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

            label = f"{gender}, {age}"

            # Draw green box and yellow overlay text
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame.shape[0] / 150)), 8)
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cropped_faces.append((f"TARGET_{idx}: {label}", cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))

        st.markdown(f"#### // ANALYSIS RESULT: `{uploaded_file.name}`")

        # Toggle Display Output Mode
        if crop_toggle:
            cols = st.columns(min(len(cropped_faces), 4))
            for idx, (label, crop_img) in enumerate(cropped_faces):
                with cols[idx % 4]:
                    st.image(crop_img, caption=label, use_container_width=True)
        else:
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)