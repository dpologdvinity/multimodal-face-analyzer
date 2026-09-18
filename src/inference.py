"""Model loading and per-face prediction logic for the Streamlit app (src/app.py).

Deliberately not shared with detect.py -- the CLI and the app duplicate the
pipeline on purpose (see CLAUDE.md); this module only exists to keep app.py
itself from growing unbounded as more model backends are added.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from nets.dan_model import DAN
    from nets.ssrnet_model import SSRNet
    TORCH_SUPPORTED = True
except ImportError:
    TORCH_SUPPORTED = False

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"
GENDER_PROTO = MODEL_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"
EYE_CASCADE_FILE = MODEL_DIR / "haarcascade_eye.xml"
EMOTION_MODEL = MODEL_DIR / "dan_affecnet7.pth"
SSRNET_MODEL = MODEL_DIR / "ssrnet_morph2.pth"

MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
EMOTION_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
EMOTION_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SSRNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SSRNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AGE_BACKENDS = ("caffe", "ssrnet")
MIN_EYES_OPEN = 2


@dataclass
class Models:
    face_net: cv2.dnn.Net
    age_net: object
    age_backend: str
    gender_net: object
    eye_cascade: object
    emotion_net: object

    @property
    def offline_features(self) -> list[str]:
        return [
            name for name, net in [
                ("AGE", self.age_net), ("GENDER", self.gender_net),
                ("DROWSINESS", self.eye_cascade), ("EMOTION", self.emotion_net),
            ] if net is None
        ]


def load_models(age_backend: str) -> Models:
    """Load neural network files into memory. Face detection is required; age, gender, drowsiness,
    and emotion are each optional and skipped (returned as None) if their model file(s) aren't
    present, so the app degrades gracefully to whichever features were built in. age_backend
    selects which age model to load: "caffe" (bucketed) or "ssrnet" (continuous)."""
    if not FACE_PROTO.exists() or not FACE_MODEL.exists():
        raise FileNotFoundError(f"Missing face detector file(s) in {MODEL_DIR}: {FACE_PROTO.name}, {FACE_MODEL.name} (required).")
    face_net = cv2.dnn.readNet(str(FACE_MODEL), str(FACE_PROTO))

    age_net = None
    if age_backend == "caffe":
        if AGE_PROTO.exists() and AGE_MODEL.exists():
            age_net = cv2.dnn.readNet(str(AGE_MODEL), str(AGE_PROTO))
    elif age_backend == "ssrnet":
        if TORCH_SUPPORTED and SSRNET_MODEL.exists():
            age_net = SSRNet()
            checkpoint = torch.load(str(SSRNET_MODEL), map_location="cpu")
            age_net.load_state_dict(checkpoint["state_dict"])
            age_net.eval()

    gender_net = None
    if GENDER_PROTO.exists() and GENDER_MODEL.exists():
        gender_net = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))

    eye_cascade = None
    if EYE_CASCADE_FILE.exists():
        eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_FILE))

    emotion_net = None
    if TORCH_SUPPORTED and EMOTION_MODEL.exists():
        emotion_net = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint = torch.load(str(EMOTION_MODEL), map_location="cpu")
        emotion_net.load_state_dict(checkpoint["model_state_dict"])
        emotion_net.eval()

    return Models(face_net, age_net, age_backend, gender_net, eye_cascade, emotion_net)


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


def predict_gender(gender_net, blob: np.ndarray) -> str:
    gender_net.setInput(blob)
    return GENDER_LIST[gender_net.forward()[0].argmax()]


def predict_age_caffe(age_net, blob: np.ndarray) -> str:
    age_net.setInput(blob)
    return AGE_LIST[age_net.forward()[0].argmax()]


def predict_age_ssrnet(age_net, face_bgr: np.ndarray) -> str:
    """Predict a continuous age with SSR-Net and format it as a label string."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - SSRNET_MEAN) / SSRNET_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        age = age_net(tensor).item()
    return f"{age:.0f}"


def predict_emotion(emotion_net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - EMOTION_MEAN) / EMOTION_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits, _, _ = emotion_net(tensor)
    return EMOTION_LABELS[logits[0].argmax().item()]


def detect_drowsiness(eye_cascade, face_bgr: np.ndarray) -> bool:
    """Return True if fewer than MIN_EYES_OPEN eyes are visible (eyes likely closed)."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    return len(eyes) < MIN_EYES_OPEN


def draw_outlined_text(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
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


def analyze_frame(models: Models, frame: np.ndarray, conf_threshold: float):
    """Detect faces and run age/gender/drowsiness/emotion inference. No Streamlit calls (safe for background threads)."""
    annotated_frame = frame.copy()
    face_boxes = detect_faces(models.face_net, frame, conf_threshold)
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

        gender, age = None, None
        if models.gender_net is not None or (models.age_net is not None and models.age_backend == "caffe"):
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            if models.gender_net is not None:
                gender = predict_gender(models.gender_net, blob)
            if models.age_net is not None and models.age_backend == "caffe":
                age = predict_age_caffe(models.age_net, blob)
        if models.age_net is not None and models.age_backend == "ssrnet":
            age = predict_age_ssrnet(models.age_net, face)

        emotion = predict_emotion(models.emotion_net, face) if models.emotion_net is not None else None
        drowsy = detect_drowsiness(models.eye_cascade, face) if models.eye_cascade is not None else None
        any_drowsy = any_drowsy or bool(drowsy)

        label_parts = [p for p in (gender, age, emotion) if p is not None]
        label = ", ".join(label_parts) if label_parts else "Face"

        # Draw green box and cyan overlay text with black outline for readability
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame.shape[0] / 150)), 8)
        draw_outlined_text(annotated_frame, label, (x1, y1 - 10), (0, 255, 255))

        status_label = None
        if drowsy is not None:
            status_label = "DROWSY" if drowsy else "ALERT"
            status_color = (0, 0, 255) if drowsy else (0, 255, 0)
            draw_outlined_text(annotated_frame, status_label, (x1, y1 + (y2 - y1) + 30), status_color)

        caption = f"TARGET_{idx}: {label}" + (f" | {status_label}" if status_label else "")
        cropped_faces.append((caption, cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))

    return annotated_frame, cropped_faces, any_drowsy, bool(face_boxes)
