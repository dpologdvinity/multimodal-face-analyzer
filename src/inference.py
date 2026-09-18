"""Model loading and per-face prediction logic for the Streamlit app (src/app.py).

Deliberately not shared with detect.py -- the CLI and the app duplicate the
pipeline on purpose (see CLAUDE.md); this module only exists to keep app.py
itself from growing unbounded as more model backends are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

try:
    from nets.deepface_race import build_race_model
    from nets.deepface_gender import build_gender_model
    from nets.mini_xception_model import build_mini_xception
    TF_SUPPORTED = True
except ImportError:
    TF_SUPPORTED = False

try:
    from nets.mivolo.inference_wrapper import MiVOLOInference
    MIVOLO_SUPPORTED = True
except ImportError:
    MIVOLO_SUPPORTED = False

try:
    import mediapipe as mp
    MEDIAPIPE_SUPPORTED = True
except ImportError:
    MEDIAPIPE_SUPPORTED = False

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
INSIGHTFACE_MODEL = MODEL_DIR / "insightface_genderage.onnx"
EFFICIENTNET_EMOTION_MODEL = MODEL_DIR / "efficientnet_b0_fer.onnx"
MINI_XCEPTION_MODEL = MODEL_DIR / "mini_xception_fer.h5"
FERPLUS_MODEL = MODEL_DIR / "emotion_ferplus.onnx"
FAIRFACE_MODEL = MODEL_DIR / "fairface_7class.onnx"
DEEPFACE_RACE_MODEL = MODEL_DIR / "deepface_race.h5"
DEEPFACE_GENDER_MODEL = MODEL_DIR / "deepface_gender.h5"
DEX_PROTO = MODEL_DIR / "dex_age.prototxt"
DEX_MODEL = MODEL_DIR / "dex_age.caffemodel"
MIVOLO_MODEL = MODEL_DIR / "mivolo_v2.safetensors"
BLENDSHAPES_MODEL = MODEL_DIR / "face_landmarker.task"

MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LABELS_DAN = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
EMOTION_LABELS_EFFICIENTNET = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_MINI_XCEPTION = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_FERPLUS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
EMOTION_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
EMOTION_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SSRNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SSRNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_EYES_OPEN = 2
RACE_LABELS_FAIRFACE = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
RACE_LABELS_DEEPFACE = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
RACE_CLOSE_MARGIN = 0.10  # show top-2 race classes together if within this probability margin
DEX_MEAN_VALUES = (103.939, 116.779, 123.68)  # VGG-16 ImageNet BGR mean, per DEX's own preprocessing

# Model keys per feature, in quickest-to-build order (first = default).
# Must match the numbered options in build-and-run.sh and the Dockerfile ARGs.
AGE_MODEL_OPTIONS = ["caffe", "insightface", "ssrnet", "fairface", "dex", "mivolo"]
GENDER_MODEL_OPTIONS = ["caffe", "insightface", "deepface", "fairface", "mivolo"]
FAIRFACE_AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
EMOTION_MODEL_OPTIONS = ["efficientnet", "ferplus", "mini_xception", "dan"]
DROWSINESS_MODEL_OPTIONS = ["haarcascade"]
RACE_MODEL_OPTIONS = ["fairface", "deepface"]
EXPRESSION_MODEL_OPTIONS = ["blendshapes"]


@dataclass
class Models:
    face_net: cv2.dnn.Net
    age_nets: dict = field(default_factory=dict)
    gender_nets: dict = field(default_factory=dict)
    emotion_nets: dict = field(default_factory=dict)
    drowsiness_nets: dict = field(default_factory=dict)
    race_nets: dict = field(default_factory=dict)
    expression_nets: dict = field(default_factory=dict)

    @property
    def offline_features(self) -> list[str]:
        return [
            name for name, nets in [
                ("AGE", self.age_nets), ("GENDER", self.gender_nets),
                ("EMOTION", self.emotion_nets), ("DROWSINESS", self.drowsiness_nets),
                ("RACE", self.race_nets), ("EXPRESSION", self.expression_nets),
            ] if not nets
        ]


def load_models() -> Models:
    """Load every model whose file(s)/dependencies are present. Face detection is required;
    age, gender, emotion, and drowsiness are each optional per-model-key -- a model is only
    present in its feature's dict if it loaded successfully, so the app degrades gracefully
    to whichever models were built in. Which of the loaded models are actually used per frame
    is chosen at runtime by the caller (see analyze_frame's active_* arguments)."""
    if not FACE_PROTO.exists() or not FACE_MODEL.exists():
        raise FileNotFoundError(f"Missing face detector file(s) in {MODEL_DIR}: {FACE_PROTO.name}, {FACE_MODEL.name} (required).")
    face_net = cv2.dnn.readNet(str(FACE_MODEL), str(FACE_PROTO))

    age_nets = {}
    if AGE_PROTO.exists() and AGE_MODEL.exists():
        age_nets["caffe"] = cv2.dnn.readNet(str(AGE_MODEL), str(AGE_PROTO))
    if TORCH_SUPPORTED and SSRNET_MODEL.exists():
        net = SSRNet()
        checkpoint = torch.load(str(SSRNET_MODEL), map_location="cpu")
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        age_nets["ssrnet"] = net
    if DEX_PROTO.exists() and DEX_MODEL.exists():
        age_nets["dex"] = cv2.dnn.readNetFromCaffe(str(DEX_PROTO), str(DEX_MODEL))

    gender_nets = {}
    if GENDER_PROTO.exists() and GENDER_MODEL.exists():
        gender_nets["caffe"] = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))

    if INSIGHTFACE_MODEL.exists():
        insightface_net = cv2.dnn.readNetFromONNX(str(INSIGHTFACE_MODEL))
        age_nets["insightface"] = insightface_net
        gender_nets["insightface"] = insightface_net

    if TF_SUPPORTED and DEEPFACE_GENDER_MODEL.exists():
        gender_nets["deepface"] = build_gender_model(str(DEEPFACE_GENDER_MODEL))

    if MIVOLO_SUPPORTED and MIVOLO_MODEL.exists():
        mivolo_config = MODEL_DIR / "mivolo_v2_config.json"
        if mivolo_config.exists():
            mivolo_net = MiVOLOInference(
                model_path=str(MIVOLO_MODEL),
                config_path=str(mivolo_config),
                device="cpu",
                half=False,
                verbose=False,
            )
            age_nets["mivolo"] = mivolo_net
            gender_nets["mivolo"] = mivolo_net

    emotion_nets = {}
    if TORCH_SUPPORTED and EMOTION_MODEL.exists():
        net = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint = torch.load(str(EMOTION_MODEL), map_location="cpu")
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        emotion_nets["dan"] = net
    if EFFICIENTNET_EMOTION_MODEL.exists():
        emotion_nets["efficientnet"] = cv2.dnn.readNetFromONNX(str(EFFICIENTNET_EMOTION_MODEL))
    if TF_SUPPORTED and MINI_XCEPTION_MODEL.exists():
        mini_xception_net = build_mini_xception((64, 64, 1), num_classes=7)
        mini_xception_net.load_weights(str(MINI_XCEPTION_MODEL))
        emotion_nets["mini_xception"] = mini_xception_net
    if FERPLUS_MODEL.exists():
        emotion_nets["ferplus"] = cv2.dnn.readNetFromONNX(str(FERPLUS_MODEL))

    drowsiness_nets = {}
    if EYE_CASCADE_FILE.exists():
        drowsiness_nets["haarcascade"] = cv2.CascadeClassifier(str(EYE_CASCADE_FILE))

    if FAIRFACE_MODEL.exists():
        fairface_net = cv2.dnn.readNetFromONNX(str(FAIRFACE_MODEL))
        age_nets["fairface"] = fairface_net
        gender_nets["fairface"] = fairface_net

    race_nets = {}
    if FAIRFACE_MODEL.exists():
        race_nets["fairface"] = age_nets["fairface"]
    if TF_SUPPORTED and DEEPFACE_RACE_MODEL.exists():
        race_nets["deepface"] = build_race_model(str(DEEPFACE_RACE_MODEL))

    expression_nets = {}
    if MEDIAPIPE_SUPPORTED and BLENDSHAPES_MODEL.exists():
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(BLENDSHAPES_MODEL)),
            output_face_blendshapes=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        expression_nets["blendshapes"] = landmarker

    return Models(face_net, age_nets, gender_nets, emotion_nets, drowsiness_nets, race_nets, expression_nets)


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


def predict_gender_caffe(net, blob: np.ndarray) -> str:
    net.setInput(blob)
    return GENDER_LIST[net.forward()[0].argmax()]


def predict_age_caffe(net, blob: np.ndarray) -> str:
    net.setInput(blob)
    return AGE_LIST[net.forward()[0].argmax()]


def predict_age_ssrnet(net, face_bgr: np.ndarray) -> str:
    """Predict a continuous age with SSR-Net and format it as a label string."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - SSRNET_MEAN) / SSRNET_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        age = net(tensor).item()
    return f"{age:.0f}"


def predict_age_dex(net, face_bgr: np.ndarray) -> str:
    """Predict a continuous age with DEX (Deep EXpectation): 101-class softmax over ages 0-100,
    decoded as an expected value (weighted sum of class centers), not argmax."""
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (224, 224), DEX_MEAN_VALUES, swapRB=False, crop=False)
    net.setInput(blob)
    probs = net.forward().flatten()
    age = sum(p * i for i, p in enumerate(probs))
    return f"{age:.0f}"


INSIGHTFACE_INPUT_SIZE = 96  # this genderage.onnx's actual input size (per its ONNX graph) --
# NOT the 112x112 insightface uses for its face-recognition/embedding models; verified via
# onnxruntime, which rejects 112x112 with a shape-mismatch error. cv2.dnn silently accepted the
# wrong shape and produced near-constant garbage output instead of erroring.


def _margin_align(frame_bgr: np.ndarray, box: tuple[int, int, int, int], output_size: int, margin: float) -> np.ndarray:
    """Crop centered on the raw detection box, scaled so the box fits into output_size with the
    given margin factor (e.g. margin=1.5 means the box occupies 1/1.5 of the output). No rotation.
    Must operate on the ORIGINAL frame and the UNPADDED detection box -- several of these models
    were trained on a specific bbox-relative or landmark-based framing, not an arbitrarily-padded
    pixel crop+resize (feeding a mismatched framing gives wrong/biased predictions, not a crash)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale = output_size / (max(w, h) * margin)
    m = np.array([
        [scale, 0, output_size / 2 - scale * cx],
        [0, scale, output_size / 2 - scale * cy],
    ], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, m, (output_size, output_size), borderValue=0.0)


def _insightface_forward(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    # Replicates insightface's own alignment (model_zoo/attribute.py + utils/face_align.py): 1.5x margin.
    aligned = _margin_align(frame_bgr, box, INSIGHTFACE_INPUT_SIZE, margin=1.5)
    blob = cv2.dnn.blobFromImage(aligned, 1.0 / 128.0, (INSIGHTFACE_INPUT_SIZE, INSIGHTFACE_INPUT_SIZE), (127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    return net.forward().flatten()


def predict_gender_insightface(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    out = _insightface_forward(net, frame_bgr, box)
    return "Male" if np.argmax(out[:2]) == 0 else "Female"


def predict_age_insightface(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    out = _insightface_forward(net, frame_bgr, box)
    return f"{round(out[2] * 100):.0f}"


def predict_age_mivolo(net: MiVOLOInference, face_bgr: np.ndarray) -> str:
    """Predict age with MiVOLO on a face crop (face-only mode)."""
    age, _, _ = net.predict_face(face_bgr)
    return f"{int(round(age))}"


def predict_gender_mivolo(net: MiVOLOInference, face_bgr: np.ndarray) -> str:
    """Predict gender with MiVOLO on a face crop (face-only mode)."""
    _, gender, _ = net.predict_face(face_bgr)
    # MiVOLO returns 'male'/'female' (lowercase); normalize to "Male"/"Female"
    return "Male" if gender == "male" else "Female"


def predict_emotion_dan(net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS_DAN."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - EMOTION_MEAN) / EMOTION_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits, _, _ = net(tensor)
    return EMOTION_LABELS_DAN[logits[0].argmax().item()]


def predict_emotion_efficientnet(net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS_EFFICIENTNET."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(face_rgb, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=False, crop=False)
    blob = (blob - EMOTION_MEAN.reshape(1, 3, 1, 1)) / EMOTION_STD.reshape(1, 3, 1, 1)
    net.setInput(blob.astype(np.float32))
    logits = net.forward().flatten()
    return EMOTION_LABELS_EFFICIENTNET[int(np.argmax(logits))]


def predict_emotion_mini_xception(net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS_MINI_XCEPTION."""
    face_gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    face_norm = (face_gray / 255.0 - 0.5) * 2.0
    tensor = face_norm[np.newaxis, ..., np.newaxis]
    probs = net.predict(tensor, verbose=0).flatten()
    return EMOTION_LABELS_MINI_XCEPTION[int(np.argmax(probs))]


def predict_emotion_ferplus(net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS_FERPLUS."""
    face_gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    blob = face_gray[np.newaxis, np.newaxis, ...]
    net.setInput(blob)
    logits = net.forward().flatten()
    return EMOTION_LABELS_FERPLUS[int(np.argmax(logits))]


def detect_drowsiness_haarcascade(eye_cascade, face_bgr: np.ndarray) -> bool:
    """Return True if fewer than MIN_EYES_OPEN eyes are visible (eyes likely closed)."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    return len(eyes) < MIN_EYES_OPEN


def _format_results(pairs: list[tuple[str, str]]) -> list[str]:
    """Plain values only -- no model-name prefix, even with multiple models active per feature."""
    return [value for _, value in pairs]


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()


def _format_race_label(probs: np.ndarray, labels: list[str]) -> str:
    """Format the top race prediction, showing the top-2 together if their probabilities are close."""
    order = np.argsort(probs)[::-1]
    top1, top2 = order[0], order[1]
    if probs[top1] - probs[top2] < RACE_CLOSE_MARGIN:
        return f"{labels[top1]} ({probs[top1] * 100:.0f}%)/{labels[top2]} ({probs[top2] * 100:.0f}%)"
    return labels[top1]


def _fairface_forward(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int], output_name: str) -> np.ndarray:
    # Same alignment as predict_race_fairface -- one ONNX graph, three named outputs
    # (race_output, gender_output, age_output); re-run per feature for simplicity, matching
    # the insightface age/gender split.
    aligned = _margin_align(frame_bgr, box, 224, margin=1.5)
    face_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - SSRNET_MEAN) / SSRNET_STD
    blob = face_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    net.setInput(blob)
    return net.forward(output_name).flatten()


def predict_race_fairface(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    # FairFace's own pipeline aligns on 5-point landmarks (dlib, padding=0.25); we have no
    # landmark model, so approximate with the same margin via a bbox-centered crop (padding=0.25
    # each side ~= a 1.5x margin), instead of an arbitrary fixed-pixel-padding crop+resize.
    logits = _fairface_forward(net, frame_bgr, box, "race_output")
    return _format_race_label(_softmax(logits), RACE_LABELS_FAIRFACE)


def predict_gender_fairface(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    out = _fairface_forward(net, frame_bgr, box, "gender_output")
    return "Male" if np.argmax(out) == 0 else "Female"


def predict_age_fairface(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    out = _fairface_forward(net, frame_bgr, box, "age_output")
    return FAIRFACE_AGE_LABELS[int(np.argmax(out))]


def predict_race_deepface(net, face_bgr: np.ndarray) -> str:
    face_resized = cv2.resize(face_bgr, (224, 224)).astype(np.float32)
    probs = net.predict(face_resized[np.newaxis, ...], verbose=0).flatten()
    return _format_race_label(probs, RACE_LABELS_DEEPFACE)


def predict_gender_deepface(net, face_bgr: np.ndarray) -> str:
    # Same VGGFace-backbone preprocessing as predict_race_deepface: 224x224 BGR, unnormalized [0,255].
    face_resized = cv2.resize(face_bgr, (224, 224)).astype(np.float32)
    probs = net.predict(face_resized[np.newaxis, ...], verbose=0).flatten()
    # deepface's GENDER_LABELS = ["Woman", "Man"]; normalize to this repo's Male/Female convention.
    return "Male" if np.argmax(probs) == 1 else "Female"


def predict_expression_blendshapes(landmarker, face_bgr: np.ndarray) -> str:
    """Predict facial expression via MediaPipe's BlendShapes (52 continuous muscle coefficients).
    Returns the top 3 highest-scoring blendshapes as a comma-separated string,
    or 'no landmarks' if no face is detected."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)

    result = landmarker.detect(mp_image)

    if not result.face_blendshapes or len(result.face_blendshapes) == 0:
        return "no landmarks"

    blendshapes = result.face_blendshapes[0]
    # Sort by score descending
    sorted_blendshapes = sorted(blendshapes, key=lambda x: x.score, reverse=True)

    # Take top 3, skip "_neutral" if present
    top_blendshapes = []
    for bs in sorted_blendshapes:
        if bs.category_name != "_neutral":
            top_blendshapes.append(bs)
        if len(top_blendshapes) >= 3:
            break

    if not top_blendshapes:
        return "no landmarks"

    # Format as "name1 0.82, name2 0.15, name3 0.09"
    return ", ".join(f"{bs.category_name} {bs.score:.2f}" for bs in top_blendshapes)


def draw_outlined_text(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Draw text with a black outline so it stays readable over any background. Clamps origin
    so text stays inside the frame, and shrinks the font if the text is wider than the frame
    itself (clamping alone can't fix that -- a line wider than the frame overflows regardless
    of x position)."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    frame_h, frame_w = frame.shape[:2]

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    while text_w > frame_w and scale > 0.3:
        scale -= 0.1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = org
    x = max(0, min(x, frame_w - text_w))
    y = max(text_h, min(y, frame_h - baseline))
    org = (x, y)

    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def analyze_frame(
    models: Models,
    frame: np.ndarray,
    conf_threshold: float,
    active_age: set,
    active_gender: set,
    active_emotion: set,
    active_drowsiness: set,
    active_race: set,
    active_expression: set,
):
    """Detect faces and run inference for whichever model keys are active per feature.
    Multiple active models for the same feature (e.g. active_age = {"caffe", "ssrnet"})
    all run and are shown together. No Streamlit calls (safe for background threads)."""
    annotated_frame = frame.copy()
    face_boxes = detect_faces(models.face_net, frame, conf_threshold)
    cropped_faces = []
    any_drowsy = False

    need_blob227 = ("caffe" in active_age and "caffe" in models.age_nets) or \
                   ("caffe" in active_gender and "caffe" in models.gender_nets)

    for idx, (x1, y1, x2, y2) in enumerate(face_boxes, 1):
        y1_crop = max(0, y1 - 20)
        y2_crop = min(y2 + 20, frame.shape[0] - 1)
        x1_crop = max(0, x1 - 20)
        x2_crop = min(x2 + 20, frame.shape[1] - 1)

        face = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        blob227 = None
        if need_blob227:
            blob227 = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        age_pairs = []
        for key in active_age:
            net = models.age_nets.get(key)
            if net is None:
                continue
            if key == "caffe":
                value = predict_age_caffe(net, blob227)
            elif key == "ssrnet":
                value = predict_age_ssrnet(net, face)
            elif key == "fairface":
                value = predict_age_fairface(net, frame, (x1, y1, x2, y2))
            elif key == "dex":
                value = predict_age_dex(net, face)
            elif key == "mivolo":
                value = predict_age_mivolo(net, face)
            else:
                value = predict_age_insightface(net, frame, (x1, y1, x2, y2))
            age_pairs.append((key, value))

        gender_pairs = []
        for key in active_gender:
            net = models.gender_nets.get(key)
            if net is None:
                continue
            if key == "caffe":
                value = predict_gender_caffe(net, blob227)
            elif key == "deepface":
                value = predict_gender_deepface(net, face)
            elif key == "fairface":
                value = predict_gender_fairface(net, frame, (x1, y1, x2, y2))
            elif key == "mivolo":
                value = predict_gender_mivolo(net, face)
            else:
                value = predict_gender_insightface(net, frame, (x1, y1, x2, y2))
            gender_pairs.append((key, value))

        emotion_pairs = []
        for key in active_emotion:
            net = models.emotion_nets.get(key)
            if net is None:
                continue
            if key == "dan":
                value = predict_emotion_dan(net, face)
            elif key == "mini_xception":
                value = predict_emotion_mini_xception(net, face)
            elif key == "ferplus":
                value = predict_emotion_ferplus(net, face)
            else:
                value = predict_emotion_efficientnet(net, face)
            emotion_pairs.append((key, value))

        race_pairs = []
        for key in active_race:
            net = models.race_nets.get(key)
            if net is None:
                continue
            value = predict_race_fairface(net, frame, (x1, y1, x2, y2)) if key == "fairface" else predict_race_deepface(net, face)
            race_pairs.append((key, value))

        expression_pairs = []
        for key in active_expression:
            net = models.expression_nets.get(key)
            if net is None:
                continue
            value = predict_expression_blendshapes(net, face)
            expression_pairs.append((key, value))

        drowsy_pairs = []
        face_drowsy = False
        for key in active_drowsiness:
            net = models.drowsiness_nets.get(key)
            if net is None:
                continue
            drowsy = detect_drowsiness_haarcascade(net, face)
            face_drowsy = face_drowsy or drowsy
            drowsy_pairs.append((key, "DROWSY" if drowsy else "ALERT"))
        any_drowsy = any_drowsy or face_drowsy

        # Attribute text is intentionally NOT drawn on the shared image -- with several faces
        # close together, per-face text overlaps illegibly. The box + a small index number is
        # the only thing burned into pixels; full results are returned as structured data for
        # the caller to render as separate per-face UI (see src/app.py's target cards).
        box_thickness = int(round(frame.shape[0] / 150)) or 1
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness, 8)
        draw_outlined_text(annotated_frame, str(idx), (x1, max(20, y1 - 10)), (0, 255, 255))

        drowsy_parts = _format_results(drowsy_pairs)
        status = drowsy_parts[0] if len(drowsy_parts) == 1 else (", ".join(drowsy_parts) if drowsy_parts else None)

        cropped_faces.append({
            "idx": idx,
            "image": cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
            "age": _format_results(age_pairs),
            "gender": _format_results(gender_pairs),
            "race": _format_results(race_pairs),
            "emotion": _format_results(emotion_pairs),
            "expression": _format_results(expression_pairs),
            "status": status,
            "drowsy": face_drowsy if drowsy_pairs else None,
        })

    return annotated_frame, cropped_faces, any_drowsy, bool(face_boxes)
