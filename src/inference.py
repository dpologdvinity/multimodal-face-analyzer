"""Model loading and per-face prediction logic for the Streamlit app (src/app.py).

Deliberately not shared with detect.py -- the CLI and the app duplicate the
pipeline on purpose (see CLAUDE.md); this module only exists to keep app.py
itself from growing unbounded as more model backends are added.
"""
from __future__ import annotations

import json
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
    from nets.deepface_recognition import build_recognition_model
    from nets.mini_xception_model import build_mini_xception
    from nets.mask_model import build_mask_model
    from nets.skin_tone_model import build_skin_tone_model
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
DEEPFACE_RECOGNITION_MODEL = MODEL_DIR / "deepface_vgg.h5"
DEX_PROTO = MODEL_DIR / "dex_age.prototxt"
DEX_MODEL = MODEL_DIR / "dex_age.caffemodel"
MIVOLO_MODEL = MODEL_DIR / "mivolo_v2.safetensors"
BLENDSHAPES_MODEL = MODEL_DIR / "face_landmarker.task"
BISENET_MODEL = MODEL_DIR / "bisenet_face_parsing.onnx"
SKIN_TONE_MODEL = MODEL_DIR / "skin_tone_mobilenetv2.h5"
GLASSES_MODEL = MODEL_DIR / "glasses_detector.onnx"
MASK_MODEL = MODEL_DIR / "mask_detector.h5"
COLORIZATION_PROTO = MODEL_DIR / "colorization_deploy_v2.prototxt"
COLORIZATION_MODEL = MODEL_DIR / "colorization_release_v2.caffemodel"
COLORIZATION_PTS = MODEL_DIR / "pts_in_hull.npy"
POSE_PROTO = MODEL_DIR / "pose_deploy_linevec_faster_4_stages.prototxt"
POSE_MODEL = MODEL_DIR / "pose_iter_160000.caffemodel"
HAND_LANDMARKER_MODEL = MODEL_DIR / "hand_landmarker.task"

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
RECOGNITION_COSINE_THRESHOLD = 0.68  # deepface's own default VGG-Face verification threshold
DEX_MEAN_VALUES = (103.939, 116.779, 123.68)  # VGG-16 ImageNet BGR mean, per DEX's own preprocessing
GALLERY_FILE = BASE_DIR / "gallery" / "known_faces.json"
KNOWN_PEOPLE_DIR = BASE_DIR / "known_people"  # bundled reference photos for identity search (see README)
IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Model keys per feature, in quickest-to-build order (first = default).
# Must match the numbered options in build-and-run.sh and the Dockerfile ARGs.
AGE_MODEL_OPTIONS = ["caffe", "insightface", "ssrnet", "fairface", "dex", "mivolo"]
GENDER_MODEL_OPTIONS = ["caffe", "insightface", "deepface", "fairface", "mivolo"]
FAIRFACE_AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
EMOTION_MODEL_OPTIONS = ["efficientnet", "ferplus", "mini_xception", "dan"]
DROWSINESS_MODEL_OPTIONS = ["haarcascade"]
RACE_MODEL_OPTIONS = ["fairface", "deepface"]
EXPRESSION_MODEL_OPTIONS = ["blendshapes"]
RECOGNITION_MODEL_OPTIONS = ["vggface"]
FACIAL_HAIR_MODEL_OPTIONS = ["bisenet"]
SKIN_TONE_MODEL_OPTIONS = ["mobilenetv2"]
GLASSES_MODEL_OPTIONS = ["mobilenet"]
MASK_MODEL_OPTIONS = ["mobilenetv2"]
HAIR_COLOR_MODEL_OPTIONS = ["colorimetric"]
EYE_COLOR_MODEL_OPTIONS = ["colorimetric"]
COLORIZATION_MODEL_OPTIONS = ["eccv16"]
GRAYSCALE_CHANNEL_DIFF_THRESHOLD = 3.0  # mean abs diff between B/G/R below this => treat as grayscale
POSE_MODEL_OPTIONS = ["mpi"]
POSE_INPUT_SIZE = 368  # square, per this model's own training resolution
POSE_CONFIDENCE_THRESHOLD = 0.1  # this specific MPI checkpoint's own confidence maps run low
MIN_POSE_POINTS = 3  # fewer confident keypoints than this => "no body in frame", skip silently
# MPI 15-point skeleton (index 15 is a "Background" channel, unused): Head, Neck, R/L
# Shoulder/Elbow/Wrist, R/L Hip/Knee/Ankle, Chest. Standard OpenCV MPI sample layout.
MPI_POSE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 14),
    (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
]
MPI_POSE_NUM_POINTS = 15
FACE_LANDMARKS_MODEL_OPTIONS = ["blendshapes"]  # reuses the same FaceLandmarker model as Expression
HAND_MODEL_OPTIONS = ["mediapipe"]
# Standard MediaPipe 21-point hand skeleton (HandLandmark enum order, see ideas/hands.md)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm
]

# Per-face image adjustment sliders (Lightroom-style), applied to each face crop before any
# classifier runs on it. Pure OpenCV/numpy, no model file. (slider_key -> (min, max, default)),
# all sliders default to 0 (no-op) so an untouched panel changes nothing.
IMAGE_ADJUSTMENT_RANGES = {
    "exposure": (-3.0, 3.0, 0.0),        # stops (2**value gain)
    "brightness": (-100.0, 100.0, 0.0),  # additive, 0-255 scale
    "contrast": (-100.0, 100.0, 0.0),
    "highlights": (-100.0, 100.0, 0.0),
    "shadows": (-100.0, 100.0, 0.0),
    "black_point": (-100.0, 100.0, 0.0),
    "saturation": (-100.0, 100.0, 0.0),
    "vibrance": (-100.0, 100.0, 0.0),
    "sharpness": (0.0, 100.0, 0.0),
    "definition": (0.0, 100.0, 0.0),
    "noise_reduction": (0.0, 100.0, 0.0),
}

SKIN_TONE_LABELS = ["black", "brown", "white"]  # index order per the source model's own class map
SKIN_TONE_INPUT_SIZE = (120, 90)  # (width, height) -- this model's own idiosyncratic input shape, not 224x224
MASK_LABELS = ["with_mask", "without_mask"]  # sklearn LabelBinarizer's alphabetical class order
HAIR_COLOR_LABELS = ["black", "brown", "blonde", "red", "grey", "white"]
EYE_COLOR_LABELS = ["brown", "blue", "green", "hazel", "grey", "amber"]
GLASSES_THRESHOLD = 0.5
FACIAL_HAIR_COVERAGE_THRESHOLD = 0.15  # fraction of lower-face pixels in BiSeNet's hair/beard class to call it "beard"


@dataclass
class Models:
    face_net: cv2.dnn.Net
    age_nets: dict = field(default_factory=dict)
    gender_nets: dict = field(default_factory=dict)
    emotion_nets: dict = field(default_factory=dict)
    drowsiness_nets: dict = field(default_factory=dict)
    race_nets: dict = field(default_factory=dict)
    expression_nets: dict = field(default_factory=dict)
    recognition_nets: dict = field(default_factory=dict)
    facial_hair_nets: dict = field(default_factory=dict)
    skin_tone_nets: dict = field(default_factory=dict)
    glasses_nets: dict = field(default_factory=dict)
    mask_nets: dict = field(default_factory=dict)
    hair_color_nets: dict = field(default_factory=dict)
    eye_color_nets: dict = field(default_factory=dict)
    colorization_nets: dict = field(default_factory=dict)
    pose_nets: dict = field(default_factory=dict)
    face_landmarks_nets: dict = field(default_factory=dict)
    hand_nets: dict = field(default_factory=dict)

    @property
    def offline_features(self) -> list[str]:
        return [
            name for name, nets in [
                ("AGE", self.age_nets), ("GENDER", self.gender_nets),
                ("EMOTION", self.emotion_nets), ("DROWSINESS", self.drowsiness_nets),
                ("RACE", self.race_nets), ("EXPRESSION", self.expression_nets),
                ("RECOGNITION", self.recognition_nets), ("FACIAL_HAIR", self.facial_hair_nets),
                ("SKIN_TONE", self.skin_tone_nets), ("GLASSES", self.glasses_nets),
                ("MASK", self.mask_nets), ("HAIR_COLOR", self.hair_color_nets),
                ("EYE_COLOR", self.eye_color_nets), ("COLORIZATION", self.colorization_nets),
                ("POSE", self.pose_nets), ("FACE_LANDMARKS", self.face_landmarks_nets),
                ("HANDS", self.hand_nets),
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

    recognition_nets = {}
    if TF_SUPPORTED and DEEPFACE_RECOGNITION_MODEL.exists():
        recognition_nets["vggface"] = build_recognition_model(str(DEEPFACE_RECOGNITION_MODEL))

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

    # Colorimetric heuristics need no model file, no dependency beyond OpenCV -- always
    # available. hair_color has no further precondition; eye_color reuses the same
    # haarcascade_eye.xml as drowsiness, so it's gated on that file existing.
    hair_color_nets = {"colorimetric": True}
    eye_color_nets = {}
    if EYE_CASCADE_FILE.exists():
        eye_color_nets["colorimetric"] = drowsiness_nets["haarcascade"] if "haarcascade" in drowsiness_nets else cv2.CascadeClassifier(str(EYE_CASCADE_FILE))

    expression_nets = {}
    face_landmarks_nets = {}
    if MEDIAPIPE_SUPPORTED and BLENDSHAPES_MODEL.exists():
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(BLENDSHAPES_MODEL)),
            output_face_blendshapes=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        expression_nets["blendshapes"] = landmarker
        face_landmarks_nets["blendshapes"] = landmarker  # same model instance, two features

    facial_hair_nets = {}
    if BISENET_MODEL.exists():
        facial_hair_nets["bisenet"] = cv2.dnn.readNetFromONNX(str(BISENET_MODEL))

    skin_tone_nets = {}
    if TF_SUPPORTED and SKIN_TONE_MODEL.exists():
        skin_tone_nets["mobilenetv2"] = build_skin_tone_model(str(SKIN_TONE_MODEL))

    glasses_nets = {}
    if GLASSES_MODEL.exists():
        glasses_nets["mobilenet"] = cv2.dnn.readNetFromONNX(str(GLASSES_MODEL))

    mask_nets = {}
    if TF_SUPPORTED and MASK_MODEL.exists():
        mask_nets["mobilenetv2"] = build_mask_model(str(MASK_MODEL))

    colorization_nets = {}
    if COLORIZATION_PROTO.exists() and COLORIZATION_MODEL.exists() and COLORIZATION_PTS.exists():
        colorization_net = cv2.dnn.readNetFromCaffe(str(COLORIZATION_PROTO), str(COLORIZATION_MODEL))
        pts = np.load(str(COLORIZATION_PTS))
        class8 = colorization_net.getLayerId("class8_ab")
        conv8 = colorization_net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        colorization_net.getLayer(class8).blobs = [pts.astype("float32")]
        colorization_net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        colorization_nets["eccv16"] = colorization_net

    pose_nets = {}
    if POSE_PROTO.exists() and POSE_MODEL.exists():
        pose_nets["mpi"] = cv2.dnn.readNetFromCaffe(str(POSE_PROTO), str(POSE_MODEL))

    hand_nets = {}
    if MEDIAPIPE_SUPPORTED and HAND_LANDMARKER_MODEL.exists():
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        hand_nets["mediapipe"] = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

    return Models(
        face_net, age_nets, gender_nets, emotion_nets, drowsiness_nets, race_nets, expression_nets, recognition_nets,
        facial_hair_nets, skin_tone_nets, glasses_nets, mask_nets, hair_color_nets, eye_color_nets, colorization_nets,
        pose_nets, face_landmarks_nets, hand_nets,
    )


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


def is_grayscale_frame(frame_bgr: np.ndarray) -> bool:
    """Heuristic: a 3-channel image that's actually grayscale (common for old photos saved
    as BGR/RGB with all channels equal, or scanned B&W) has near-zero difference between its
    B/G/R channels across the whole image. Downsamples first -- only the mean matters, and a
    small sample is far cheaper than scanning a full-resolution frame."""
    small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    diff = (np.abs(b - g) + np.abs(g - r) + np.abs(b - r)) / 3.0
    return float(diff.mean()) < GRAYSCALE_CHANNEL_DIFF_THRESHOLD


def colorize_frame(net, frame_bgr: np.ndarray) -> np.ndarray:
    """ECCV16 colorization (Zhang et al., richzhang/colorization): predict the Lab 'ab' channels
    from the 'L' channel and rejoin. See ideas/colorization.md for the reference implementation
    this follows."""
    scaled = frame_bgr.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (frame_bgr.shape[1], frame_bgr.shape[0]))

    L_full = cv2.split(lab_img)[0]
    colorized = np.concatenate((L_full[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")


def maybe_colorize(models: "Models", frame_bgr: np.ndarray, active_colorization: set) -> tuple[np.ndarray, bool]:
    """Auto-colorize frame_bgr if it's detected as grayscale and the colorization backend is
    active; otherwise return it unchanged. Returns (frame, was_colorized)."""
    net = models.colorization_nets.get("eccv16")
    if net is None or "eccv16" not in active_colorization:
        return frame_bgr, False
    if not is_grayscale_frame(frame_bgr):
        return frame_bgr, False
    return colorize_frame(net, frame_bgr), True


def detect_pose_mpi(net, frame_bgr: np.ndarray) -> list[tuple[int, int] | None]:
    """CMU OpenPose MPI 15-point body pose (see ideas/pose.md). Single-person, whole-frame --
    returns one (x, y) per keypoint in frame_bgr's own coordinates, or None where confidence
    doesn't clear POSE_CONFIDENCE_THRESHOLD."""
    frame_h, frame_w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0 / 255, (POSE_INPUT_SIZE, POSE_INPUT_SIZE), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    out_h, out_w = output.shape[2], output.shape[3]
    points: list[tuple[int, int] | None] = []
    for i in range(MPI_POSE_NUM_POINTS):
        prob_map = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(prob_map)
        x = int((frame_w * point[0]) / out_w)
        y = int((frame_h * point[1]) / out_h)
        points.append((x, y) if prob > POSE_CONFIDENCE_THRESHOLD else None)
    return points


def draw_pose_skeleton(frame: np.ndarray, points: list[tuple[int, int] | None]) -> None:
    """Draw the MPI skeleton (joints + connecting bones) directly onto frame, HUD-style
    (matches the green/cyan palette used for face boxes elsewhere)."""
    for point_a, point_b in MPI_POSE_PAIRS:
        if points[point_a] is not None and points[point_b] is not None:
            cv2.line(frame, points[point_a], points[point_b], (0, 255, 0), 2, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)


def _adjust_exposure(img: np.ndarray, stops: float) -> np.ndarray:
    return img * (2.0 ** stops)


def _adjust_brightness(img: np.ndarray, amount: float) -> np.ndarray:
    return img + amount


def _adjust_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    c = amount * 2.55  # slider -100..100 -> classic contrast-correction-factor's -255..255
    factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
    return factor * (img - 128.0) + 128.0


def _adjust_tone_region(img_bgr: np.ndarray, amount: float, region: str) -> np.ndarray:
    """Shift highlights or shadows via a luminance-weighted mask in HSV's V channel.
    Positive `amount` brightens highlights / lifts shadows (Lightroom convention)."""
    hsv = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2]
    if region == "highlights":
        mask = np.clip((v - 128.0) / 127.0, 0.0, 1.0)
    else:
        mask = np.clip((128.0 - v) / 128.0, 0.0, 1.0)
    hsv[..., 2] = np.clip(v + (amount / 100.0) * 50.0 * mask, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def _adjust_black_point(img: np.ndarray, amount: float) -> np.ndarray:
    bp = np.clip((amount / 100.0) * 60.0, -60.0, 250.0)
    return (img - bp) * (255.0 / max(255.0 - bp, 1.0))


def _adjust_saturation(img_bgr: np.ndarray, amount: float, vibrance: bool = False) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    if vibrance:
        # Boost low-saturation pixels more than already-saturated ones (protects skin tones).
        s = s + (amount / 100.0) * 60.0 * (1.0 - s / 255.0)
    else:
        s = s * (1.0 + amount / 100.0)
    hsv[..., 1] = np.clip(s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def _adjust_sharpness(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Classic unsharp mask -- small-radius blur subtracted back out to boost edge contrast."""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.5)
    return img_bgr + (amount / 100.0) * 1.5 * (img_bgr - blurred)


def _adjust_definition(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """'Clarity'-style local contrast: large-radius unsharp mask on the LAB lightness channel
    only, so it boosts midtone structure without shifting color."""
    lab = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    blurred = cv2.GaussianBlur(L, (0, 0), sigmaX=12.0)
    lab[..., 0] = np.clip(L + (amount / 100.0) * 1.2 * (L - blurred), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def _adjust_noise_reduction(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Edge-preserving denoise (bilateral filter); strength scales with the slider."""
    strength = amount / 100.0
    return cv2.bilateralFilter(np.clip(img_bgr, 0, 255).astype(np.uint8), d=5, sigmaColor=strength * 100, sigmaSpace=strength * 100).astype(np.float32)


def apply_image_adjustments(face_bgr: np.ndarray, adjustments: dict) -> np.ndarray:
    """Apply the Lightroom-style slider stack to one face crop, in a fixed pipeline order
    (denoise first so later steps don't amplify grain; sharpen last so it acts on the final
    tonal/color state). Any slider left at its default (0) is skipped entirely -- cheap when
    the panel is untouched, since this runs once per face per frame."""
    img = face_bgr.astype(np.float32)

    if adjustments.get("noise_reduction", 0):
        img = _adjust_noise_reduction(img, adjustments["noise_reduction"])
    if adjustments.get("exposure", 0):
        img = _adjust_exposure(img, adjustments["exposure"])
    if adjustments.get("black_point", 0):
        img = _adjust_black_point(img, adjustments["black_point"])
    if adjustments.get("shadows", 0):
        img = _adjust_tone_region(img, adjustments["shadows"], "shadows")
    if adjustments.get("highlights", 0):
        img = _adjust_tone_region(img, adjustments["highlights"], "highlights")
    if adjustments.get("contrast", 0):
        img = _adjust_contrast(img, adjustments["contrast"])
    if adjustments.get("brightness", 0):
        img = _adjust_brightness(img, adjustments["brightness"])
    if adjustments.get("saturation", 0):
        img = _adjust_saturation(img, adjustments["saturation"])
    if adjustments.get("vibrance", 0):
        img = _adjust_saturation(img, adjustments["vibrance"], vibrance=True)
    if adjustments.get("definition", 0):
        img = _adjust_definition(img, adjustments["definition"])
    if adjustments.get("sharpness", 0):
        img = _adjust_sharpness(img, adjustments["sharpness"])

    return np.clip(img, 0, 255).astype(np.uint8)


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


def _estimate_roll_angle(face_bgr: np.ndarray, eye_cascade) -> float | None:
    """Detect two eyes via Haar cascade and return the roll angle (degrees) needed to
    level them, or None if fewer than 2 eyes found or the angle looks like noise."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    if len(eyes) < 2:
        return None
    # take the two largest detections (most confident), left-to-right by x center
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(eyes, key=lambda e: e[0])
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    angle = np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1))
    return angle if abs(angle) <= 45 else None  # >45 deg is almost certainly a bad detection


def _rotate_region(frame_bgr: np.ndarray, box: tuple[int, int, int, int], angle_deg: float, pad_factor: float = 0.8) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a generously padded region around box from frame_bgr, rotate it level by
    -angle_deg around the box center, and return (rotated_region, box_in_region_coords).
    Padding is large enough that rotating the box never clips its corners."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = int(pad_factor * max(w, h))
    fy, fx = frame_bgr.shape[:2]
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(fx, x2 + pad), min(fy, y2 + pad)
    region = frame_bgr[ry1:ry2, rx1:rx2]
    local_box = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)
    lcx, lcy = (local_box[0] + local_box[2]) / 2.0, (local_box[1] + local_box[3]) / 2.0
    m = cv2.getRotationMatrix2D((lcx, lcy), -angle_deg, 1.0)
    rotated = cv2.warpAffine(region, m, (region.shape[1], region.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return rotated, local_box


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


def compute_face_embedding(net, face_bgr: np.ndarray) -> np.ndarray:
    # Same VGGFace-backbone preprocessing as predict_race_deepface/predict_gender_deepface:
    # 224x224 BGR, unnormalized [0,255].
    face_resized = cv2.resize(face_bgr, (224, 224)).astype(np.float32)
    emb = net.predict(face_resized[np.newaxis, ...], verbose=0).flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def match_face_identity(embedding: np.ndarray, gallery: dict) -> tuple[str, float] | None:
    """Cosine similarity (paper's 'unsupervised' inner-product metric, embeddings pre-normalized)
    against every enrolled identity; return (name, similarity) for the best match if it clears
    RECOGNITION_COSINE_THRESHOLD, else None."""
    best_name, best_sim = None, -1.0
    for name, gal_emb in gallery.items():
        sim = float(np.dot(embedding, gal_emb))
        if sim > best_sim:
            best_name, best_sim = name, sim
    return (best_name, best_sim) if best_sim >= RECOGNITION_COSINE_THRESHOLD else None


def load_gallery() -> dict:
    if not GALLERY_FILE.exists():
        return {}
    raw = json.loads(GALLERY_FILE.read_text())
    return {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}


def save_gallery(gallery: dict) -> None:
    GALLERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    GALLERY_FILE.write_text(json.dumps({name: vec.tolist() for name, vec in gallery.items()}))


def build_gallery_from_directory(face_net, recognition_net, directory: str | Path) -> dict[str, np.ndarray]:
    """Identity search's directory-matching mode (see README): scan `directory` for image
    files, detect the largest face in each, and embed it with the same VGGFace backbone as
    the enrolled gallery. Returns {display_name: embedding}, display_name being the filename
    stem with underscores turned into spaces (e.g. Barack_Obama.jpg -> "Barack Obama").
    Images with no detected face are skipped silently. Not cached here -- call sites (the web
    app) are expected to cache this themselves since it re-runs face detection + embedding for
    every file on each call."""
    directory = Path(directory)
    gallery: dict[str, np.ndarray] = {}
    if not directory.is_dir():
        return gallery

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        boxes = detect_faces(face_net, image, conf_threshold=0.7)
        if not boxes:
            continue
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        h, w = image.shape[:2]
        y1c, y2c = max(0, y1 - 20), min(y2 + 20, h - 1)
        x1c, x2c = max(0, x1 - 20), min(x2 + 20, w - 1)
        face = image[y1c:y2c, x1c:x2c]
        if face.size == 0:
            continue
        gallery[path.stem.replace("_", " ")] = compute_face_embedding(recognition_net, face)

    return gallery


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


BISENET_HAIR_CLASS = 17  # CelebAMask-HQ 19-class scheme (yakhyo/face-parsing's own utils/prepare_labels.py
# attribute order, 1-indexed after background=0): skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear,
# ear_r, nose, mouth, u_lip, l_lip, neck, neck_l, cloth, hair, hat -- there is NO separate beard/facial-hair
# class; annotators fold facial hair into "hair" too. We approximate facial hair by restricting "hair"-class
# coverage to the lower part of the crop (jaw/chin/mouth), where scalp hair rarely appears in a tight face box.


def predict_facial_hair_bisenet(net, face_bgr: np.ndarray) -> str:
    """BiSeNet (yakhyo/face-parsing) 19-class face parsing. No dedicated beard class exists in
    CelebAMask-HQ's scheme (see BISENET_HAIR_CLASS) -- this reports 'beard' if enough of the
    HAIR class falls in the lower part of the crop, else 'clean-shaven'. Heuristic, not a
    purpose-trained facial-hair classifier."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (512, 512)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - SSRNET_MEAN) / SSRNET_STD
    blob = face_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    net.setInput(blob)
    output = net.forward()  # (1, 19, H, W)
    class_map = output[0].argmax(axis=0)

    h = class_map.shape[0]
    lower = class_map[int(h * 0.6):, :]
    if lower.size == 0:
        return "unknown"
    coverage = float(np.mean(lower == BISENET_HAIR_CLASS))
    return "beard" if coverage >= FACIAL_HAIR_COVERAGE_THRESHOLD else "clean-shaven"


def predict_skin_tone_vgg16(net, face_bgr: np.ndarray) -> str:
    """behra527/Skin-Tone-Classification-model: MobileNetV2 backbone (despite the repo's
    README describing VGG16), RGB, its own idiosyncratic 90x120 (h,w) input,
    keras.applications.mobilenet_v2.preprocess_input scaling."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, SKIN_TONE_INPUT_SIZE), cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = face_rgb / 127.5 - 1.0
    probs = net.predict(face_norm[np.newaxis, ...], verbose=0).flatten()
    return SKIN_TONE_LABELS[int(np.argmax(probs))]


def predict_glasses_mobilenet(net, face_bgr: np.ndarray) -> str:
    """Sorour190/Glasses-Detector's glasses_face224.onnx: MobileNetV3-Large, 224x224 RGB,
    uint8 NHWC input (normalization baked into the ONNX graph itself), outputs a named
    'eyeglasses_prob' scalar already through softmax. License unstated by the source repo
    (flagged in README, same treatment as DAN/SSR-Net)."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    blob = face_rgb[np.newaxis, ...].astype(np.uint8)
    net.setInput(blob)
    prob = float(net.forward("eyeglasses_prob").flatten()[0])
    return "glasses" if prob >= GLASSES_THRESHOLD else "none"


def predict_mask_mobilenetv2(net, face_bgr: np.ndarray) -> str:
    """chandrikadeb7/Face-Mask-Detection: MobileNetV2 backbone (imagenet weights,
    include_top=False) + AveragePooling2D(7,7) + Flatten + Dense(128, relu) + Dropout(0.5) +
    Dense(2, softmax), 224x224 RGB, keras.applications.mobilenet_v2.preprocess_input scaling.
    Class order (sklearn LabelBinarizer, alphabetical) is MASK_LABELS = ['with_mask', 'without_mask']."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = face_rgb / 127.5 - 1.0
    probs = net.predict(face_norm[np.newaxis, ...], verbose=0).flatten()
    return MASK_LABELS[int(np.argmax(probs))]


def _is_skin_hsv(hsv_pixels: np.ndarray) -> np.ndarray:
    """Boolean mask for common skin-tone hue/sat/val ranges in OpenCV HSV (H:0-179).
    Rough heuristic, not a trained model -- used only to exclude forehead skin bleeding
    into the hair-color sample region, not for any skin-tone classification."""
    h, s, v = hsv_pixels[..., 0], hsv_pixels[..., 1], hsv_pixels[..., 2]
    return (h <= 25) & (s >= 30) & (s <= 180) & (v >= 40)


def predict_hair_color_colorimetric(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """Heuristic (not ML): sample the region above the face box, exclude likely-skin
    pixels, take the median color, and bucket by HSV hue/saturation/value into
    HAIR_COLOR_LABELS. Sensitive to lighting/pose/hats -- much rougher than the
    model-backed attributes."""
    x1, y1, x2, y2 = box
    fh, fw = frame_bgr.shape[:2]
    h = y2 - y1
    ry1 = max(0, int(y1 - 0.6 * h))
    ry2 = max(ry1 + 1, y1)
    rx1, rx2 = max(0, x1), min(fw, x2)
    region = frame_bgr[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    skin_mask = _is_skin_hsv(hsv)
    non_skin = hsv[~skin_mask]
    sample = non_skin if non_skin.size > 0 else hsv.reshape(-1, 3)

    med_h, med_s, med_v = (np.median(sample[..., i]) for i in range(3))

    if med_v < 50:
        return "black"
    if med_s < 30:
        return "white" if med_v > 180 else "grey"
    if 8 < med_h < 25 and med_v > 150 and med_s > 60:
        return "blonde"
    if (med_h <= 8 or med_h >= 170) and med_s > 90:
        return "red"
    return "brown"


def predict_eye_color_colorimetric(eye_cascade, face_bgr: np.ndarray) -> str:
    """Heuristic (not ML): locate the largest detected eye via the drowsiness Haar
    cascade, sample the center 40% of its box (avoiding sclera/eyelid), and bucket
    the median HSV into EYE_COLOR_LABELS. Rough by nature -- lighting/pose-sensitive."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    if len(eyes) == 0:
        return "unknown"

    ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
    cx1 = ex + int(ew * 0.3)
    cx2 = ex + int(ew * 0.7)
    cy1 = ey + int(eh * 0.3)
    cy2 = ey + int(eh * 0.7)
    iris_region = face_bgr[cy1:cy2, cx1:cx2]
    if iris_region.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(iris_region, cv2.COLOR_BGR2HSV)
    med_h = float(np.median(hsv[..., 0]))
    med_s = float(np.median(hsv[..., 1]))
    med_v = float(np.median(hsv[..., 2]))

    if med_v < 60:
        return "brown"
    if med_s < 40:
        return "grey"
    if 95 <= med_h <= 130:
        return "blue"
    if 40 <= med_h < 95:
        return "green"
    if 15 <= med_h < 40 and med_s > 100:
        return "amber"
    if med_h < 15 or med_h >= 170:
        return "brown" if med_v < 130 else "hazel"
    return "hazel"


def predict_face_landmarks_mediapipe(landmarker, face_bgr: np.ndarray) -> list[tuple[float, float]] | None:
    """MediaPipe FaceLandmarker (same model instance as Expression's blendshapes backend --
    one model, two features, same pattern as insightface/fairface elsewhere in this file).
    Returns 468 (x, y) points normalized to [0, 1] within face_bgr, or None if no face found."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    return [(lm.x, lm.y) for lm in result.face_landmarks[0]]


def draw_face_landmarks(frame: np.ndarray, points_normalized: list[tuple[float, float]], box: tuple[int, int, int, int]) -> None:
    """Draw face mesh points directly onto frame, scaled into box's pixel extent. Dots only
    (no contour/connection lines) -- 468 points is dense enough to read as a mesh on its own."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    for nx, ny in points_normalized:
        cv2.circle(frame, (x1 + int(nx * w), y1 + int(ny * h)), 1, (255, 0, 255), thickness=-1, lineType=cv2.LINE_AA)


def detect_hand_landmarks_mediapipe(landmarker, frame_bgr: np.ndarray) -> list[list[tuple[int, int]]]:
    """MediaPipe HandLandmarker, whole-frame (hands aren't tied to a detected face box).
    Returns a list of hands, each a list of 21 (x, y) pixel points in frame_bgr's own
    coordinates -- empty list if no hands found (that's how 'if hands are visible' is decided,
    no separate hand-presence check needed)."""
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    return [
        [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand]
        for hand in result.hand_landmarks
    ]


def draw_hand_landmarks(frame: np.ndarray, hands: list[list[tuple[int, int]]]) -> None:
    """Draw each hand's skeleton (joints + connecting bones) directly onto frame, same
    HUD palette as the body pose skeleton."""
    for hand in hands:
        for point_a, point_b in HAND_CONNECTIONS:
            cv2.line(frame, hand[point_a], hand[point_b], (0, 255, 0), 2, cv2.LINE_AA)
        for point in hand:
            cv2.circle(frame, point, 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)


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
    active_recognition: set,
    gallery: dict,
    active_facial_hair: set,
    active_skin_tone: set,
    active_glasses: set,
    active_mask: set,
    active_hair_color: set,
    active_eye_color: set,
    active_pose: set,
    active_face_landmarks: set,
    active_hands: set,
    global_adjustments: dict,
    face_adjustments: dict,
):
    """Detect faces and run inference for whichever model keys are active per feature.
    Multiple active models for the same feature (e.g. active_age = {"caffe", "ssrnet"})
    all run and are shown together. No Streamlit calls (safe for background threads).

    global_adjustments apply to the whole frame first, before face detection even runs --
    every output derived from this call (the annotated image, every face crop, every
    classification) sees the adjusted pixels. face_adjustments apply again, per detected
    face, to that face's own crop only, after detection but before classification -- they
    affect just that one face's thumbnail/attributes, not the shared frame or other faces."""
    if global_adjustments and any(global_adjustments.values()):
        frame = apply_image_adjustments(frame, global_adjustments)

    annotated_frame = frame.copy()
    face_boxes = detect_faces(models.face_net, frame, conf_threshold)
    cropped_faces = []
    any_drowsy = False

    pose_detected = False
    pose_net = models.pose_nets.get("mpi")
    if pose_net is not None and "mpi" in active_pose:
        pose_points = detect_pose_mpi(pose_net, frame)
        if sum(p is not None for p in pose_points) >= MIN_POSE_POINTS:
            pose_detected = True
            draw_pose_skeleton(annotated_frame, pose_points)

    hands_detected = False
    hand_net = models.hand_nets.get("mediapipe")
    if hand_net is not None and "mediapipe" in active_hands:
        hands = detect_hand_landmarks_mediapipe(hand_net, frame)
        if hands:
            hands_detected = True
            draw_hand_landmarks(annotated_frame, hands)

    need_blob227 = ("caffe" in active_age and "caffe" in models.age_nets) or \
                   ("caffe" in active_gender and "caffe" in models.gender_nets)

    eye_cascade = models.drowsiness_nets.get("haarcascade")

    for idx, (x1, y1, x2, y2) in enumerate(face_boxes, 1):
        # Correct in-plane roll (tilted head) before cropping/classifying, using the same
        # eye cascade as drowsiness detection -- no new model/dependency. crop_frame/cx*/cy*
        # are the rotation-corrected region+box; x1..y2 stay untouched for the box overlay
        # drawn on annotated_frame further below.
        crop_frame, (cx1, cy1, cx2, cy2) = frame, (x1, y1, x2, y2)
        if eye_cascade is not None:
            probe = frame[max(0, y1 - 20):min(y2 + 20, frame.shape[0] - 1), max(0, x1 - 20):min(x2 + 20, frame.shape[1] - 1)]
            angle = _estimate_roll_angle(probe, eye_cascade) if probe.size else None
            if angle is not None and abs(angle) > 3:  # skip work for near-level faces
                crop_frame, (cx1, cy1, cx2, cy2) = _rotate_region(frame, (x1, y1, x2, y2), angle)

        y1_crop = max(0, cy1 - 20)
        y2_crop = min(cy2 + 20, crop_frame.shape[0] - 1)
        x1_crop = max(0, cx1 - 20)
        x2_crop = min(cx2 + 20, crop_frame.shape[1] - 1)

        face = crop_frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        if face_adjustments and any(face_adjustments.values()):
            face = apply_image_adjustments(face, face_adjustments)

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
                value = predict_age_fairface(net, crop_frame, (cx1, cy1, cx2, cy2))
            elif key == "dex":
                value = predict_age_dex(net, face)
            elif key == "mivolo":
                value = predict_age_mivolo(net, face)
            else:
                value = predict_age_insightface(net, crop_frame, (cx1, cy1, cx2, cy2))
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
                value = predict_gender_fairface(net, crop_frame, (cx1, cy1, cx2, cy2))
            elif key == "mivolo":
                value = predict_gender_mivolo(net, face)
            else:
                value = predict_gender_insightface(net, crop_frame, (cx1, cy1, cx2, cy2))
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
            value = predict_race_fairface(net, crop_frame, (cx1, cy1, cx2, cy2)) if key == "fairface" else predict_race_deepface(net, face)
            race_pairs.append((key, value))

        expression_pairs = []
        for key in active_expression:
            net = models.expression_nets.get(key)
            if net is None:
                continue
            value = predict_expression_blendshapes(net, face)
            expression_pairs.append((key, value))

        recognition_pairs = []
        face_embedding = None
        for key in active_recognition:
            net = models.recognition_nets.get(key)
            if net is None:
                continue
            face_embedding = compute_face_embedding(net, face)
            match = match_face_identity(face_embedding, gallery)
            value = f"{match[0]} ({match[1] * 100:.0f}%)" if match else "UNKNOWN"
            recognition_pairs.append((key, value))

        facial_hair_pairs = []
        for key in active_facial_hair:
            net = models.facial_hair_nets.get(key)
            if net is None:
                continue
            value = predict_facial_hair_bisenet(net, face)
            facial_hair_pairs.append((key, value))

        skin_tone_pairs = []
        for key in active_skin_tone:
            net = models.skin_tone_nets.get(key)
            if net is None:
                continue
            value = predict_skin_tone_vgg16(net, face)
            skin_tone_pairs.append((key, value))

        glasses_pairs = []
        for key in active_glasses:
            net = models.glasses_nets.get(key)
            if net is None:
                continue
            value = predict_glasses_mobilenet(net, face)
            glasses_pairs.append((key, value))

        mask_pairs = []
        for key in active_mask:
            net = models.mask_nets.get(key)
            if net is None:
                continue
            value = predict_mask_mobilenetv2(net, face)
            mask_pairs.append((key, value))

        hair_color_pairs = []
        for key in active_hair_color:
            if key not in models.hair_color_nets:
                continue
            value = predict_hair_color_colorimetric(crop_frame, (cx1, cy1, cx2, cy2))
            hair_color_pairs.append((key, value))

        eye_color_pairs = []
        for key in active_eye_color:
            net = models.eye_color_nets.get(key)
            if net is None:
                continue
            value = predict_eye_color_colorimetric(net, face)
            eye_color_pairs.append((key, value))

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

        landmarks_net = models.face_landmarks_nets.get("blendshapes")
        if landmarks_net is not None and "blendshapes" in active_face_landmarks:
            landmark_points = predict_face_landmarks_mediapipe(landmarks_net, face)
            if landmark_points is not None:
                draw_face_landmarks(annotated_frame, landmark_points, (x1, y1, x2, y2))

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
            "identity": _format_results(recognition_pairs),
            "facial_hair": _format_results(facial_hair_pairs),
            "skin_tone": _format_results(skin_tone_pairs),
            "glasses": _format_results(glasses_pairs),
            "mask": _format_results(mask_pairs),
            "hair_color": _format_results(hair_color_pairs),
            "eye_color": _format_results(eye_color_pairs),
            "embedding": face_embedding.tolist() if face_embedding is not None else None,
            "status": status,
            "drowsy": face_drowsy if drowsy_pairs else None,
        })

    return annotated_frame, cropped_faces, any_drowsy, bool(face_boxes), pose_detected, hands_detected
