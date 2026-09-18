"""Model loading and per-face prediction logic for the Streamlit app (src/app.py).

This module only exists to keep app.py itself from growing unbounded as
more model backends are added.
"""
from __future__ import annotations

import json
import os
import hashlib
import random
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from .liveness import (
        LivenessTracker,
        assess_static_liveness,
        blink_score_from_landmarker,
        texture_artifact_score,
    )
    from .body_composition import BODY_COMPOSITION_MODEL_OPTIONS, estimate_face_composition
except ImportError:  # app.py runs with src/ on sys.path in the container
    from liveness import (
        LivenessTracker,
        assess_static_liveness,
        blink_score_from_landmarker,
        texture_artifact_score,
    )
    from body_composition import BODY_COMPOSITION_MODEL_OPTIONS, estimate_face_composition

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

try:
    import onnxruntime
    ONNXRUNTIME_SUPPORTED = True
except ImportError:
    ONNXRUNTIME_SUPPORTED = False

try:
    from nets.deep3d_recon import (
        build_deep3d_recon_model, ParametricFaceModel, load_lm3d_template,
        landmarks_5pt_from_mediapipe, reconstruct_face_3d, mesh_to_obj_str,
    )
    TORCHVISION_SUPPORTED = True
except ImportError:
    TORCHVISION_SUPPORTED = False

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
YOLO_FACE_MODEL = MODEL_DIR / "yolov8n_face.onnx"
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
HSEMOTION_MODEL = MODEL_DIR / "hsemotion_enet_b0_8_best_vgaf.onnx"
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
BFM_DIR = MODEL_DIR / "BFM"
DEEP3D_RECON_MODEL = MODEL_DIR / "deep3d_recon_resnet50.pth"  # gated, not bundled -- see README
BFM_MODEL_PATH = BFM_DIR / "BFM_model_front.mat"  # gated, not bundled -- see README
BFM_LM3D_PATH = BFM_DIR / "similarity_Lm3D_all.mat"  # bundled (MIT, small landmark template)

MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LABELS_DAN = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
EMOTION_LABELS_EFFICIENTNET = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_MINI_XCEPTION = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_FERPLUS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
EMOTION_LABELS_HSEMOTION = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
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
LBPH_GALLERY_DIR = BASE_DIR / "gallery" / "lbph"
LBPH_FACE_SIZE = (200, 200)
LBPH_CONFIDENCE_THRESHOLD = 80.0  # LBPH's own distance metric -- LOWER is a better match (opposite of cosine similarity)
KNOWN_PEOPLE_DIR = BASE_DIR / "known_people"  # bundled reference photos for identity search (see README)
IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FACES_DB_FILE = BASE_DIR / "db" / "faces.db"  # sparse-column SQLite database, see save_face()
FACES_DIR = BASE_DIR / "faces"  # saved face crops (color, as-classified), one per saved face
EIGEN_DIR = BASE_DIR / "eigen"  # saved faces' grayscale/zoomed eigenfaces training images
EIGEN_FACE_SIZE = (100, 100)  # (width, height) every eigen/ image is normalized to
EIGENFACE_DISTANCE_THRESHOLD = 3000.0  # untuned heuristic (see match_face_eigenfaces docstring)

# Model keys per feature, in quickest-to-build order (first = default).
# Must match the numbered options in build-and-run.sh and the Dockerfile ARGs.
AGE_MODEL_OPTIONS = ["caffe", "insightface", "ssrnet", "fairface", "dex", "mivolo"]
GENDER_MODEL_OPTIONS = ["caffe", "insightface", "deepface", "fairface", "mivolo"]
FAIRFACE_AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
EMOTION_MODEL_OPTIONS = ["efficientnet", "ferplus", "mini_xception", "dan", "hsemotion"]
DROWSINESS_MODEL_OPTIONS = ["haarcascade"]
RACE_MODEL_OPTIONS = ["fairface", "deepface"]
EXPRESSION_MODEL_OPTIONS = ["blendshapes"]
LIVENESS_MODEL_OPTIONS = ["mediapipe"]
RECOGNITION_MODEL_OPTIONS = ["vggface", "lbph"]
FACE_DETECTOR_OPTIONS = ["ssd", "yolo"]  # ssd is the original required detector, always on
IOU_TRACKING_THRESHOLD = 0.3  # greedy-match a track to a detection only above this IoU
TRACKING_MAX_MISSED_FRAMES = 10  # frames a track survives with zero matching detections
# (brief occlusion) before its ID is dropped and freed for reuse
YOLO_FACE_INPUT_SIZE = 640
YOLO_FACE_STRIDES = (8, 16, 32)
YOLO_FACE_IOU_THRESHOLD = 0.45
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
RECONSTRUCTION_3D_MODEL_OPTIONS = ["deep3d"]
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


# --- Thread safety for shared model instances (#19, #B) -----------------------------------
# cv2.dnn.Net, cv2.CascadeClassifier, Keras/TF models, and the MediaPipe Tasks API are not
# documented as safe for concurrent setInput/forward/predict/detect calls on the SAME cached
# instance -- st.cache_resource shares one Models instance across every face, frame, and
# Streamlit session. analyze_frame() runs each face's per-feature predictions concurrently
# (see _run_feature_tasks below); this lock, keyed by the shared net object's identity, is
# what makes concurrent calls onto the same net safe (they serialize) while calls onto
# DIFFERENT nets still run in true parallel. Torch nn.Module.forward (ssrnet/dan) and
# onnxruntime InferenceSession.run (glasses, yolo face detector) are both documented safe for
# concurrent inference on one instance, so those are intentionally left unlocked.
_NET_LOCKS: dict[int, threading.Lock] = {}
_NET_LOCKS_GUARD = threading.Lock()


def _lock_for(net) -> threading.Lock | "nullcontext[None]":
    """Return a lock keyed by net's identity, or a no-op if net is just a boolean presence
    marker (e.g. hair_color's "colorimetric" / recognition's "lbph" entries in Models, which
    aren't a real shared native object)."""
    if net is None or isinstance(net, bool):
        return nullcontext()
    key = id(net)
    lock = _NET_LOCKS.get(key)
    if lock is None:
        with _NET_LOCKS_GUARD:
            lock = _NET_LOCKS.setdefault(key, threading.Lock())
    return lock


# Shared across the process (and every Streamlit session) -- per-feature tasks are short-lived
# native calls (cv2.dnn/TF/torch all release the GIL during their own compute), so a modest
# pool sized off the CPU count lets independent features (different nets) genuinely overlap
# without oversubscribing a CPU-only deployment.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 4)), thread_name_prefix="inference")


@dataclass
class Models:
    face_net: cv2.dnn.Net
    age_nets: dict = field(default_factory=dict)
    gender_nets: dict = field(default_factory=dict)
    emotion_nets: dict = field(default_factory=dict)
    drowsiness_nets: dict = field(default_factory=dict)
    race_nets: dict = field(default_factory=dict)
    expression_nets: dict = field(default_factory=dict)
    liveness_nets: dict = field(default_factory=dict)
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
    reconstruction_3d_nets: dict = field(default_factory=dict)
    yolo_face_nets: dict = field(default_factory=dict)
    gaze_nets: dict = field(default_factory=dict)
    body_composition_nets: dict = field(default_factory=dict)

    @property
    def offline_features(self) -> list[str]:
        return [
            name for name, nets in [
                ("AGE", self.age_nets), ("GENDER", self.gender_nets),
                ("EMOTION", self.emotion_nets), ("DROWSINESS", self.drowsiness_nets),
                ("RACE", self.race_nets), ("EXPRESSION", self.expression_nets),
                ("LIVENESS", self.liveness_nets),
                ("GAZE", self.gaze_nets),
                ("RECOGNITION", self.recognition_nets), ("FACIAL_HAIR", self.facial_hair_nets),
                ("SKIN_TONE", self.skin_tone_nets), ("GLASSES", self.glasses_nets),
                ("MASK", self.mask_nets), ("HAIR_COLOR", self.hair_color_nets),
                ("EYE_COLOR", self.eye_color_nets), ("COLORIZATION", self.colorization_nets),
                ("POSE", self.pose_nets), ("FACE_LANDMARKS", self.face_landmarks_nets),
                ("HANDS", self.hand_nets), ("RECONSTRUCTION_3D", self.reconstruction_3d_nets),
                ("FACE_DETECTOR_YOLO", self.yolo_face_nets),
                ("BODY_COMPOSITION", self.body_composition_nets),
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
    if hasattr(cv2, "face"):
        recognition_nets["lbph"] = True  # no pretrained weights -- trains fresh from gallery/lbph/ on demand

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
    if HSEMOTION_MODEL.exists():
        emotion_nets["hsemotion"] = cv2.dnn.readNetFromONNX(str(HSEMOTION_MODEL))

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
    liveness_nets = {}
    face_landmarks_nets = {}
    gaze_nets = {}
    body_composition_nets = {}
    if MEDIAPIPE_SUPPORTED and BLENDSHAPES_MODEL.exists():
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(BLENDSHAPES_MODEL)),
            output_face_blendshapes=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        expression_nets["blendshapes"] = landmarker
        liveness_nets["mediapipe"] = landmarker
        face_landmarks_nets["blendshapes"] = landmarker  # same model instance, two features
        gaze_nets["mediapipe"] = landmarker
        body_composition_nets[BODY_COMPOSITION_MODEL_OPTIONS[0]] = landmarker

    facial_hair_nets = {}
    if BISENET_MODEL.exists():
        facial_hair_nets["bisenet"] = cv2.dnn.readNetFromONNX(str(BISENET_MODEL))

    skin_tone_nets = {}
    if TF_SUPPORTED and SKIN_TONE_MODEL.exists():
        skin_tone_nets["mobilenetv2"] = build_skin_tone_model(str(SKIN_TONE_MODEL))

    glasses_nets = {}
    if ONNXRUNTIME_SUPPORTED and GLASSES_MODEL.exists():
        glasses_nets["mobilenet"] = onnxruntime.InferenceSession(str(GLASSES_MODEL), providers=["CPUExecutionProvider"])

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

    reconstruction_3d_nets = {}
    if TORCHVISION_SUPPORTED and DEEP3D_RECON_MODEL.exists() and BFM_MODEL_PATH.exists() and BFM_LM3D_PATH.exists():
        recon_net = build_deep3d_recon_model(str(DEEP3D_RECON_MODEL))
        bfm_model = ParametricFaceModel(str(BFM_MODEL_PATH))
        lm3d_template = load_lm3d_template(str(BFM_DIR))
        reconstruction_3d_nets["deep3d"] = (recon_net, bfm_model, lm3d_template)

    yolo_face_nets = {}
    if ONNXRUNTIME_SUPPORTED and YOLO_FACE_MODEL.exists():
        yolo_face_nets["yolo"] = onnxruntime.InferenceSession(str(YOLO_FACE_MODEL), providers=["CPUExecutionProvider"])

    return Models(
        face_net, age_nets, gender_nets, emotion_nets, drowsiness_nets, race_nets, expression_nets, liveness_nets, recognition_nets,
        facial_hair_nets, skin_tone_nets, glasses_nets, mask_nets, hair_color_nets, eye_color_nets, colorization_nets,
        pose_nets, face_landmarks_nets, hand_nets, reconstruction_3d_nets, yolo_face_nets,
        gaze_nets, body_composition_nets,
    )


def _yolo_letterbox(image: np.ndarray, target_size: int = YOLO_FACE_INPUT_SIZE) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Vendored from yakhyo/yolov8-face-onnx-inference's utils/general.py: resize preserving
    aspect ratio + pad to a square target_size."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (target_size - new_w) / 2, (target_size - new_h) / 2
    top, bottom = int(dh), int(target_size - new_h - int(dh))
    left, right = int(dw), int(target_size - new_w - int(dw))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, (dw, dh)


def _yolo_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def detect_faces_yolo(session, frame: np.ndarray, conf_threshold: float = 0.5) -> list[list[int]]:
    """YOLOv8-Face (yakhyo/yolov8-face-onnx-inference, weights unlicensed -- see README) via
    onnxruntime -- cv2.dnn cannot load this ONNX export (verified: fails identically on both
    OpenCV 4.10 and 5.0 with a mixed-dtype Cast/Mul error in its DFL decode subgraph), so this
    feature needs onnxruntime specifically rather than this repo's usual cv2.dnn ONNX
    convention. Decodes the raw 3-feature-map DFL output (strides 8/16/32) matching upstream's
    own models/yolov8.py exactly, but only for boxes/scores -- the 5-point facial landmarks
    this model also predicts aren't decoded since nothing downstream uses them. Returns boxes
    in the same [x1, y1, x2, y2] int-list contract as detect_faces(), so it's a drop-in swap."""
    letterboxed, scale, (dw, dh) = _yolo_letterbox(frame)
    blob = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    all_boxes, all_scores = [], []
    for pred, stride in zip(outputs, YOLO_FACE_STRIDES):
        _, channels, h, w = pred.shape
        pred = pred.reshape(1, channels, -1).transpose(0, 2, 1)[0]  # (H*W, 80)

        grid_y, grid_x = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        grid_x, grid_y = grid_x.flatten(), grid_y.flatten()

        bbox_pred = pred[:, :64].reshape(-1, 4, 16)
        bbox_dist = _yolo_softmax(bbox_pred, axis=-1) @ np.arange(16)
        cls_conf = 1 / (1 + np.exp(-pred[:, 64]))  # sigmoid

        x1 = (grid_x - bbox_dist[:, 0]) * stride
        y1 = (grid_y - bbox_dist[:, 1]) * stride
        x2 = (grid_x + bbox_dist[:, 2]) * stride
        y2 = (grid_y + bbox_dist[:, 3]) * stride
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
        all_scores.append(cls_conf)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    mask = scores >= conf_threshold
    boxes, scores = boxes[mask], scores[mask]
    if len(boxes) == 0:
        return []

    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]  # cv2.dnn.NMSBoxes wants (x, y, w, h)
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, YOLO_FACE_IOU_THRESHOLD)
    if len(keep) == 0:
        return []
    boxes = boxes[np.array(keep).flatten()]

    # Undo the letterbox padding/scale to map back to frame's own coordinates.
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= scale
    frame_h, frame_w = frame.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h)

    return boxes.astype(int).tolist()


def detect_faces(net: cv2.dnn.Net, frame: np.ndarray, conf_threshold: float = 0.7) -> list[list[int]]:
    """Detect faces and return bounding box limits."""
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    with _lock_for(net):
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


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Standard intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Track:
    box: tuple[int, int, int, int]
    missed_frames: int = 0


class FaceTracker:
    """#2: greedy IoU-based multi-face tracker. Assigns a stable integer ID to each detected
    face box across consecutive analyze_frame() calls on the SAME video stream, so a face
    keeps its identity as it moves instead of every frame renumbering faces 1..N by detection
    order (which is what analyze_frame() does on its own, per-frame, with no tracker passed).

    Matching is frame-to-frame only, no motion prediction: each track's last known box is
    compared by IoU against this frame's detections, and the highest-IoU pairs at or above
    IOU_TRACKING_THRESHOLD are greedily accepted one-to-one (highest IoU first, each track and
    each detection used at most once). A track that matches no detection this frame is kept,
    not dropped immediately -- only after TRACKING_MAX_MISSED_FRAMES consecutive unmatched
    frames is it deleted and its ID freed. This is what "survives brief occlusion" means here:
    a face that's blocked (or missed by the detector) for a few frames keeps its ID as long as
    it reappears close to where it was last seen within that window.

    One instance tracks one video stream. Safe to call update()/reset() from a different
    thread than the one that created it (e.g. streamlit-webrtc's own callback thread) via an
    internal lock -- this is the ONLY per-frame state analyze_frame() has ever needed, so the
    lock is scoped tightly to this class rather than adding any shared mutable state to
    analyze_frame() itself, which stays otherwise stateless."""

    def __init__(self, iou_threshold: float = IOU_TRACKING_THRESHOLD, max_missed_frames: int = TRACKING_MAX_MISSED_FRAMES):
        self._lock = threading.Lock()
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, boxes: list[tuple[int, int, int, int]]) -> list[int]:
        """Match this frame's detections against existing tracks. Returns one stable track ID
        per box, in the same order as `boxes`."""
        with self._lock:
            candidates = []  # (iou, track_id, detection_index)
            for track_id, track in self._tracks.items():
                for det_index, box in enumerate(boxes):
                    iou = _box_iou(track.box, tuple(box))
                    if iou >= self._iou_threshold:
                        candidates.append((iou, track_id, det_index))
            candidates.sort(key=lambda c: c[0], reverse=True)

            track_for_detection: dict[int, int] = {}
            used_tracks: set[int] = set()
            for iou, track_id, det_index in candidates:
                if track_id in used_tracks or det_index in track_for_detection:
                    continue
                track_for_detection[det_index] = track_id
                used_tracks.add(track_id)

            result_ids = []
            for det_index, box in enumerate(boxes):
                track_id = track_for_detection.get(det_index)
                if track_id is None:
                    track_id = self._next_id
                    self._next_id += 1
                self._tracks[track_id] = _Track(box=tuple(box), missed_frames=0)
                result_ids.append(track_id)

            handled_this_frame = set(result_ids)
            for track_id in list(self._tracks):
                if track_id in handled_this_frame:
                    continue
                track = self._tracks[track_id]
                track.missed_frames += 1
                if track.missed_frames > self._max_missed_frames:
                    del self._tracks[track_id]

            return result_ids

    def reset(self) -> None:
        """Drop every tracked face and restart ID numbering from 1. Call this when a video
        stream (re)starts -- a new stream has no relationship to the previous one's faces, so
        continuing the old numbering (or keeping stale tracks alive) would be misleading."""
        with self._lock:
            self._tracks.clear()
            self._next_id = 1


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

    with _lock_for(net):
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
    with _lock_for(net):
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


# --- #10: voice+face multimodal fusion (webcam LIVE mode only) --------------------------------
# Design decision: this repo has shipped features before whose only known trained-weights
# source was broken/gated/unlicensed ("no working weights shipped" is an established pattern
# here) -- a speech-emotion-recognition model is
# exactly that kind of risk, and there's no way to vet a specific checkpoint's license/quality
# from inside this session. So there is no voice EMOTION classifier here. Instead, voice
# contributes one honestly-scoped signal -- short-term loudness (RMS energy), a heuristic
# exactly like this file's existing hair_color/eye_color "colorimetric" functions (not ML,
# documented as rough) -- and fusion means cross-checking that signal against the face's
# already-computed emotion label, not inventing a new blended "emotion" that would imply
# accuracy neither signal actually has.
#
# Scope: webcam LIVE mode only. Upload/snapshot are single static images with no audio
# alongside them in this app, so voice fusion has nothing to sync against there.
#
# Sync model: audio arrives via its own streamlit-webrtc callback, on its own thread, at its
# own rate -- independent of video_frame_callback's rate. This is NOT frame-accurate
# lip-sync; it answers "how loud has the mic been for the last ~AUDIO_AROUSAL_WINDOW_SECONDS",
# which the video callback reads at whatever instant a video frame arrives. That's the right
# granularity for "is this person currently speaking with energy" -- finer sync isn't
# meaningful for a loudness-only signal anyway.
AUDIO_AROUSAL_WINDOW_SECONDS = 1.5
AUDIO_AROUSAL_QUIET_RMS = 0.02   # below this normalized RMS: treat as silence/background noise
AUDIO_AROUSAL_LOUD_RMS = 0.15    # at/above this: "loud" rather than just "speaking"
# Coarse arousal bucketing per emotion label, independent of which emotion backend produced it
# (DAN/EfficientNet/FERPlus/mini_xception all use different label spellings -- this maps every
# label spelling this app can produce). Loosely follows Russell's circumplex model (high-arousal
# vs. low-arousal quadrants) -- a cross-check heuristic, not a validated psychological measure.
EMOTION_HIGH_AROUSAL_LABELS = {
    "happy", "happiness", "surprise", "anger", "angry", "fear", "disgust",
}
EMOTION_LOW_AROUSAL_LABELS = {"neutral", "sad", "sadness"}


def audio_frame_to_mono_float(samples: np.ndarray) -> np.ndarray:
    """Normalize whatever shape/dtype PyAV's AudioFrame.to_ndarray() handed back into a 1-D
    float32 array in [-1, 1]: mixes multi-channel audio down to mono (av's ndarray is
    channel-major for planar formats, i.e. shape (channels, samples), so averaging axis 0
    is correct there; a already-1-D array is left as-is)."""
    if np.issubdtype(samples.dtype, np.integer):
        max_value = float(np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / max_value
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2 and samples.shape[0] <= 8:
        samples = samples.mean(axis=0)
    return samples.flatten()


def classify_voice_arousal(rms: float) -> str:
    """Bucket a normalized RMS energy level into QUIET / SPEAKING / LOUD."""
    if rms >= AUDIO_AROUSAL_LOUD_RMS:
        return "LOUD"
    if rms >= AUDIO_AROUSAL_QUIET_RMS:
        return "SPEAKING"
    return "QUIET"


def _emotion_arousal_category(emotion_label: str) -> str | None:
    label = emotion_label.lower()
    if label in EMOTION_HIGH_AROUSAL_LABELS:
        return "high"
    if label in EMOTION_LOW_AROUSAL_LABELS:
        return "low"
    return None  # composite labels (e.g. blendshapes' "jawOpen 0.82, ...") aren't mapped


def fuse_voice_and_emotion(voice_arousal: str, emotion_label: str) -> str | None:
    """Cross-check voice loudness against one face's emotion label. Returns "consistent" if
    the two roughly agree on high/low arousal, "inconsistent" if they roughly disagree, or
    None if emotion_label isn't one this heuristic can categorize (see
    _emotion_arousal_category) -- None means "no opinion", not "disagreement"."""
    emotion_category = _emotion_arousal_category(emotion_label)
    if emotion_category is None:
        return None
    voice_category = "low" if voice_arousal == "QUIET" else "high"
    return "consistent" if voice_category == emotion_category else "inconsistent"


class VoiceFaceFusion:
    """Thread-safe rolling audio buffer for #10. One instance per webcam LIVE stream: the
    audio_frame_callback (streamlit-webrtc's own audio thread) appends samples via
    ingest_audio(); the video_frame_callback (a different thread, different rate) calls
    current_arousal() to read "how loud has the mic been recently" at the instant a video
    frame arrives. Same "module-level/st.cache_resource-held thread-safe buffer" pattern as
    #9's emotion-over-time buffer and #2's FaceTracker -- this is the third feature needing
    exactly this shape of cross-thread state, all solved the same way for consistency."""

    def __init__(self, window_seconds: float = AUDIO_AROUSAL_WINDOW_SECONDS):
        self._lock = threading.Lock()
        self._window_seconds = window_seconds
        self._samples: list[np.ndarray] = []
        self._buffered_seconds = 0.0
        self._sample_rate: int | None = None
        self._latest_status: dict | None = None

    def ingest_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        """Append one audio frame's samples (mono float32, see audio_frame_to_mono_float) to
        the rolling window, dropping the oldest samples once the window exceeds
        window_seconds -- bounds memory the same way #9's buffer caps its sample count."""
        if samples.size == 0 or sample_rate <= 0:
            return
        with self._lock:
            self._sample_rate = sample_rate
            self._samples.append(samples)
            self._buffered_seconds += samples.size / sample_rate
            while self._buffered_seconds > self._window_seconds and len(self._samples) > 1:
                oldest = self._samples.pop(0)
                self._buffered_seconds -= oldest.size / self._sample_rate

    def current_arousal(self) -> str:
        """Return QUIET / SPEAKING / LOUD for the current buffered window, or "QUIET" if no
        audio has been ingested yet (e.g. mic permission not granted, or fusion just enabled)."""
        with self._lock:
            if not self._samples:
                return "QUIET"
            window = np.concatenate(self._samples)
        rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
        return classify_voice_arousal(rms)

    def set_latest_status(self, status: dict) -> None:
        """Stash the video callback's most recent fusion result for the main Streamlit script
        thread to read on its next rerun (see the module docstring's "no per-face text burned
        onto the shared image" convention -- this is the same shared-state-read-on-rerun
        pattern as #9's buffer, applied to a single status dict instead of a time series)."""
        with self._lock:
            self._latest_status = status

    def get_latest_status(self) -> dict | None:
        with self._lock:
            return self._latest_status

    def reset(self) -> None:
        with self._lock:
            self._samples = []
            self._buffered_seconds = 0.0
            self._latest_status = None


# --- Rectangle-select geometric transforms (ideas/transform.md, ideas/geo-transform.md) ---
# Applied to an arbitrary user-selected sub-rectangle of the whole image, independent of face
# detection -- these operate on any region, not just faces.
GEOMETRIC_TRANSFORM_OPTIONS = ["translate", "reflect", "rotate", "scale", "shear"]


def crop_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop an arbitrary rectangle, clamped to frame bounds."""
    h, w = frame.shape[:2]
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    return frame[y1:y2, x1:x2]


def apply_geometric_transform(region: np.ndarray, transform_type: str, **params) -> np.ndarray:
    """Apply one geometric transform to a cropped region. Matches the matrices in
    ideas/transform.md / ideas/geo-transform.md directly (translation, reflection, rotation,
    scaling, shearing)."""
    h, w = region.shape[:2]

    if transform_type == "translate":
        dx, dy = params.get("dx", 0), params.get("dy", 0)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(region, m, (w, h))

    if transform_type == "reflect":
        axis = params.get("axis", "horizontal")
        return cv2.flip(region, 1 if axis == "horizontal" else 0)

    if transform_type == "rotate":
        angle, scale = params.get("angle", 0.0), params.get("scale", 1.0)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        return cv2.warpAffine(region, m, (w, h))

    if transform_type == "scale":
        fx, fy = params.get("fx", 1.0), params.get("fy", 1.0)
        interp = cv2.INTER_AREA if fx < 1 and fy < 1 else cv2.INTER_CUBIC
        return cv2.resize(region, None, fx=fx, fy=fy, interpolation=interp)

    if transform_type == "shear":
        axis, factor = params.get("axis", "x"), params.get("factor", 0.0)
        if axis == "x":
            out_w, out_h = max(1, int(np.ceil(w + abs(factor) * h))), h
            m = np.float32([[1, factor, max(0, -factor * h)], [0, 1, 0], [0, 0, 1]])
        else:
            out_w, out_h = w, max(1, int(np.ceil(h + abs(factor) * w)))
            m = np.float32([[1, 0, 0], [factor, 1, max(0, -factor * w)], [0, 0, 1]])
        return cv2.warpPerspective(region, m, (out_w, out_h))

    raise ValueError(f"Unknown transform_type: {transform_type}")


# --- Per-face one-click image operations (ideas/intensity.md, enhance.md, sharpen.md,
# color-correct.md, denoise.md, bilateral-filter.md, wavelet-denoise.md) ---
IMAGE_OP_OPTIONS = ["intensity", "enhance", "sharpen", "color_correct", "denoise", "bilateral_filter", "wavelet_denoise"]
INTENSITY_METHODS = ["negative", "log", "gamma", "contrast_stretch"]
SHARPEN_METHODS = ["laplacian", "high_boost"]
DENOISE_METHODS = ["gaussian", "median", "nlm"]


def apply_intensity_transform(face_bgr: np.ndarray, method: str = "gamma", gamma: float = 0.7, r1: int = 70, s1: int = 0, r2: int = 140, s2: int = 255) -> np.ndarray:
    """Intensity transformations (ideas/intensity.md): negative (s = L-1-r), log (s =
    c*log(1+r), expands dark detail), gamma/power-law (s = c*r^gamma, gamma<1 brightens,
    gamma>1 darkens), or piecewise-linear contrast stretching."""
    img = face_bgr.astype(np.float32)

    if method == "negative":
        return (255 - img).astype(np.uint8)

    if method == "log":
        c = 255.0 / np.log(1 + img.max()) if img.max() > 0 else 1.0
        return np.clip(c * np.log(1 + img), 0, 255).astype(np.uint8)

    if method == "gamma":
        return np.clip(255.0 * (img / 255.0) ** gamma, 0, 255).astype(np.uint8)

    if method == "contrast_stretch":
        out = np.empty_like(img)
        low = img <= r1
        mid = (img > r1) & (img <= r2)
        high = img > r2
        out[low] = (s1 / r1) * img[low] if r1 else 0
        out[mid] = ((s2 - s1) / (r2 - r1)) * (img[mid] - r1) + s1
        out[high] = ((255 - s2) / (255 - r2)) * (img[high] - r2) + s2
        return np.clip(out, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown intensity method: {method}")


def apply_enhance(face_bgr: np.ndarray, brightness: float = 10.0, contrast: float = 1.3) -> np.ndarray:
    """General enhancement (ideas/enhance.md): brightness/contrast adjustment
    (cv2.addWeighted) followed by luminance histogram equalization (LAB's L channel, so color
    isn't distorted the way equalizing each BGR channel independently would)."""
    adjusted = cv2.addWeighted(face_bgr, contrast, np.zeros_like(face_bgr), 0, brightness)
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_sharpen(face_bgr: np.ndarray, method: str = "laplacian") -> np.ndarray:
    """Sharpening (ideas/sharpen.md): basic Laplacian kernel, or a stronger high-boost filter
    (larger center coefficient -> more pronounced edge emphasis)."""
    if method == "high_boost":
        kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(face_bgr, -1, kernel)


def apply_color_correct(face_bgr: np.ndarray) -> np.ndarray:
    """Color correction (ideas/color-correct.md): BGR -> LAB, CLAHE (adaptive histogram
    equalization) on the L channel only, back to BGR -- corrects contrast/color balance
    without the color-shifting artifacts of equalizing in RGB/BGR space directly."""
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_denoise(face_bgr: np.ndarray, method: str = "nlm") -> np.ndarray:
    """Classical denoising (ideas/denoise.md): Gaussian (smooths Gaussian/sensor noise,
    slightly blurs edges), median (strong against salt-and-pepper/impulse noise, better edge
    preservation), or Non-Local Means (searches the whole image for similar patches, best
    texture preservation, slowest). CNN/GAN-based methods from the same doc are skipped --
    they need trained weights this repo doesn't have a source for (same category of gap as
    skin_tone/deep3d elsewhere in this app)."""
    if method == "gaussian":
        return cv2.GaussianBlur(face_bgr, (5, 5), 1.5)
    if method == "median":
        return cv2.medianBlur(face_bgr, 5)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(face_bgr, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    raise ValueError(f"Unknown denoise method: {method}")


def apply_bilateral_filter(face_bgr: np.ndarray, diameter: int = 15, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Edge-preserving denoise (ideas/bilateral-filter.md.md): weights nearby pixels by both
    spatial closeness and intensity similarity, so edges (large intensity jumps) are preserved
    while flat/noisy regions get smoothed -- unlike Gaussian blur, which smooths everything
    uniformly regardless of edges."""
    return cv2.bilateralFilter(face_bgr, diameter, sigma_color, sigma_space)


def apply_wavelet_denoise(face_bgr: np.ndarray, threshold: float = 0.05, wavelet: str = "db1") -> np.ndarray:
    """Wavelet denoising (ideas/wavelet-denoise.md): the source doc uses ImageMagick/Wand's
    wavelet_denoise(); this app has no ImageMagick dependency, so this is the equivalent
    operation via PyWavelets instead -- a multi-level discrete wavelet decomposition per
    channel, soft-thresholding the detail (noise-dominated) coefficients, then reconstructing.
    threshold is fractional (0-1), scaled against each channel's own coefficient magnitude
    range so it behaves similarly across images regardless of absolute brightness."""
    import pywt

    channels = cv2.split(face_bgr.astype(np.float32))
    denoised_channels = []
    for channel in channels:
        coeffs = pywt.wavedec2(channel, wavelet, level=2)
        detail_coeffs = coeffs[1:]
        max_detail = max((np.abs(d).max() for level in detail_coeffs for d in level), default=1.0) or 1.0
        abs_threshold = threshold * max_detail
        thresholded = [coeffs[0]] + [
            tuple(pywt.threshold(d, abs_threshold, mode="soft") for d in level) for level in detail_coeffs
        ]
        reconstructed = pywt.waverec2(thresholded, wavelet)
        denoised_channels.append(reconstructed[:channel.shape[0], :channel.shape[1]])

    return np.clip(cv2.merge(denoised_channels), 0, 255).astype(np.uint8)


def apply_image_op(face_bgr: np.ndarray, op: str, **params) -> np.ndarray:
    """Dispatch for the per-face IMAGE OP button -- one entry point for all 7 one-click
    operations, so the caller doesn't need to know each function's name."""
    dispatch = {
        "intensity": apply_intensity_transform,
        "enhance": apply_enhance,
        "sharpen": apply_sharpen,
        "color_correct": apply_color_correct,
        "denoise": apply_denoise,
        "bilateral_filter": apply_bilateral_filter,
        "wavelet_denoise": apply_wavelet_denoise,
    }
    if op not in dispatch:
        raise ValueError(f"Unknown image op: {op}")
    return dispatch[op](face_bgr, **params)


def predict_gender_caffe(net, blob: np.ndarray) -> str:
    with _lock_for(net):
        net.setInput(blob)
        return GENDER_LIST[net.forward()[0].argmax()]


def predict_age_caffe(net, blob: np.ndarray) -> str:
    with _lock_for(net):
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
    with _lock_for(net):
        net.setInput(blob)
        probs = net.forward().flatten()
    age = sum(p * i for i, p in enumerate(probs))
    return f"{age:.0f}"


INSIGHTFACE_INPUT_SIZE = 96  # this genderage.onnx's actual input size (per its ONNX graph) --
# NOT the 112x112 insightface uses for its face-recognition/embedding models; verified via
# onnxruntime, which rejects 112x112 with a shape-mismatch error. The graph also embeds
# Sub/Mul normalization, so normalizing again produces near-constant garbage output.


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
    with _lock_for(eye_cascade):
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
    # This export starts with Sub/Mul normalization nodes, so feed raw pixels.
    blob = cv2.dnn.blobFromImage(aligned, 1.0, (INSIGHTFACE_INPUT_SIZE, INSIGHTFACE_INPUT_SIZE), (0, 0, 0), swapRB=True)
    with _lock_for(net):
        net.setInput(blob)
        return net.forward().flatten()


def predict_gender_insightface(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    out = _insightface_forward(net, frame_bgr, box)
    return "Female" if np.argmax(out[:2]) == 0 else "Male"


def predict_age_insightface(net, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    out = _insightface_forward(net, frame_bgr, box)
    return f"{round(out[2] * 100):.0f}"


def predict_age_mivolo(net: MiVOLOInference, face_bgr: np.ndarray) -> str:
    """Predict age with MiVOLO on a face crop (face-only mode)."""
    with _lock_for(net):
        age, _, _ = net.predict_face(face_bgr)
    return f"{int(round(age))}"


def predict_gender_mivolo(net: MiVOLOInference, face_bgr: np.ndarray) -> str:
    """Predict gender with MiVOLO on a face crop (face-only mode)."""
    with _lock_for(net):
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
    with _lock_for(net):
        net.setInput(blob.astype(np.float32))
        logits = net.forward().flatten()
    return EMOTION_LABELS_EFFICIENTNET[int(np.argmax(logits))]


def predict_emotion_mini_xception(net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS_MINI_XCEPTION."""
    face_gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    face_norm = (face_gray / 255.0 - 0.5) * 2.0
    tensor = face_norm[np.newaxis, ..., np.newaxis]
    with _lock_for(net):
        probs = net.predict(tensor, verbose=0).flatten()
    return EMOTION_LABELS_MINI_XCEPTION[int(np.argmax(probs))]


def predict_emotion_ferplus(net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS_FERPLUS."""
    face_gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    blob = face_gray[np.newaxis, np.newaxis, ...]
    with _lock_for(net):
        net.setInput(blob)
        logits = net.forward().flatten()
    return EMOTION_LABELS_FERPLUS[int(np.argmax(logits))]


def predict_emotion_hsemotion(net, face_bgr: np.ndarray) -> str:
    """Classify facial expression into one of EMOTION_LABELS_HSEMOTION."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(face_rgb, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=False, crop=False)
    blob = (blob - EMOTION_MEAN.reshape(1, 3, 1, 1)) / EMOTION_STD.reshape(1, 3, 1, 1)
    with _lock_for(net):
        net.setInput(blob.astype(np.float32))
        logits = net.forward().flatten()
    return EMOTION_LABELS_HSEMOTION[int(np.argmax(logits))]


def detect_drowsiness_haarcascade(eye_cascade, face_bgr: np.ndarray) -> bool:
    """Return True if fewer than MIN_EYES_OPEN eyes are visible (eyes likely closed)."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    with _lock_for(eye_cascade):
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
    with _lock_for(net):
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
    with _lock_for(net):
        probs = net.predict(face_resized[np.newaxis, ...], verbose=0).flatten()
    return _format_race_label(probs, RACE_LABELS_DEEPFACE)


def predict_gender_deepface(net, face_bgr: np.ndarray) -> str:
    # Same VGGFace-backbone preprocessing as predict_race_deepface: 224x224 BGR, unnormalized [0,255].
    face_resized = cv2.resize(face_bgr, (224, 224)).astype(np.float32)
    with _lock_for(net):
        probs = net.predict(face_resized[np.newaxis, ...], verbose=0).flatten()
    # deepface's GENDER_LABELS = ["Woman", "Man"]; normalize to this repo's Male/Female convention.
    return "Male" if np.argmax(probs) == 1 else "Female"


def compute_face_embedding(net, face_bgr: np.ndarray) -> np.ndarray:
    # Same VGGFace-backbone preprocessing as predict_race_deepface/predict_gender_deepface:
    # 224x224 BGR, unnormalized [0,255].
    face_resized = cv2.resize(face_bgr, (224, 224)).astype(np.float32)
    with _lock_for(net):
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


def _lbph_preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + resize to a fixed size -- LBPH compares histograms computed over a fixed
    cell grid, so training and query images need consistent dimensions."""
    return cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), LBPH_FACE_SIZE)


def enroll_lbph_face(name: str, face_bgr: np.ndarray) -> None:
    """LBPH's enrollment (see ideas/lbph.md): unlike vggface's single embedding per name,
    LBPH trains directly on raw face images, so ENROLL saves the actual (grayscale, fixed-
    size) crop -- one file per enrollment click, accumulating under gallery/lbph/<name>/.
    More enrolled photos per person generally improves LBPH's accuracy."""
    name = validate_lbph_name(name)
    person_dir = LBPH_GALLERY_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(person_dir.glob("*.png")))
    cv2.imwrite(str(person_dir / f"{existing:04d}.png"), _lbph_preprocess(face_bgr))


def validate_lbph_name(name: str) -> str:
    """Return a safe gallery directory name or raise for path traversal input."""
    name = name.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Enrollment name must be a non-empty name without path separators.")
    return name


_LBPH_CACHE_LOCK = threading.Lock()
_LBPH_CACHE_SIGNATURE = None
_LBPH_CACHE_RESULT = None


def _lbph_gallery_signature() -> str | None:
    if not LBPH_GALLERY_DIR.is_dir():
        return None
    entries = []
    for path in sorted(LBPH_GALLERY_DIR.glob("*/*.png")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append(f"{path.relative_to(LBPH_GALLERY_DIR)}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def train_lbph_recognizer():
    """Train an LBPHFaceRecognizer fresh from gallery/lbph/ (same 'no persisted model, retrain
    on demand' spirit as this app's eigenfaces feature -- cheap at the scale of a personal
    enrolled gallery). Returns (recognizer, label_names) where label_names[i] is the enrolled
    name for numeric label i, or None if opencv-contrib's cv2.face isn't available or nothing
    is enrolled yet."""
    global _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT
    if not hasattr(cv2, "face"):
        return None

    signature = _lbph_gallery_signature()
    with _LBPH_CACHE_LOCK:
        if signature == _LBPH_CACHE_SIGNATURE:
            return _LBPH_CACHE_RESULT
        if signature is None:
            _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, None
            return None

    label_names = sorted(p.name for p in LBPH_GALLERY_DIR.iterdir() if p.is_dir())
    if not label_names:
        _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, None
        return None

    features, labels = [], []
    for label, name in enumerate(label_names):
        for path in (LBPH_GALLERY_DIR / name).glob("*.png"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features.append(img)
                labels.append(label)
    if not features:
        _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, None
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(features, np.array(labels))
    with _LBPH_CACHE_LOCK:
        _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, (recognizer, label_names)
        return _LBPH_CACHE_RESULT


def decode_image_bytes(file_bytes: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Decode uploaded image bytes and reject empty or unsupported payloads clearly."""
    encoded = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("The uploaded file is empty or could not be read.")
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("The uploaded file is not a valid supported image.")
    return frame


def predict_identity_lbph(recognizer, label_names: list[str], face_bgr: np.ndarray) -> tuple[str, float] | None:
    """LOWER LBPH confidence is a better match (opposite convention from vggface's cosine
    similarity) -- accept only below LBPH_CONFIDENCE_THRESHOLD."""
    with _lock_for(recognizer):
        label, confidence = recognizer.predict(_lbph_preprocess(face_bgr))
    return (label_names[label], confidence) if confidence < LBPH_CONFIDENCE_THRESHOLD else None


PREDICTION_CACHE_MAX_SIZE = 2048

# Content-addressed cache for per-face classifier outputs, keyed on the exact preprocessed
# input bytes rather than a face-identity embedding: a face-embedding hash is NOT a stable
# cache key (an adjusted or re-cropped version of "the same" face has different bytes and
# may legitimately warrant a different output), but identical bytes fed to the same model
# always produce identical deterministic output, so hashing the input itself is correct by
# construction. Real speedup comes from Streamlit re-running the whole script (and thus
# every classifier for every face) on any unrelated widget interaction even when the image,
# adjustments, and active models haven't changed. Module-level and unlocked: concurrent
# sessions may occasionally race and recompute the same missing key redundantly, but a plain
# dict get/set can't corrupt the cache under the GIL, so that's a wasted-work risk, not a
# correctness one. Bounded (LRU-evicted) so a long session doesn't grow this unboundedly.
_PREDICTION_CACHE: "OrderedDict[tuple, object]" = OrderedDict()


def _cached_face_predict(feature: str, model_key: str, face_bgr: np.ndarray, predict_fn, *args):
    """Memoize a predict_*(net, face, ...) call on (feature, model_key, hash(face bytes)).
    Only used for predictors whose sole content input is the face crop itself -- predictors
    that instead take the full frame + box (fairface, insightface) or hash a much larger,
    more adjustment-sensitive buffer for comparatively little benefit are left uncached here."""
    cache_key = (feature, model_key, face_bgr.shape, hashlib.blake2b(face_bgr.tobytes(), digest_size=16).digest())
    cached = _PREDICTION_CACHE.get(cache_key)
    if cached is not None or cache_key in _PREDICTION_CACHE:
        _PREDICTION_CACHE.move_to_end(cache_key)
        return cached
    value = predict_fn(*args)
    _PREDICTION_CACHE[cache_key] = value
    if len(_PREDICTION_CACHE) > PREDICTION_CACHE_MAX_SIZE:
        _PREDICTION_CACHE.popitem(last=False)
    return value


def _sanitize_column_name(feature: str, model_key: str) -> str:
    return f"{feature}_{model_key}".lower().replace(" ", "_").replace("-", "_")


def _gather_face_results(pairs_by_feature: dict[str, list[tuple[str, str]]]) -> dict[str, str]:
    """Flatten analyze_frame's per-feature (model_key, value) pairs into
    {column_name: value}, one entry per (feature, model) that actually produced a value for
    this face. A model that wasn't active, or produced no result, contributes no key here --
    this is what makes save_face()'s column creation lazy/sparse."""
    results = {}
    for feature, pairs in pairs_by_feature.items():
        for model_key, value in pairs:
            results[_sanitize_column_name(feature, model_key)] = value
    return results


def _crop_and_resize_for_eigenfaces(face_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + center-square crop (tighter than the padded face crop already saved to
    faces/, i.e. 'zoomed in') + resize to EIGEN_FACE_SIZE. Used both when saving a new face
    to eigen/ and when preprocessing a live query face for match_face_eigenfaces, so the two
    are directly comparable."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    side = min(h, w)
    cy, cx = h // 2, w // 2
    zoomed = gray[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2]
    return cv2.resize(zoomed, EIGEN_FACE_SIZE)


def save_face(face_bgr: np.ndarray, raw_columns: dict[str, str]) -> int:
    """Save one classified face: a DB row (sparse columns, see module docstring above),
    the color crop to faces/{id}.jpg, and a grayscale/zoomed crop to eigen/{id}.jpg for
    eigenfaces matching. raw_columns is {column_name: value} from _gather_face_results --
    only columns present here get created (ALTER TABLE), so a model that was never run on
    any saved face never gets a column. Returns the randomly generated id (1..999999)."""
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(FACES_DB_FILE))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY)")
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(faces)")}
        # Column names are interpolated directly (sqlite3 can't parameterize identifiers) --
        # safe here because raw_columns' keys only ever come from _sanitize_column_name(feature,
        # model_key), where both parts are drawn from this codebase's own fixed *_MODEL_OPTIONS
        # lists, never from user-supplied text.
        for col in raw_columns:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE faces ADD COLUMN {col} TEXT")
                existing_cols.add(col)

        face_id = None
        for _ in range(20):
            candidate = random.randint(1, 999_999)
            cols = ["id"] + list(raw_columns.keys())
            placeholders = ", ".join("?" for _ in cols)
            try:
                conn.execute(f"INSERT INTO faces ({', '.join(cols)}) VALUES ({placeholders})", [candidate] + list(raw_columns.values()))
                face_id = candidate
                break
            except sqlite3.IntegrityError:
                continue
        if face_id is None:
            raise RuntimeError("Could not generate a unique face id after 20 attempts")
        conn.commit()
    finally:
        conn.close()

    cv2.imwrite(str(FACES_DIR / f"{face_id}.jpg"), face_bgr)
    cv2.imwrite(str(EIGEN_DIR / f"{face_id}.jpg"), _crop_and_resize_for_eigenfaces(face_bgr))

    return face_id


def _load_eigen_images() -> tuple[list[int], np.ndarray]:
    """Load every image in eigen/ as a flattened float64 row vector. Returns (ids, data)
    where data has shape (M, EIGEN_FACE_SIZE[0]*EIGEN_FACE_SIZE[1])."""
    ids: list[int] = []
    vectors = []
    if EIGEN_DIR.is_dir():
        for path in sorted(EIGEN_DIR.iterdir()):
            if path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
                continue
            try:
                face_id = int(path.stem)
            except ValueError:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[::-1] != EIGEN_FACE_SIZE:
                img = cv2.resize(img, EIGEN_FACE_SIZE)
            ids.append(face_id)
            vectors.append(img.flatten().astype(np.float64))
    dim = EIGEN_FACE_SIZE[0] * EIGEN_FACE_SIZE[1]
    return ids, (np.array(vectors) if vectors else np.empty((0, dim)))


def _train_eigenfaces(k: int = 15) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray] | None:
    """Turk & Pentland eigenfaces (PCA) training step, per ideas/eigenfaces.md: trains fresh
    against every image in eigen/ (the training set is just previously-SAVEd faces, so this is
    cheap at the scale it's meant for). Uses the M x M covariance trick from the paper (A @ A.T
    instead of A.T @ A) since the number of saved faces M is much smaller than the pixel
    dimension N*N. Returns (ids, mean_face, eigenfaces, weights) or None if eigen/ has fewer
    than 2 images (PCA needs at least 2 samples to have any variance to project onto)."""
    ids, data = _load_eigen_images()
    if len(ids) < 2:
        return None

    mean_face = data.mean(axis=0)
    A = data - mean_face  # (M, N*N)

    cov_small = A @ A.T  # (M, M)
    eigvals, eigvecs_small = np.linalg.eigh(cov_small)
    order = np.argsort(eigvals)[::-1][:k]
    eigvecs_small = eigvecs_small[:, order]

    eigenfaces = A.T @ eigvecs_small  # (N*N, k') -- k' = min(k, M)
    norms = np.linalg.norm(eigenfaces, axis=0)
    norms[norms == 0] = 1.0
    eigenfaces = eigenfaces / norms

    weights = A @ eigenfaces  # (M, k') -- training faces' coordinates in eigenspace
    return ids, mean_face, eigenfaces, weights


def match_face_eigenfaces(face_bgr: np.ndarray, k: int = 15) -> tuple[int, float] | None:
    """Match one face against eigen/ (see _train_eigenfaces). Returns (matched_face_id,
    L2_distance) for the closest training face if it clears EIGENFACE_DISTANCE_THRESHOLD, else
    None. For matching several faces from the same image, prefer match_faces_eigenfaces_batch
    -- this trains PCA fresh on every call, which is wasteful when called once per face.

    EIGENFACE_DISTANCE_THRESHOLD is an untuned heuristic -- unlike RECOGNITION_COSINE_THRESHOLD
    (deepface's own published default), there's no established reference value for raw
    grayscale-pixel eigenspace distance at this face size; treat match/no-match near the
    threshold with skepticism until tuned against real saved-face data."""
    trained = _train_eigenfaces(k)
    if trained is None:
        return None
    ids, mean_face, eigenfaces, weights = trained
    return _match_against_trained(face_bgr, ids, mean_face, eigenfaces, weights)


def _match_against_trained(face_bgr: np.ndarray, ids: list[int], mean_face: np.ndarray, eigenfaces: np.ndarray, weights: np.ndarray) -> tuple[int, float] | None:
    query = _crop_and_resize_for_eigenfaces(face_bgr).flatten().astype(np.float64) - mean_face
    query_weights = query @ eigenfaces
    distances = np.linalg.norm(weights - query_weights, axis=1)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    return (ids[best_idx], best_dist) if best_dist <= EIGENFACE_DISTANCE_THRESHOLD else None


def match_faces_eigenfaces_batch(faces_bgr: list[np.ndarray], k: int = 15) -> list[tuple[int, float] | None]:
    """Match several faces from the same image against eigen/ in one PCA training pass --
    used by the 'scan all faces' recognized/unrecognized button so an N-face image doesn't
    retrain PCA N times. Returns one match (or None) per input face, same order."""
    trained = _train_eigenfaces(k)
    if trained is None:
        return [None] * len(faces_bgr)
    ids, mean_face, eigenfaces, weights = trained
    return [_match_against_trained(face_bgr, ids, mean_face, eigenfaces, weights) for face_bgr in faces_bgr]


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


def _detect_face_landmarker(landmarker, face_bgr: np.ndarray):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    with _lock_for(landmarker):
        return landmarker.detect(mp_image)


def predict_texture_artifact_score(face_bgr: np.ndarray) -> float:
    """Score regular high-frequency texture in a face crop as a replay cue."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sample = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return texture_artifact_score(sample.tolist())


def predict_expression_blendshapes(landmarker, face_bgr: np.ndarray, result=None) -> str:
    """Predict facial expression via MediaPipe's BlendShapes (52 continuous muscle coefficients).
    Returns the top 3 highest-scoring blendshapes as a comma-separated string,
    or 'no landmarks' if no face is detected."""
    result = result if result is not None else _detect_face_landmarker(landmarker, face_bgr)

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


def predict_body_composition_face_geometry(landmarker, face_bgr: np.ndarray, result=None) -> str:
    """Return the transparent, relative face-geometry body-composition proxy.

    The MediaPipe result is shared with expression, gaze, landmarks, and liveness so this
    backend adds no second landmark inference.  ``face_bgr`` remains in the signature to
    match the per-face predictor contract; the proxy intentionally uses landmarks only.
    """
    result = result if result is not None else _detect_face_landmarker(landmarker, face_bgr)
    return estimate_face_composition(result)


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
    with _lock_for(net):
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
    with _lock_for(net):
        probs = net.predict(face_norm[np.newaxis, ...], verbose=0).flatten()
    return SKIN_TONE_LABELS[int(np.argmax(probs))]


def predict_glasses_mobilenet(net, face_bgr: np.ndarray) -> str:
    """Sorour190/Glasses-Detector's glasses_face224.onnx: MobileNetV3-Large, 224x224 RGB,
    uint8 NHWC input (normalization baked into the ONNX graph itself), outputs a named
    'eyeglasses_prob' scalar already through softmax. License unstated by the source repo
    (flagged in README, same treatment as DAN/SSR-Net)."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    blob = face_rgb[np.newaxis, ...].astype(np.uint8)
    prob = float(net.run(["eyeglasses_prob"], {net.get_inputs()[0].name: blob})[0].flatten()[0])
    return "glasses" if prob >= GLASSES_THRESHOLD else "none"


def predict_mask_mobilenetv2(net, face_bgr: np.ndarray) -> str:
    """chandrikadeb7/Face-Mask-Detection: MobileNetV2 backbone (imagenet weights,
    include_top=False) + AveragePooling2D(7,7) + Flatten + Dense(128, relu) + Dropout(0.5) +
    Dense(2, softmax), 224x224 RGB, keras.applications.mobilenet_v2.preprocess_input scaling.
    Class order (sklearn LabelBinarizer, alphabetical) is MASK_LABELS = ['with_mask', 'without_mask']."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = face_rgb / 127.5 - 1.0
    with _lock_for(net):
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
    with _lock_for(eye_cascade):
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


def predict_face_landmarks_mediapipe(landmarker, face_bgr: np.ndarray, result=None) -> list[tuple[float, float]] | None:
    """MediaPipe FaceLandmarker (same model instance as Expression's blendshapes backend --
    one model, two features, same pattern as insightface/fairface elsewhere in this file).
    Returns 468 (x, y) points normalized to [0, 1] within face_bgr, or None if no face found."""
    result = result if result is not None else _detect_face_landmarker(landmarker, face_bgr)
    if not result.face_landmarks:
        return None
    return [(lm.x, lm.y) for lm in result.face_landmarks[0]]


def predict_gaze_mediapipe(landmarker, face_bgr: np.ndarray, result=None) -> str:
    """Estimate coarse gaze direction from MediaPipe iris and eye landmarks.

    The result describes the direction relative to the face crop. It is a geometric
    attention cue, not a calibrated eye tracker.
    """
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr, result)
    if points is None or len(points) < 478:
        return "unknown"

    def center(indices: tuple[int, ...]) -> np.ndarray:
        return np.mean([points[index] for index in indices], axis=0)

    directions = []
    for iris, corners, vertical in (
        ((468, 469, 470, 471, 472), (33, 133), (159, 145)),
        ((473, 474, 475, 476, 477), (362, 263), (386, 374)),
    ):
        iris_center = center(iris)
        left_corner, right_corner = (points[index] for index in corners)
        eye_width = abs(right_corner[0] - left_corner[0])
        eye_height = abs(points[vertical[0]][1] - points[vertical[1]][1])
        if eye_width < 1e-6 or eye_height < 1e-6:
            continue
        horizontal = (iris_center[0] - min(left_corner[0], right_corner[0])) / eye_width
        vertical_position = (iris_center[1] - min(points[index][1] for index in vertical)) / eye_height
        directions.append((horizontal, vertical_position))

    if not directions:
        return "unknown"
    horizontal, vertical_position = np.mean(directions, axis=0)
    horizontal_label = "left" if horizontal < 0.38 else "right" if horizontal > 0.62 else "center"
    vertical_label = "up" if vertical_position < 0.35 else "down" if vertical_position > 0.65 else "level"
    return f"{horizontal_label}/{vertical_label}"


def predict_head_pose_mediapipe(landmarker, face_bgr: np.ndarray, result=None) -> str:
    """Estimate coarse yaw/pitch from stable MediaPipe face landmarks."""
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr, result)
    if points is None or len(points) < 264:
        return "unknown"
    image_points = np.float32([points[i] for i in (1, 152, 33, 263, 61, 291)])
    h, w = face_bgr.shape[:2]
    image_points[:, 0] *= w
    image_points[:, 1] *= h
    model_points = np.float32([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
    ])
    focal = float(w)
    camera = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return "unknown"
    matrix, _ = cv2.Rodrigues(rotation)
    pitch = np.degrees(np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2])))
    yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return f"yaw={yaw:.0f}°, pitch={pitch:.0f}°"


def predict_makeup_heuristic(face_bgr: np.ndarray) -> str:
    """Flag strong cosmetic-like color contrast; heuristic, not a trained classifier."""
    if face_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    central = hsv[int(h * 0.2):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    saturation = float(np.percentile(central[..., 1], 90))
    red_ratio = float(np.mean((central[..., 0] < 12) | (central[..., 0] > 165)))
    return "possible" if saturation > 150 and red_ratio > 0.12 else "not detected"


def _record_model_latency(metrics: dict | None, feature: str, model: str, started: float) -> None:
    if metrics is None:
        return
    metrics.setdefault("model_latency_ms", {}).setdefault(f"{feature}/{model}", []).append(
        (time.perf_counter() - started) * 1000
    )


def run_3d_reconstruction(models: "Models", face_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Deep3DFaceRecon_pytorch-based 3D reconstruction (see src/nets/deep3d_recon.py) for one
    face crop. Reuses the same FaceLandmarker instance as Expression/Face Landmarks to derive
    the 5-point alignment landmarks this pipeline needs. Returns (vertices, faces, per-vertex
    RGB colors) or None if the deep3d model, the BFM data, or the face landmarker aren't
    available, or if no face landmarks were found in this crop."""
    bundle = models.reconstruction_3d_nets.get("deep3d")
    landmarker = models.face_landmarks_nets.get("blendshapes")
    if bundle is None or landmarker is None:
        return None

    recon_net, bfm_model, lm3d_template = bundle
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr)
    if points is None:
        return None

    h, w = face_bgr.shape[:2]
    landmarks_5pt = landmarks_5pt_from_mediapipe(points, w, h)
    return reconstruct_face_3d(recon_net, bfm_model, face_bgr, landmarks_5pt, lm3d_template)


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
    with _lock_for(landmarker):
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


def draw_recognition_scan(frame: np.ndarray, faces: list[tuple[tuple[int, int, int, int], bool]]) -> None:
    """Draw the 'scan all faces' button's result onto frame: a green box + 'Recognized' label
    for faces matched against eigen/, red + 'Unrecognized' otherwise -- same color convention
    as ideas/recognition.md's own implementation."""
    for (x1, y1, x2, y2), recognized in faces:
        color = (0, 255, 0) if recognized else (0, 0, 255)
        box_thickness = int(round(frame.shape[0] / 150)) or 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness, 8)
        draw_outlined_text(frame, "Recognized" if recognized else "Unrecognized", (x1, max(20, y1 - 10)), color)


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
    active_gaze: set,
    global_adjustments: dict,
    face_adjustments: dict,
    face_detector: str = "ssd",
    metrics: dict | None = None,
    tracker: "FaceTracker | None" = None,
    liveness_tracker: "LivenessTracker | None" = None,
    active_liveness: set | None = None,
    active_body_composition: set | None = None,
):
    """Detect faces and run inference for whichever model keys are active per feature.
    Multiple active models for the same feature (e.g. active_age = {"caffe", "ssrnet"})
    all run and are shown together. No Streamlit calls (safe for background threads).

    global_adjustments apply to the whole frame first, before face detection even runs --
    every output derived from this call (the annotated image, every face crop, every
    classification) sees the adjusted pixels. face_adjustments apply again, per detected
    face, to that face's own crop only, after detection but before classification -- they
    affect just that one face's thumbnail/attributes, not the shared frame or other faces.

    face_detector picks which face detection backend runs (unlike every other feature,
    exactly one runs per frame -- running two detectors and merging their boxes would just
    produce duplicate/overlapping faces, not a meaningfully combined result). "yolo" falls
    back to "ssd" (the always-required detector) if the YOLO model isn't loaded.

    tracker (#2) is optional and stays None for single-image callers (upload/snapshot have no
    "next frame" for an ID to persist into). When a FaceTracker is passed -- video/webcam LIVE
    mode only -- each face's dict also carries a stable "track_id" (see FaceTracker), and the
    number burned into the annotated frame is that track_id instead of this frame's
    detection-order position, so tracking is visible, not just data the caller ignores).

    liveness_tracker is optional for the same reason. Static callers get texture-only evidence;
    LIVE callers also get blink transitions keyed by the stable track ID. active_liveness selects
    the loaded liveness backend; omitted callers use every loaded backend."""
    if active_liveness is None:
        active_liveness = set(models.liveness_nets)
    if active_body_composition is None:
        active_body_composition = set(models.body_composition_nets)
    if global_adjustments and any(global_adjustments.values()):
        frame = apply_image_adjustments(frame, global_adjustments)

    annotated_frame = frame.copy()
    yolo_net = models.yolo_face_nets.get("yolo")
    if face_detector == "yolo" and yolo_net is not None:
        face_boxes = detect_faces_yolo(yolo_net, frame, conf_threshold)
    else:
        face_boxes = detect_faces(models.face_net, frame, conf_threshold)
    track_ids = tracker.update(face_boxes) if tracker is not None else [None] * len(face_boxes)
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

    # Trained once per frame, not once per face -- LBPH has no persisted model, retraining per
    # face would multiply an already-nontrivial cost by the face count for no benefit.
    lbph_trained = train_lbph_recognizer() if "lbph" in active_recognition and models.recognition_nets.get("lbph") else None

    for idx, ((x1, y1, x2, y2), track_id) in enumerate(zip(face_boxes, track_ids), 1):
        # Correct in-plane roll (tilted head) before cropping/classifying, using the same
        # eye cascade as drowsiness detection -- no new model/dependency. crop_frame/cx*/cy*
        # are the rotation-corrected region+box; x1..y2 stay untouched for the box overlay
        # drawn on annotated_frame further below.
        crop_frame, (cx1, cy1, cx2, cy2) = frame, (x1, y1, x2, y2)
        if eye_cascade is not None:
            probe = frame[max(0, y1 - 20):min(y2 + 20, frame.shape[0]), max(0, x1 - 20):min(x2 + 20, frame.shape[1])]
            angle = _estimate_roll_angle(probe, eye_cascade) if probe.size else None
            if angle is not None and abs(angle) > 3:  # skip work for near-level faces
                crop_frame, (cx1, cy1, cx2, cy2) = _rotate_region(frame, (x1, y1, x2, y2), angle)

        y1_crop = max(0, cy1 - 20)
        y2_crop = min(cy2 + 20, crop_frame.shape[0])
        x1_crop = max(0, cx1 - 20)
        x2_crop = min(cx2 + 20, crop_frame.shape[1])

        face = crop_frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        if face_adjustments and any(face_adjustments.values()):
            face = apply_image_adjustments(face, face_adjustments)

        # Expression, gaze, head pose, and drawn landmarks all consume the
        # same MediaPipe FaceLandmarker result. Detect once before dispatching
        # feature tasks so the shared model is not run repeatedly per crop.
        liveness_net = models.liveness_nets.get("mediapipe") if "mediapipe" in active_liveness else None
        body_composition_net = (
            models.body_composition_nets.get("face_geometry")
            if "face_geometry" in active_body_composition else None
        )
        face_landmarker = (
            liveness_net if liveness_net is not None
            else body_composition_net or models.face_landmarks_nets.get("blendshapes")
        )
        needs_face_landmarks = (
            "blendshapes" in active_expression
            or "mediapipe" in active_gaze
            or "blendshapes" in active_face_landmarks
            or body_composition_net is not None
            or face_landmarker is not None
        )
        landmarker_result = (
            _detect_face_landmarker(face_landmarker, face)
            if face_landmarker is not None and needs_face_landmarks
            else None
        )
        texture_score = predict_texture_artifact_score(face)
        blink_score = blink_score_from_landmarker(landmarker_result)

        blob227 = None
        if need_blob227:
            blob227 = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # #19: each feature below is independent of every other feature for this face (they
        # read the same face/crop_frame/blob227 but never share mutable state with each
        # other -- _cached_face_predict's cache and _record_model_latency's metrics dict are
        # both documented/verified safe for this, see their own docstrings/comments), so they
        # run concurrently on _INFERENCE_EXECUTOR instead of one after another. Shared model
        # instances (e.g. one fairface/insightface net backing both age and gender, or one
        # MediaPipe landmarker backing expression/gaze/face_landmarks) are made safe for this
        # by _lock_for(), applied at each net's actual setInput/forward/predict/detect call
        # site (see the top of this file) -- concurrent calls onto the SAME net serialize
        # there, while calls onto DIFFERENT nets still overlap for real.
        def _age_task():
            pairs = []
            for key in active_age:
                net = models.age_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "caffe":
                    value = _cached_face_predict("age", key, face, predict_age_caffe, net, blob227)
                elif key == "ssrnet":
                    value = _cached_face_predict("age", key, face, predict_age_ssrnet, net, face)
                elif key == "fairface":
                    value = predict_age_fairface(net, crop_frame, (cx1, cy1, cx2, cy2))
                elif key == "dex":
                    value = _cached_face_predict("age", key, face, predict_age_dex, net, face)
                elif key == "mivolo":
                    value = _cached_face_predict("age", key, face, predict_age_mivolo, net, face)
                else:
                    value = predict_age_insightface(net, crop_frame, (cx1, cy1, cx2, cy2))
                pairs.append((key, value))
                _record_model_latency(metrics, "age", key, started)
            return pairs

        def _gender_task():
            pairs = []
            for key in active_gender:
                net = models.gender_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "caffe":
                    value = _cached_face_predict("gender", key, face, predict_gender_caffe, net, blob227)
                elif key == "deepface":
                    value = _cached_face_predict("gender", key, face, predict_gender_deepface, net, face)
                elif key == "fairface":
                    value = predict_gender_fairface(net, crop_frame, (cx1, cy1, cx2, cy2))
                elif key == "mivolo":
                    value = _cached_face_predict("gender", key, face, predict_gender_mivolo, net, face)
                else:
                    value = predict_gender_insightface(net, crop_frame, (cx1, cy1, cx2, cy2))
                pairs.append((key, value))
                _record_model_latency(metrics, "gender", key, started)
            return pairs

        def _emotion_task():
            pairs = []
            for key in active_emotion:
                net = models.emotion_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "dan":
                    value = _cached_face_predict("emotion", key, face, predict_emotion_dan, net, face)
                elif key == "mini_xception":
                    value = _cached_face_predict("emotion", key, face, predict_emotion_mini_xception, net, face)
                elif key == "ferplus":
                    value = _cached_face_predict("emotion", key, face, predict_emotion_ferplus, net, face)
                elif key == "hsemotion":
                    value = _cached_face_predict("emotion", key, face, predict_emotion_hsemotion, net, face)
                else:
                    value = _cached_face_predict("emotion", key, face, predict_emotion_efficientnet, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "emotion", key, started)
            return pairs

        def _race_task():
            pairs = []
            for key in active_race:
                net = models.race_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = predict_race_fairface(net, crop_frame, (cx1, cy1, cx2, cy2)) if key == "fairface" else _cached_face_predict("race", key, face, predict_race_deepface, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "race", key, started)
            return pairs

        def _expression_task():
            pairs = []
            for key in active_expression:
                net = models.expression_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict(
                    "expression", key, face, predict_expression_blendshapes, net, face, landmarker_result
                )
                pairs.append((key, value))
                _record_model_latency(metrics, "expression", key, started)
            return pairs

        def _body_composition_task():
            pairs = []
            for key in active_body_composition:
                net = models.body_composition_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "face_geometry":
                    value = predict_body_composition_face_geometry(net, face, landmarker_result)
                else:
                    continue
                pairs.append((key, value))
                _record_model_latency(metrics, "body_composition", key, started)
            return pairs

        def _gaze_task():
            pairs = []
            for key in active_gaze:
                net = models.gaze_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = predict_gaze_mediapipe(net, face, landmarker_result)
                pairs.append((key, value))
                _record_model_latency(metrics, "gaze", key, started)
            return pairs

        def _head_pose_task():
            pairs = []
            for key in active_gaze:
                net = models.gaze_nets.get(key)
                if net is not None:
                    pairs.append((key, predict_head_pose_mediapipe(net, face, landmarker_result)))
            return pairs

        def _recognition_task():
            pairs = []
            embedding = None
            for key in active_recognition:
                net = models.recognition_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "lbph":
                    if lbph_trained is None:
                        value = "UNKNOWN"
                    else:
                        recognizer, label_names = lbph_trained
                        match = predict_identity_lbph(recognizer, label_names, face)
                        value = f"{match[0]} ({match[1]:.0f})" if match else "UNKNOWN"
                else:
                    # Only the embedding step is cached, not the match -- the gallery can
                    # change (enrollment/deletion) between calls with the same face bytes,
                    # and a stale cached match result would silently ignore that.
                    embedding = _cached_face_predict("embedding", key, face, compute_face_embedding, net, face)
                    match = match_face_identity(embedding, gallery)
                    value = f"{match[0]} ({match[1] * 100:.0f}%)" if match else "UNKNOWN"
                pairs.append((key, value))
                _record_model_latency(metrics, "recognition", key, started)
            return pairs, embedding

        def _facial_hair_task():
            pairs = []
            for key in active_facial_hair:
                net = models.facial_hair_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict("facial_hair", key, face, predict_facial_hair_bisenet, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "facial_hair", key, started)
            return pairs

        def _skin_tone_task():
            pairs = []
            for key in active_skin_tone:
                net = models.skin_tone_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict("skin_tone", key, face, predict_skin_tone_vgg16, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "skin_tone", key, started)
            return pairs

        def _glasses_task():
            pairs = []
            for key in active_glasses:
                net = models.glasses_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict("glasses", key, face, predict_glasses_mobilenet, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "glasses", key, started)
            return pairs

        def _mask_task():
            pairs = []
            for key in active_mask:
                net = models.mask_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict("mask", key, face, predict_mask_mobilenetv2, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "mask", key, started)
            return pairs

        def _hair_color_task():
            pairs = []
            for key in active_hair_color:
                if key not in models.hair_color_nets:
                    continue
                started = time.perf_counter()
                value = predict_hair_color_colorimetric(crop_frame, (cx1, cy1, cx2, cy2))
                pairs.append((key, value))
                _record_model_latency(metrics, "hair_color", key, started)
            return pairs

        def _eye_color_task():
            pairs = []
            for key in active_eye_color:
                net = models.eye_color_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                value = _cached_face_predict("eye_color", key, face, predict_eye_color_colorimetric, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "eye_color", key, started)
            return pairs

        def _drowsiness_task():
            pairs = []
            for key in active_drowsiness:
                net = models.drowsiness_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                drowsy = _cached_face_predict("drowsiness", key, face, detect_drowsiness_haarcascade, net, face)
                pairs.append((key, "DROWSY" if drowsy else "ALERT"))
                _record_model_latency(metrics, "drowsiness", key, started)
            return pairs

        def _liveness_task():
            started = time.perf_counter()
            if liveness_net is None:
                result = assess_static_liveness(texture_score)
                key = "heuristic"
            elif liveness_tracker is not None and track_id is not None:
                result = liveness_tracker.update(track_id, blink_score, texture_score)
                key = "mediapipe"
            else:
                result = assess_static_liveness(texture_score)
                key = "mediapipe"
            _record_model_latency(metrics, "liveness", key, started)
            return [(key, result.summary)], result

        futures = {
            "age": _INFERENCE_EXECUTOR.submit(_age_task),
            "gender": _INFERENCE_EXECUTOR.submit(_gender_task),
            "emotion": _INFERENCE_EXECUTOR.submit(_emotion_task),
            "race": _INFERENCE_EXECUTOR.submit(_race_task),
            "expression": _INFERENCE_EXECUTOR.submit(_expression_task),
            "body_composition": _INFERENCE_EXECUTOR.submit(_body_composition_task),
            "gaze": _INFERENCE_EXECUTOR.submit(_gaze_task),
            "head_pose": _INFERENCE_EXECUTOR.submit(_head_pose_task),
            "recognition": _INFERENCE_EXECUTOR.submit(_recognition_task),
            "facial_hair": _INFERENCE_EXECUTOR.submit(_facial_hair_task),
            "skin_tone": _INFERENCE_EXECUTOR.submit(_skin_tone_task),
            "glasses": _INFERENCE_EXECUTOR.submit(_glasses_task),
            "mask": _INFERENCE_EXECUTOR.submit(_mask_task),
            "hair_color": _INFERENCE_EXECUTOR.submit(_hair_color_task),
            "eye_color": _INFERENCE_EXECUTOR.submit(_eye_color_task),
            "drowsiness": _INFERENCE_EXECUTOR.submit(_drowsiness_task),
            "liveness": _INFERENCE_EXECUTOR.submit(_liveness_task),
        }

        age_pairs = futures["age"].result()
        gender_pairs = futures["gender"].result()
        emotion_pairs = futures["emotion"].result()
        race_pairs = futures["race"].result()
        expression_pairs = futures["expression"].result()
        body_composition_pairs = futures["body_composition"].result()
        gaze_pairs = futures["gaze"].result()
        head_pose_pairs = futures["head_pose"].result()
        recognition_pairs, face_embedding = futures["recognition"].result()
        facial_hair_pairs = futures["facial_hair"].result()
        skin_tone_pairs = futures["skin_tone"].result()
        glasses_pairs = futures["glasses"].result()
        mask_pairs = futures["mask"].result()
        hair_color_pairs = futures["hair_color"].result()
        eye_color_pairs = futures["eye_color"].result()
        drowsy_pairs = futures["drowsiness"].result()
        liveness_pairs, liveness_result = futures["liveness"].result()

        if metrics is not None and emotion_pairs:
            metrics.setdefault("emotion_samples", []).extend(
                {"model": key, "emotion": value} for key, value in emotion_pairs
            )

        face_drowsy = any(value == "DROWSY" for _, value in drowsy_pairs)
        any_drowsy = any_drowsy or face_drowsy

        # Attribute text is intentionally NOT drawn on the shared image -- with several faces
        # close together, per-face text overlaps illegibly. The box + a small index number is
        # the only thing burned into pixels; full results are returned as structured data for
        # the caller to render as separate per-face UI (see src/app.py's target cards).
        box_thickness = int(round(frame.shape[0] / 150)) or 1
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness, 8)
        display_id = track_id if track_id is not None else idx
        draw_outlined_text(annotated_frame, str(display_id), (x1, max(20, y1 - 10)), (0, 255, 255))

        landmarks_net = models.face_landmarks_nets.get("blendshapes")
        if landmarks_net is not None and "blendshapes" in active_face_landmarks:
            landmark_points = predict_face_landmarks_mediapipe(landmarks_net, face, landmarker_result)
            if landmark_points is not None:
                draw_face_landmarks(annotated_frame, landmark_points, (x1, y1, x2, y2))

        drowsy_parts = _format_results(drowsy_pairs)
        status = drowsy_parts[0] if len(drowsy_parts) == 1 else (", ".join(drowsy_parts) if drowsy_parts else None)
        eye_contact = [f"{key}=yes" if value.startswith("center/") else f"{key}=no" for key, value in gaze_pairs]

        raw_columns = _gather_face_results({
            "age": age_pairs, "gender": gender_pairs, "race": race_pairs, "emotion": emotion_pairs,
            "expression": expression_pairs, "gaze": gaze_pairs, "identity": recognition_pairs, "facial_hair": facial_hair_pairs,
            "body_composition": body_composition_pairs,
            "eye_contact": [("derived", value) for value in eye_contact], "head_pose": head_pose_pairs,
            "skin_tone": skin_tone_pairs, "glasses": glasses_pairs, "mask": mask_pairs,
            "hair_color": hair_color_pairs, "eye_color": eye_color_pairs, "drowsiness": drowsy_pairs,
            "liveness": liveness_pairs,
        })
        model_results = [
            {"Feature": feature.replace("_", " ").upper(), "Model": model, "Output": str(value)}
            for feature, pairs in {
                "age": age_pairs, "gender": gender_pairs, "race": race_pairs, "emotion": emotion_pairs,
                "expression": expression_pairs, "gaze": gaze_pairs, "identity": recognition_pairs,
                "body composition": body_composition_pairs,
                "eye contact": [("derived", value) for value in eye_contact],
                "head pose": head_pose_pairs,
                "facial hair": facial_hair_pairs, "skin tone": skin_tone_pairs, "glasses": glasses_pairs,
                "mask": mask_pairs, "hair color": hair_color_pairs, "eye color": eye_color_pairs,
                "drowsiness": drowsy_pairs, "liveness": liveness_pairs,
            }.items()
            for model, value in pairs
        ]

        cropped_faces.append({
            "idx": idx,
            "track_id": track_id,
            "box": (x1, y1, x2, y2),
            "image": cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
            "age": _format_results(age_pairs),
            "gender": _format_results(gender_pairs),
            "race": _format_results(race_pairs),
            "emotion": _format_results(emotion_pairs),
            "expression": _format_results(expression_pairs),
            "body_composition": _format_results(body_composition_pairs),
            "gaze": _format_results(gaze_pairs),
            "eye_contact": eye_contact,
            "head_pose": _format_results(head_pose_pairs),
            "makeup": [predict_makeup_heuristic(face)],
            "identity": _format_results(recognition_pairs),
            "facial_hair": _format_results(facial_hair_pairs),
            "skin_tone": _format_results(skin_tone_pairs),
            "glasses": _format_results(glasses_pairs),
            "mask": _format_results(mask_pairs),
            "hair_color": _format_results(hair_color_pairs),
            "eye_color": _format_results(eye_color_pairs),
            "liveness": [liveness_result.summary],
            "liveness_status": liveness_result.status,
            "blink_count": liveness_result.blink_count,
            "blink_rate": liveness_result.blink_rate,
            "texture_score": liveness_result.texture_score,
            "texture_artifact": liveness_result.texture_artifact,
            "embedding": face_embedding.tolist() if face_embedding is not None else None,
            "raw_columns": raw_columns,
            "model_results": model_results,
            "status": status,
            "drowsy": face_drowsy if drowsy_pairs else None,
        })

    return annotated_frame, cropped_faces, any_drowsy, bool(face_boxes), pose_detected, hands_detected


AGGREGATE_FEATURES = ("age", "gender", "race")  # demographic breakdown scope for crowd counting


def aggregate_demographics(cropped_faces: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    """Whole-image demographic aggregate over already-computed per-face results (age/gender/race
    only) -- no new model, just a tally over cropped_faces' raw_columns. Reuses whatever
    model(s) were already active per feature; if two models are active for the same feature
    (e.g. caffe + ssrnet age), each gets its own independent tally since their label sets/value
    granularity generally differ (same reasoning as DAN vs EfficientNet emotion labels not being
    mixed). Returns {feature: {model_key: {label: count}}}; a feature/model with
    no faces contributing a value for it is simply absent, not a zero-filled entry."""
    totals: dict[str, dict[str, dict[str, int]]] = {feature: {} for feature in AGGREGATE_FEATURES}
    for face in cropped_faces:
        for column, value in face["raw_columns"].items():
            for feature in AGGREGATE_FEATURES:
                prefix = f"{feature}_"
                if not column.startswith(prefix) or not value:
                    continue
                model_key = column[len(prefix):]
                bucket = totals[feature].setdefault(model_key, {})
                bucket[value] = bucket.get(value, 0) + 1
    return {feature: models for feature, models in totals.items() if models}
