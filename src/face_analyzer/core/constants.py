"""System-wide constants, label sets, model paths, and parameter defaults."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("FACE_ANALYZER_MODEL_DIR") or (BASE_DIR / "models"))

# Model weight and configuration paths
FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
YOLO_FACE_MODEL = MODEL_DIR / "yolov8n_face.onnx"
SCRFD_FACE_MODEL = MODEL_DIR / "scrfd_2.5g_bnkps.onnx"
RETINAFACE_MODEL = MODEL_DIR / "retinaface_mobilenet0.25.onnx"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"
GENDER_PROTO = MODEL_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"
EYE_CASCADE_FILE = MODEL_DIR / "haarcascade_eye.xml"
EMOTION_MODEL = MODEL_DIR / "dan_affecnet7.pth"
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
MIVOLO_CONFIG = MODEL_DIR / "mivolo_v2_config.json"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
GLASSES_MODEL = MODEL_DIR / "glasses_detector.onnx"
MASK_MODEL = MODEL_DIR / "mask_detector.h5"
COLORIZATION_PROTO = MODEL_DIR / "colorization_deploy_v2.prototxt"
COLORIZATION_MODEL = MODEL_DIR / "colorization_release_v2.caffemodel"
COLORIZATION_PTS = MODEL_DIR / "pts_in_hull.npy"
HAND_LANDMARKER_MODEL = MODEL_DIR / "hand_landmarker.task"
BFM_DIR = MODEL_DIR / "BFM"
DEEP3D_RECON_MODEL = MODEL_DIR / "deep3d_recon_resnet50.pth"
BFM_MODEL_PATH = BFM_DIR / "BFM_model_front.mat"
BFM_LM3D_PATH = BFM_DIR / "similarity_Lm3D_all.mat"
FACE_REAGING_MODEL = MODEL_DIR / "face_reaging_unet.pth"

# Normalization & preprocessing constants
MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEX_MEAN_VALUES = (103.939, 116.779, 123.68)
DEX_MAX_AGE_SD = 10.0

# Classification labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LABELS_DAN = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
EMOTION_LABELS_MINI_XCEPTION = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_FERPLUS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
EMOTION_LABELS_HSEMOTION = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
RACE_LABELS_FAIRFACE = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
RACE_LABELS_DEEPFACE = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
FAIRFACE_AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
MASK_LABELS = ["with_mask", "without_mask"]

# Thresholds & margins
RACE_CLOSE_MARGIN = 0.10
RECOGNITION_COSINE_THRESHOLD = 0.68
GLASSES_THRESHOLD = 0.5
GRAYSCALE_CHANNEL_DIFF_THRESHOLD = 3.0

# Gallery, DB, and identity
GALLERY_FILE = BASE_DIR / "gallery" / "known_faces.json"
LBPH_GALLERY_DIR = BASE_DIR / "gallery" / "lbph"
LBPH_FACE_SIZE = (200, 200)
LBPH_CONFIDENCE_THRESHOLD = 80.0
KNOWN_PEOPLE_DIR = BASE_DIR / "known_people"
IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FACES_DB_FILE = BASE_DIR / "db" / "faces.db"
FACES_DIR = BASE_DIR / "faces"
EIGEN_DIR = BASE_DIR / "eigen"
EIGEN_FACE_SIZE = (100, 100)
EIGENFACE_DISTANCE_THRESHOLD = 3000.0

# Face detector options
FACE_DETECTOR_OPTIONS = ["ssd", "yolo", "scrfd", "retinaface"]

# Tracking & detection hyperparameters
IOU_TRACKING_THRESHOLD = 0.3
TRACKING_MAX_MISSED_FRAMES = 10
YOLO_FACE_INPUT_SIZE = 640
YOLO_FACE_STRIDES = (8, 16, 32)
YOLO_FACE_IOU_THRESHOLD = 0.45
SCRFD_FACE_INPUT_SIZE = 640
SCRFD_FACE_STRIDES = (8, 16, 32)
SCRFD_FACE_NUM_ANCHORS = 2
SCRFD_FACE_NMS_THRESHOLD = 0.4
RETINAFACE_INPUT_HEIGHT = 608
RETINAFACE_INPUT_WIDTH = 640
RETINAFACE_STEPS = (8, 16, 32)
RETINAFACE_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
RETINAFACE_VARIANCE = (0.1, 0.2)
RETINAFACE_MEAN = (104, 117, 123)
RETINAFACE_NMS_THRESHOLD = 0.4

# Skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm
]

# Image adjustment slider defaults
IMAGE_ADJUSTMENT_RANGES = {
    "exposure": (-3.0, 3.0, 0.0),
    "brightness": (-100.0, 100.0, 0.0),
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

# Multimodal audio arousal
AUDIO_AROUSAL_WINDOW_SECONDS = 1.5
AUDIO_AROUSAL_QUIET_RMS = 0.02
AUDIO_AROUSAL_LOUD_RMS = 0.15
EMOTION_HIGH_AROUSAL_LABELS = {
    "happy", "happiness", "surprise", "anger", "angry", "fear", "disgust",
}
EMOTION_LOW_AROUSAL_LABELS = {"neutral", "sad", "sadness"}

# Image editing options
IMAGE_OP_OPTIONS = ["intensity", "enhance", "sharpen", "color_correct", "denoise", "bilateral_filter", "wavelet_denoise"]
INTENSITY_METHODS = ["negative", "log", "gamma", "contrast_stretch"]
SHARPEN_METHODS = ["laplacian", "high_boost"]
DENOISE_METHODS = ["gaussian", "median", "nlm"]

# Reference landmarks
FAIRFACE_LANDMARK_INDICES = (263, 362, 33, 133, 1)
FAIRFACE_REFERENCE_LANDMARKS = np.array([
    [0.8595674595992, 0.2134981538014], [0.6460604764104, 0.2289674387677],
    [0.1205750620789, 0.2137274526848], [0.3340850613712, 0.2290642403242],
    [0.4901123135679, 0.6277975316475],
], dtype=np.float64)

# Fusion and model combination keys & weights
FUSED_MODEL_KEY = "fused"
BEST_MODEL_KEY = "best"
HEADLINE_MODEL_KEYS = (FUSED_MODEL_KEY, BEST_MODEL_KEY)
AGE_MODEL_RELIABILITY = ("mivolo", "fairface", "dex", "caffe")
# Ordered by held-out FairFace accuracy; see docs/eval/heldout_fairface.md. Not fused, because
# deepface's near one-hot probabilities dominated any blend and dragged it below fairface alone.
RACE_MODEL_RELIABILITY = ("fairface", "deepface")
GENDER_FUSION_WEIGHTS = {"mivolo": 3.0, "fairface": 2.0, "caffe": 0.5, "deepface": 0.5}
EMOTION_FUSION_WEIGHTS = {"dan": 3.0, "hsemotion": 3.0, "ferplus": 2.0, "mini_xception": 1.0}

FAIRFACE_AGE_RANGES = [
    (0, 2), (3, 9), (10, 19), (20, 29), (30, 39),
    (40, 49), (50, 59), (60, 69), (70, 100),
]
AGE_LIST_RANGES = [
    (0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100),
]

RACE_CANONICAL_LABELS = {
    "white": "White", "black": "Black", "asian": "Asian", "indian": "Indian",
    "latino": "Latino", "middle_eastern": "Middle Eastern",
}
RACE_LABEL_TO_CANONICAL = {
    "white": "white", "black": "black", "indian": "indian",
    "east asian": "asian", "southeast asian": "asian", "asian": "asian",
    "latino_hispanic": "latino", "latino hispanic": "latino",
    "middle eastern": "middle_eastern",
}
EMOTION_CANONICAL = {
    "happy": "happy", "happiness": "happy", "sad": "sad", "sadness": "sad",
    "angry": "angry", "anger": "angry", "surprise": "surprise", "fear": "fear",
    "disgust": "disgust", "neutral": "neutral", "contempt": "contempt",
}

# Upload and cache limits
MAX_UPLOAD_DIMENSION = 2000
PREDICTION_CACHE_MAX_SIZE = 2048

# Aggregation features
AGGREGATE_FEATURES = ("age", "gender", "race")
