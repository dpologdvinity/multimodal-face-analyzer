"""Model loading and initialization routines for facial analysis backends."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ..core.constants import (
    AGE_MODEL,
    AGE_PROTO,
    BFM_DIR,
    BFM_LM3D_PATH,
    BFM_MODEL_PATH,
    COLORIZATION_MODEL,
    COLORIZATION_PROTO,
    COLORIZATION_PTS,
    DEEP3D_RECON_MODEL,
    DEEPFACE_GENDER_MODEL,
    DEEPFACE_RACE_MODEL,
    DEEPFACE_RECOGNITION_MODEL,
    DEX_MODEL,
    DEX_PROTO,
    EMOTION_MODEL,
    EYE_CASCADE_FILE,
    FACE_LANDMARKER_MODEL,
    FACE_MODEL,
    FACE_PROTO,
    FACE_REAGING_MODEL,
    FAIRFACE_MODEL,
    FERPLUS_MODEL,
    GENDER_MODEL,
    GENDER_PROTO,
    GLASSES_MODEL,
    HAND_LANDMARKER_MODEL,
    HSEMOTION_MODEL,
    MASK_MODEL,
    MINI_XCEPTION_MODEL,
    MIVOLO_CONFIG,
    MIVOLO_MODEL,
    MODEL_DIR,
    RETINAFACE_MODEL,
    SCRFD_FACE_MODEL,
    YOLO_FACE_MODEL,
)
from ..core.types import Models
from ..demo import is_demo_mode
from ..model_selection import native_model_selected
from .drawing import _silence_native_logs

try:
    with _silence_native_logs():
        import torch

        from ..nets.dan_model import DAN
    TORCH_SUPPORTED = True
except ImportError:
    TORCH_SUPPORTED = False

try:
    with _silence_native_logs():
        from ..nets.deepface_gender import build_gender_model
        from ..nets.deepface_race import build_race_model
        from ..nets.deepface_recognition import build_recognition_model
        from ..nets.mask_model import build_mask_model
        from ..nets.mini_xception_model import build_mini_xception
    TF_SUPPORTED = True
except ImportError:
    TF_SUPPORTED = False

try:
    with _silence_native_logs():
        from ..nets.mivolo.inference_wrapper import MiVOLOInference
    MIVOLO_SUPPORTED = True
except ImportError:
    MIVOLO_SUPPORTED = False

try:
    with _silence_native_logs():
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
    with _silence_native_logs():
        from ..nets.deep3d_recon import (
            ParametricFaceModel,
            build_deep3d_recon_model,
            load_lm3d_template,
        )
    TORCHVISION_SUPPORTED = True
except ImportError:
    TORCHVISION_SUPPORTED = False

try:
    with _silence_native_logs():
        from ..nets.face_reaging_model import build_face_reaging_model
    FACE_REAGING_SUPPORTED = True
except ImportError:
    FACE_REAGING_SUPPORTED = False


_LFS_POINTER_PREFIX = b"version https://git-lfs"
# Caps ONNX Runtime's per-session thread pool when set; unset keeps its default of one per core.
ORT_THREADS_ENV_VAR = "FACE_ANALYZER_ORT_THREADS"


def _ort_session(path: Path):
    """Create a CPU ONNX Runtime session, limited to FACE_ANALYZER_ORT_THREADS threads when set."""
    options = None
    threads = os.environ.get(ORT_THREADS_ENV_VAR, "").strip()
    if threads:
        if not threads.isdecimal() or int(threads) < 1:
            raise ValueError(f"{ORT_THREADS_ENV_VAR} must be an integer of at least 1, got {threads!r}")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _present(*paths: Path) -> bool:
    """Return whether every path is a real file rather than missing or an unpulled git-lfs pointer."""
    for path in paths:
        if not path.is_file():
            return False
        # A clone without git-lfs leaves ~130-byte text stubs in place of the weights;
        # parsing one crashes cv2/torch/keras, so it must count as an absent model.
        with path.open("rb") as fh:
            if fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                return False
    return True


def load_models() -> Models:
    """Load every model whose file(s)/dependencies are present and return a Models container.

    At least one face detector must load; SSD is optional when another detector is present, and
    demo mode never loads it because its weights carry no license statement.
    """
    face_net = None
    if not is_demo_mode() and _present(FACE_PROTO, FACE_MODEL):
        face_net = cv2.dnn.readNet(str(FACE_MODEL), str(FACE_PROTO))

    yolo_face_nets = {}
    if native_model_selected("YOLO_FACE_MODEL", "yolo") and ONNXRUNTIME_SUPPORTED and _present(YOLO_FACE_MODEL):
        yolo_face_nets["yolo"] = _ort_session(YOLO_FACE_MODEL)

    scrfd_face_nets = {}
    if native_model_selected("SCRFD_FACE_MODEL", "scrfd") and ONNXRUNTIME_SUPPORTED and _present(SCRFD_FACE_MODEL):
        scrfd_face_nets["scrfd"] = _ort_session(SCRFD_FACE_MODEL)

    retinaface_nets = {}
    if native_model_selected("RETINAFACE_MODEL", "retinaface") and ONNXRUNTIME_SUPPORTED and _present(RETINAFACE_MODEL):
        retinaface_nets["retinaface"] = _ort_session(RETINAFACE_MODEL)

    if face_net is None and not (yolo_face_nets or scrfd_face_nets or retinaface_nets):
        if is_demo_mode():
            raise FileNotFoundError(
                f"Demo mode needs the RetinaFace detector: {RETINAFACE_MODEL.name} in {MODEL_DIR} and the "
                "onnxruntime package. Run `pip install -r requirements.txt` and "
                "`python tools/fetch_models.py --keys RETINAFACE_MODEL=retinaface`."
            )
        raise FileNotFoundError(
            f"Missing face detector file(s) in {MODEL_DIR}: {FACE_PROTO.name}, {FACE_MODEL.name} (required). "
            "Run `python tools/fetch_models.py --keys` to download them."
        )

    age_nets = {}
    if native_model_selected("AGE_MODEL", "caffe") and _present(AGE_PROTO, AGE_MODEL):
        age_nets["caffe"] = cv2.dnn.readNet(str(AGE_MODEL), str(AGE_PROTO))
    if native_model_selected("AGE_MODEL", "dex") and _present(DEX_PROTO, DEX_MODEL):
        age_nets["dex"] = cv2.dnn.readNetFromCaffe(str(DEX_PROTO), str(DEX_MODEL))

    gender_nets = {}
    if native_model_selected("GENDER_MODEL", "caffe") and _present(GENDER_PROTO, GENDER_MODEL):
        gender_nets["caffe"] = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))

    if native_model_selected("GENDER_MODEL", "deepface") and TF_SUPPORTED and _present(DEEPFACE_GENDER_MODEL):
        gender_nets["deepface"] = build_gender_model(str(DEEPFACE_GENDER_MODEL))

    recognition_nets = {}
    if native_model_selected("RECOGNITION_MODEL", "vggface") and TF_SUPPORTED and _present(DEEPFACE_RECOGNITION_MODEL):
        recognition_nets["vggface"] = build_recognition_model(str(DEEPFACE_RECOGNITION_MODEL))
    if native_model_selected("RECOGNITION_MODEL", "lbph") and hasattr(cv2, "face"):
        recognition_nets["lbph"] = True

    if MIVOLO_SUPPORTED and _present(MIVOLO_MODEL):
        if _present(MIVOLO_CONFIG):
            mivolo_net = MiVOLOInference(
                model_path=str(MIVOLO_MODEL),
                config_path=str(MIVOLO_CONFIG),
                device="cpu",
                half=False,
                verbose=False,
            )
            if native_model_selected("AGE_MODEL", "mivolo"):
                age_nets["mivolo"] = mivolo_net
            if native_model_selected("GENDER_MODEL", "mivolo"):
                gender_nets["mivolo"] = mivolo_net

    emotion_nets = {}
    if native_model_selected("EMOTION_MODEL", "dan") and TORCH_SUPPORTED and _present(EMOTION_MODEL):
        net = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint = torch.load(str(EMOTION_MODEL), map_location="cpu")
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        emotion_nets["dan"] = net
    if native_model_selected("EMOTION_MODEL", "mini_xception") and TF_SUPPORTED and _present(MINI_XCEPTION_MODEL):
        mini_xception_net = build_mini_xception((64, 64, 1), num_classes=7)
        mini_xception_net.load_weights(str(MINI_XCEPTION_MODEL))
        emotion_nets["mini_xception"] = mini_xception_net
    if native_model_selected("EMOTION_MODEL", "ferplus") and _present(FERPLUS_MODEL):
        emotion_nets["ferplus"] = cv2.dnn.readNetFromONNX(str(FERPLUS_MODEL))
    if native_model_selected("EMOTION_MODEL", "hsemotion") and _present(HSEMOTION_MODEL):
        emotion_nets["hsemotion"] = cv2.dnn.readNetFromONNX(str(HSEMOTION_MODEL))

    fairface_net = None
    if _present(FAIRFACE_MODEL):
        fairface_net = cv2.dnn.readNetFromONNX(str(FAIRFACE_MODEL))
        if native_model_selected("AGE_MODEL", "fairface"):
            age_nets["fairface"] = fairface_net
        if native_model_selected("GENDER_MODEL", "fairface"):
            gender_nets["fairface"] = fairface_net

    race_nets = {}
    if native_model_selected("RACE_MODEL", "fairface") and fairface_net is not None:
        race_nets["fairface"] = fairface_net
    if native_model_selected("RACE_MODEL", "deepface") and TF_SUPPORTED and _present(DEEPFACE_RACE_MODEL):
        race_nets["deepface"] = build_race_model(str(DEEPFACE_RACE_MODEL))

    eye_color_nets = {}
    if _present(EYE_CASCADE_FILE):
        eye_color_nets["colorimetric"] = cv2.CascadeClassifier(str(EYE_CASCADE_FILE))

    liveness_nets: dict = {}
    face_landmarks_nets = {}
    gaze_nets = {}
    face_landmarker_selected = (
        native_model_selected("FACE_LANDMARKS_MODEL", "mediapipe")
        or native_model_selected("LIVENESS_MODEL", "mediapipe")
    )
    if face_landmarker_selected and MEDIAPIPE_SUPPORTED and _present(FACE_LANDMARKER_MODEL):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
            output_face_blendshapes=native_model_selected("LIVENESS_MODEL", "mediapipe"),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        with _silence_native_logs():
            landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        if native_model_selected("LIVENESS_MODEL", "mediapipe"):
            liveness_nets["mediapipe"] = landmarker
        if native_model_selected("FACE_LANDMARKS_MODEL", "mediapipe"):
            face_landmarks_nets["mediapipe"] = landmarker
        gaze_nets["mediapipe"] = landmarker

    glasses_nets = {}
    if native_model_selected("GLASSES_MODEL", "mobilenet") and ONNXRUNTIME_SUPPORTED and _present(GLASSES_MODEL):
        glasses_nets["mobilenet"] = _ort_session(GLASSES_MODEL)

    mask_nets = {}
    if native_model_selected("MASK_MODEL", "mobilenetv2") and TF_SUPPORTED and _present(MASK_MODEL):
        mask_nets["mobilenetv2"] = build_mask_model(str(MASK_MODEL))

    colorization_nets = {}
    if (
        native_model_selected("COLORIZATION_MODEL", "eccv16")
        and _present(COLORIZATION_PROTO)
        and _present(COLORIZATION_MODEL)
        and _present(COLORIZATION_PTS)
    ):
        colorization_net = cv2.dnn.readNetFromCaffe(str(COLORIZATION_PROTO), str(COLORIZATION_MODEL))
        pts = np.load(str(COLORIZATION_PTS))
        class8 = colorization_net.getLayerId("class8_ab")
        conv8 = colorization_net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        colorization_net.getLayer(class8).blobs = [pts.astype("float32")]
        colorization_net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        colorization_nets["eccv16"] = colorization_net

    hand_nets = {}
    if native_model_selected("HAND_MODEL", "mediapipe") and MEDIAPIPE_SUPPORTED and _present(HAND_LANDMARKER_MODEL):
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        with _silence_native_logs():
            hand_nets["mediapipe"] = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

    reconstruction_3d_nets = {}
    if (
        native_model_selected("RECONSTRUCTION_3D_MODEL", "deep3d")
        and TORCHVISION_SUPPORTED
        and _present(DEEP3D_RECON_MODEL)
        and _present(BFM_MODEL_PATH)
        and _present(BFM_LM3D_PATH)
    ):
        recon_net = build_deep3d_recon_model(str(DEEP3D_RECON_MODEL))
        bfm_model = ParametricFaceModel(str(BFM_MODEL_PATH))
        lm3d_template = load_lm3d_template(str(BFM_DIR))
        reconstruction_3d_nets["deep3d"] = (recon_net, bfm_model, lm3d_template)

    age_progression_nets = {}
    if native_model_selected("AGE_PROGRESSION_MODEL", "franunet") and FACE_REAGING_SUPPORTED and _present(FACE_REAGING_MODEL):
        age_progression_nets["franunet"] = build_face_reaging_model(str(FACE_REAGING_MODEL))

    return Models(
        face_net,
        age_nets,
        gender_nets,
        emotion_nets,
        race_nets,
        liveness_nets,
        recognition_nets,
        glasses_nets,
        mask_nets,
        eye_color_nets,
        colorization_nets,
        face_landmarks_nets,
        hand_nets,
        reconstruction_3d_nets,
        yolo_face_nets,
        scrfd_face_nets,
        retinaface_nets,
        gaze_nets,
        age_progression_nets,
    )


__all__ = ["load_models"]
