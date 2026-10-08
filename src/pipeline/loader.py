"""Model loading and initialization routines for facial analysis backends."""
from __future__ import annotations

import cv2
import numpy as np

try:
    from src.core.constants import (
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
        MIVOLO_MODEL,
        MODEL_DIR,
        RETINAFACE_MODEL,
        SCRFD_FACE_MODEL,
        YOLO_FACE_MODEL,
    )
    from src.core.types import Models
    from src.pipeline.drawing import _silence_native_logs
except ImportError:
    from core.constants import (
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
        MIVOLO_MODEL,
        MODEL_DIR,
        RETINAFACE_MODEL,
        SCRFD_FACE_MODEL,
        YOLO_FACE_MODEL,
    )
    from core.types import Models
    from pipeline.drawing import _silence_native_logs

try:
    from src.model_selection import native_model_selected
except ImportError:
    try:
        from model_selection import native_model_selected
    except ImportError:
        def native_model_selected(var_name: str, key: str) -> bool:
            return True

try:
    with _silence_native_logs():
        import torch
        try:
            from src.nets.dan_model import DAN
        except ImportError:
            from nets.dan_model import DAN
    TORCH_SUPPORTED = True
except ImportError:
    TORCH_SUPPORTED = False

try:
    with _silence_native_logs():
        try:
            from src.nets.deepface_gender import build_gender_model
            from src.nets.deepface_race import build_race_model
            from src.nets.deepface_recognition import build_recognition_model
            from src.nets.mask_model import build_mask_model
            from src.nets.mini_xception_model import build_mini_xception
        except ImportError:
            from nets.deepface_gender import build_gender_model
            from nets.deepface_race import build_race_model
            from nets.deepface_recognition import build_recognition_model
            from nets.mask_model import build_mask_model
            from nets.mini_xception_model import build_mini_xception
    TF_SUPPORTED = True
except ImportError:
    TF_SUPPORTED = False

try:
    with _silence_native_logs():
        try:
            from src.nets.mivolo.inference_wrapper import MiVOLOInference
        except ImportError:
            from nets.mivolo.inference_wrapper import MiVOLOInference
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
        try:
            from src.nets.deep3d_recon import (
                ParametricFaceModel,
                build_deep3d_recon_model,
                load_lm3d_template,
            )
        except ImportError:
            from nets.deep3d_recon import (
                ParametricFaceModel,
                build_deep3d_recon_model,
                load_lm3d_template,
            )
    TORCHVISION_SUPPORTED = True
except ImportError:
    TORCHVISION_SUPPORTED = False

try:
    with _silence_native_logs():
        try:
            from src.nets.face_reaging_model import build_face_reaging_model
        except ImportError:
            from nets.face_reaging_model import build_face_reaging_model
    FACE_REAGING_SUPPORTED = True
except ImportError:
    FACE_REAGING_SUPPORTED = False


def load_models() -> Models:
    """Load every model whose file(s)/dependencies are present and return a Models container."""
    if not FACE_PROTO.exists() or not FACE_MODEL.exists():
        raise FileNotFoundError(
            f"Missing face detector file(s) in {MODEL_DIR}: {FACE_PROTO.name}, {FACE_MODEL.name} (required)."
        )
    face_net = cv2.dnn.readNet(str(FACE_MODEL), str(FACE_PROTO))

    age_nets = {}
    if native_model_selected("AGE_MODEL", "caffe") and AGE_PROTO.exists() and AGE_MODEL.exists():
        age_nets["caffe"] = cv2.dnn.readNet(str(AGE_MODEL), str(AGE_PROTO))
    if native_model_selected("AGE_MODEL", "dex") and DEX_PROTO.exists() and DEX_MODEL.exists():
        age_nets["dex"] = cv2.dnn.readNetFromCaffe(str(DEX_PROTO), str(DEX_MODEL))

    gender_nets = {}
    if native_model_selected("GENDER_MODEL", "caffe") and GENDER_PROTO.exists() and GENDER_MODEL.exists():
        gender_nets["caffe"] = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))

    if native_model_selected("GENDER_MODEL", "deepface") and TF_SUPPORTED and DEEPFACE_GENDER_MODEL.exists():
        gender_nets["deepface"] = build_gender_model(str(DEEPFACE_GENDER_MODEL))

    recognition_nets = {}
    if native_model_selected("RECOGNITION_MODEL", "vggface") and TF_SUPPORTED and DEEPFACE_RECOGNITION_MODEL.exists():
        recognition_nets["vggface"] = build_recognition_model(str(DEEPFACE_RECOGNITION_MODEL))
    if native_model_selected("RECOGNITION_MODEL", "lbph") and hasattr(cv2, "face"):
        recognition_nets["lbph"] = True

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
            if native_model_selected("AGE_MODEL", "mivolo"):
                age_nets["mivolo"] = mivolo_net
            if native_model_selected("GENDER_MODEL", "mivolo"):
                gender_nets["mivolo"] = mivolo_net

    emotion_nets = {}
    if native_model_selected("EMOTION_MODEL", "dan") and TORCH_SUPPORTED and EMOTION_MODEL.exists():
        net = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint = torch.load(str(EMOTION_MODEL), map_location="cpu")
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        emotion_nets["dan"] = net
    if native_model_selected("EMOTION_MODEL", "mini_xception") and TF_SUPPORTED and MINI_XCEPTION_MODEL.exists():
        mini_xception_net = build_mini_xception((64, 64, 1), num_classes=7)
        mini_xception_net.load_weights(str(MINI_XCEPTION_MODEL))
        emotion_nets["mini_xception"] = mini_xception_net
    if native_model_selected("EMOTION_MODEL", "ferplus") and FERPLUS_MODEL.exists():
        emotion_nets["ferplus"] = cv2.dnn.readNetFromONNX(str(FERPLUS_MODEL))
    if native_model_selected("EMOTION_MODEL", "hsemotion") and HSEMOTION_MODEL.exists():
        emotion_nets["hsemotion"] = cv2.dnn.readNetFromONNX(str(HSEMOTION_MODEL))

    fairface_net = None
    if FAIRFACE_MODEL.exists():
        fairface_net = cv2.dnn.readNetFromONNX(str(FAIRFACE_MODEL))
        if native_model_selected("AGE_MODEL", "fairface"):
            age_nets["fairface"] = fairface_net
        if native_model_selected("GENDER_MODEL", "fairface"):
            gender_nets["fairface"] = fairface_net

    race_nets = {}
    if native_model_selected("RACE_MODEL", "fairface") and fairface_net is not None:
        race_nets["fairface"] = fairface_net
    if native_model_selected("RACE_MODEL", "deepface") and TF_SUPPORTED and DEEPFACE_RACE_MODEL.exists():
        race_nets["deepface"] = build_race_model(str(DEEPFACE_RACE_MODEL))

    hair_color_nets = (
        {"colorimetric": True}
        if native_model_selected("HAIR_COLOR_MODEL", "colorimetric")
        else {}
    )
    eye_color_nets = {}
    if EYE_CASCADE_FILE.exists():
        eye_color_nets["colorimetric"] = cv2.CascadeClassifier(str(EYE_CASCADE_FILE))

    liveness_nets: dict = {}
    face_landmarks_nets = {}
    gaze_nets = {}
    face_landmarker_selected = (
        native_model_selected("FACE_LANDMARKS_MODEL", "mediapipe")
        or native_model_selected("LIVENESS_MODEL", "mediapipe")
    )
    if face_landmarker_selected and MEDIAPIPE_SUPPORTED and FACE_LANDMARKER_MODEL.exists():
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
    if native_model_selected("GLASSES_MODEL", "mobilenet") and ONNXRUNTIME_SUPPORTED and GLASSES_MODEL.exists():
        glasses_nets["mobilenet"] = onnxruntime.InferenceSession(str(GLASSES_MODEL), providers=["CPUExecutionProvider"])

    mask_nets = {}
    if native_model_selected("MASK_MODEL", "mobilenetv2") and TF_SUPPORTED and MASK_MODEL.exists():
        mask_nets["mobilenetv2"] = build_mask_model(str(MASK_MODEL))

    colorization_nets = {}
    if (
        native_model_selected("COLORIZATION_MODEL", "eccv16")
        and COLORIZATION_PROTO.exists()
        and COLORIZATION_MODEL.exists()
        and COLORIZATION_PTS.exists()
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
    if native_model_selected("HAND_MODEL", "mediapipe") and MEDIAPIPE_SUPPORTED and HAND_LANDMARKER_MODEL.exists():
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
        and DEEP3D_RECON_MODEL.exists()
        and BFM_MODEL_PATH.exists()
        and BFM_LM3D_PATH.exists()
    ):
        recon_net = build_deep3d_recon_model(str(DEEP3D_RECON_MODEL))
        bfm_model = ParametricFaceModel(str(BFM_MODEL_PATH))
        lm3d_template = load_lm3d_template(str(BFM_DIR))
        reconstruction_3d_nets["deep3d"] = (recon_net, bfm_model, lm3d_template)

    yolo_face_nets = {}
    if native_model_selected("YOLO_FACE_MODEL", "yolo") and ONNXRUNTIME_SUPPORTED and YOLO_FACE_MODEL.exists():
        yolo_face_nets["yolo"] = onnxruntime.InferenceSession(str(YOLO_FACE_MODEL), providers=["CPUExecutionProvider"])

    scrfd_face_nets = {}
    if native_model_selected("SCRFD_FACE_MODEL", "scrfd") and ONNXRUNTIME_SUPPORTED and SCRFD_FACE_MODEL.exists():
        scrfd_face_nets["scrfd"] = onnxruntime.InferenceSession(str(SCRFD_FACE_MODEL), providers=["CPUExecutionProvider"])

    retinaface_nets = {}
    if native_model_selected("RETINAFACE_MODEL", "retinaface") and ONNXRUNTIME_SUPPORTED and RETINAFACE_MODEL.exists():
        retinaface_nets["retinaface"] = onnxruntime.InferenceSession(str(RETINAFACE_MODEL), providers=["CPUExecutionProvider"])

    age_progression_nets = {}
    if native_model_selected("AGE_PROGRESSION_MODEL", "franunet") and FACE_REAGING_SUPPORTED and FACE_REAGING_MODEL.exists():
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
        hair_color_nets,
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
