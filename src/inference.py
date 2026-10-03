"""Model loading and per-face prediction logic for the Streamlit app (src/app.py).

This module only exists to keep app.py itself from growing unbounded as
more model backends are added.
"""
from __future__ import annotations

import contextlib
import functools
import itertools
import json
import os
import hashlib
import random
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF INFO/WARNING banners (oneDNN, cpu_feature_guard) before TF import
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # no GPU in this environment; skip cuInit probe entirely rather than logging its failure
os.environ.setdefault("GLOG_minloglevel", "2")  # silence glog/absl banners emitted by mediapipe's C++ backend


@contextlib.contextmanager
def _silence_native_logs():
    """Redirect the process's real stderr fd during noisy native-lib calls.

    TF/absl/glog emit some startup banners (oneDNN, cudart_stub, mediapipe
    graph setup) straight to the OS-level stderr fd before Python-side log
    level env vars (TF_CPP_MIN_LOG_LEVEL, GLOG_minloglevel) take effect, so
    those env vars alone don't silence them -- only an fd-level redirect does.
    """
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, fd)
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(devnull_fd)
        os.close(saved_fd)

import warnings

# Vendored net code (src/nets/) uses APIs (torch.jit.script, torchvision positional
# `weights`, Keras `input_shape` on non-Input layers) that only emit deprecation noise --
# not actionable here since that code isn't ours to change. Silence before those modules import.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*input_shape.*")

import cv2
import numpy as np

try:
    from .liveness import (
        LivenessTracker,
        assess_static_liveness,
        blink_score_from_landmarker,
        texture_artifact_score,
    )
except ImportError:  # app.py runs with src/ on sys.path in the container
    from liveness import (
        LivenessTracker,
        assess_static_liveness,
        blink_score_from_landmarker,
        texture_artifact_score,
    )

try:
    from .model_selection import native_model_selected
except ImportError:  # app.py runs with src/ on sys.path in the container
    from model_selection import native_model_selected

try:
    with _silence_native_logs():
        import torch
        from nets.dan_model import DAN
    TORCH_SUPPORTED = True
except ImportError:
    TORCH_SUPPORTED = False

try:
    with _silence_native_logs():
        from nets.deepface_race import build_race_model
        from nets.deepface_gender import build_gender_model
        from nets.deepface_recognition import build_recognition_model
        from nets.mini_xception_model import build_mini_xception
        from nets.mask_model import build_mask_model
    TF_SUPPORTED = True
except ImportError:
    TF_SUPPORTED = False

try:
    with _silence_native_logs():
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
        from nets.deep3d_recon import (
            build_deep3d_recon_model, ParametricFaceModel, load_lm3d_template,
            landmarks_5pt_from_mediapipe, reconstruct_face_3d, mesh_to_obj_str,
        )
    TORCHVISION_SUPPORTED = True
except ImportError:
    TORCHVISION_SUPPORTED = False

try:
    with _silence_native_logs():
        from nets.face_reaging_model import build_face_reaging_model, age_progress_face
    FACE_REAGING_SUPPORTED = True
except ImportError:
    FACE_REAGING_SUPPORTED = False

try:
    from .core import (
        BoundingBox,
        Detection,
        FaceResult,
        Models,
        is_grayscale_frame,
        crop_region,
        face_crop_bounds,
        apply_image_adjustments,
        _is_skin_hsv,
        _adjust_exposure,
        _adjust_brightness,
        _adjust_contrast,
        _adjust_tone_region,
        _adjust_black_point,
        _adjust_saturation,
        _adjust_sharpness,
        _adjust_definition,
        _adjust_noise_reduction,
    )
    from .core.constants import *
except ImportError:
    from core import (
        BoundingBox,
        Detection,
        FaceResult,
        Models,
        is_grayscale_frame,
        crop_region,
        face_crop_bounds,
        apply_image_adjustments,
        _is_skin_hsv,
        _adjust_exposure,
        _adjust_brightness,
        _adjust_contrast,
        _adjust_tone_region,
        _adjust_black_point,
        _adjust_saturation,
        _adjust_sharpness,
        _adjust_definition,
        _adjust_noise_reduction,
    )
    from core.constants import *

try:
    from .detectors import (
        BaseFaceDetector,
        detect_faces_ssd,
        detect_faces_yolo,
        detect_faces_scrfd,
        detect_faces_retinaface,
        detect_faces,
    )
    from .detectors.yolo import _yolo_letterbox, _yolo_softmax
    from .detectors.scrfd import _scrfd_distance2bbox
    from .detectors.retinaface import _retinaface_priors, _retinaface_decode
except ImportError:
    from detectors import (
        BaseFaceDetector,
        detect_faces_ssd,
        detect_faces_yolo,
        detect_faces_scrfd,
        detect_faces_retinaface,
        detect_faces,
    )
    from detectors.yolo import _yolo_letterbox, _yolo_softmax
    from detectors.scrfd import _scrfd_distance2bbox
    from detectors.retinaface import _retinaface_priors, _retinaface_decode



try:
    from .attributes import (
        _lock_for,
        colorize_frame,
        maybe_colorize,
        apply_geometric_transform,
        apply_intensity_transform,
        apply_enhance,
        apply_sharpen,
        apply_color_correct,
        apply_denoise,
        apply_bilateral_filter,
        apply_wavelet_denoise,
        apply_image_op,
        run_3d_reconstruction,
        run_age_progression,
        _margin_align,
        fairface_landmarks_from_mediapipe,
        align_face_with_landmarks,
        _estimate_roll_angle,
        _rotate_region,
        _softmax,
        _format_race_label,
        _fairface_forward,
        fairface_probabilities,
        fairface_race_label,
        fairface_gender_label,
        fairface_age_label,
        predict_race_fairface,
        deepface_probabilities,
        predict_race_deepface,
        caffe_probabilities,
        predict_age_caffe,
        crop_face_dex,
        dex_age_estimate,
        format_dex_age,
        predict_age_dex,
        mivolo_estimate,
        mivolo_age_estimate,
        predict_age_mivolo,
        predict_age_fairface,
        predict_gender_caffe,
        predict_gender_mivolo,
        predict_gender_fairface,
        predict_gender_deepface,
        predict_emotion_dan,
        predict_emotion_mini_xception,
        predict_emotion_ferplus,
        predict_emotion_hsemotion,
        audio_frame_to_mono_float,
        classify_voice_arousal,
        _emotion_arousal_category,
        fuse_voice_and_emotion,
        VoiceFaceFusion,
        predict_texture_artifact_score,
        predict_glasses_mobilenet,
        predict_mask_mobilenetv2,
        predict_hair_color_colorimetric,
        predict_eye_color_colorimetric,
    )
    from .attributes._lock import _NET_LOCKS, _NET_LOCKS_GUARD
except ImportError:
    from attributes import (
        _lock_for,
        colorize_frame,
        maybe_colorize,
        apply_geometric_transform,
        apply_intensity_transform,
        apply_enhance,
        apply_sharpen,
        apply_color_correct,
        apply_denoise,
        apply_bilateral_filter,
        apply_wavelet_denoise,
        apply_image_op,
        run_3d_reconstruction,
        run_age_progression,
        _margin_align,
        fairface_landmarks_from_mediapipe,
        align_face_with_landmarks,
        _estimate_roll_angle,
        _rotate_region,
        _softmax,
        _format_race_label,
        _fairface_forward,
        fairface_probabilities,
        fairface_race_label,
        fairface_gender_label,
        fairface_age_label,
        predict_race_fairface,
        deepface_probabilities,
        predict_race_deepface,
        caffe_probabilities,
        predict_age_caffe,
        crop_face_dex,
        dex_age_estimate,
        format_dex_age,
        predict_age_dex,
        mivolo_estimate,
        mivolo_age_estimate,
        predict_age_mivolo,
        predict_age_fairface,
        predict_gender_caffe,
        predict_gender_mivolo,
        predict_gender_fairface,
        predict_gender_deepface,
        predict_emotion_dan,
        predict_emotion_mini_xception,
        predict_emotion_ferplus,
        predict_emotion_hsemotion,
        audio_frame_to_mono_float,
        classify_voice_arousal,
        _emotion_arousal_category,
        fuse_voice_and_emotion,
        VoiceFaceFusion,
        predict_texture_artifact_score,
        predict_glasses_mobilenet,
        predict_mask_mobilenetv2,
        predict_hair_color_colorimetric,
        predict_eye_color_colorimetric,
    )
    from attributes._lock import _NET_LOCKS, _NET_LOCKS_GUARD

try:
    from .fusion import *
except ImportError:
    from fusion import *


# Shared across the process (and every Streamlit session) -- per-feature tasks are short-lived
# native calls (cv2.dnn/TF/torch all release the GIL during their own compute), so a modest
# pool sized off the CPU count lets independent features (different nets) genuinely overlap
# without oversubscribing a CPU-only deployment.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 4)), thread_name_prefix="inference")
# Models dataclass (including liveness_nets: dict, etc.) is imported from core.types



def load_models() -> Models:
    """Load every model whose file(s)/dependencies are present. Face detection is required;
    age, gender, and emotion are each optional per-model-key -- a model is only
    present in its feature's dict if it loaded successfully, so the app degrades gracefully
    to whichever models were built in. Which of the loaded models are actually used per frame
    is chosen at runtime by the caller (see analyze_frame's active_* arguments)."""
    if not FACE_PROTO.exists() or not FACE_MODEL.exists():
        raise FileNotFoundError(f"Missing face detector file(s) in {MODEL_DIR}: {FACE_PROTO.name}, {FACE_MODEL.name} (required).")
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

    # Colorimetric heuristics need no model file, no dependency beyond OpenCV -- always
    # available. hair_color has no further precondition; eye_color uses haarcascade_eye.xml.
    hair_color_nets = (
        {"colorimetric": True}
        if native_model_selected("HAIR_COLOR_MODEL", "colorimetric")
        else {}
    )
    eye_color_nets = {}
    if EYE_CASCADE_FILE.exists():
        eye_color_nets["colorimetric"] = cv2.CascadeClassifier(str(EYE_CASCADE_FILE))

    liveness_nets = {}
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
    if native_model_selected("COLORIZATION_MODEL", "eccv16") and COLORIZATION_PROTO.exists() and COLORIZATION_MODEL.exists() and COLORIZATION_PTS.exists():
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
    if native_model_selected("RECONSTRUCTION_3D_MODEL", "deep3d") and TORCHVISION_SUPPORTED and DEEP3D_RECON_MODEL.exists() and BFM_MODEL_PATH.exists() and BFM_LM3D_PATH.exists():
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
        face_net, age_nets, gender_nets, emotion_nets, race_nets, liveness_nets, recognition_nets,
        glasses_nets, mask_nets, hair_color_nets, eye_color_nets, colorization_nets,
        face_landmarks_nets, hand_nets, reconstruction_3d_nets, yolo_face_nets, scrfd_face_nets, retinaface_nets,
        gaze_nets, age_progression_nets,
    )


# Face detection backends (SSD, YOLO, SCRFD, RetinaFace) and detect_faces factory
# are imported from .detectors above.


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
    """Multi-face tracker that assigns stable IDs across video frames via greedy IoU matching.

    Without a tracker, each frame renumbers faces 1..N by detection order, causing IDs to
    flicker as a face moves. A tracker instead keeps IDs stable: once a face is detected,
    its ID persists across frames as long as a high-IoU match exists within the tracking
    threshold and missed-frame window.

    Matching strategy: greedy one-to-one assignment (highest IoU first) between tracks and
    detections. Tracks can survive brief occlusion (up to TRACKING_MAX_MISSED_FRAMES frames
    without a matching detection before the ID is freed).

    Thread-safe: one instance per video stream. Safe to call update()/reset() from different
    threads (e.g., streamlit-webrtc callback vs. main thread) via an internal lock.
    """

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

# is_grayscale_frame imported from core.image_utils


# colorize_frame and maybe_colorize imported from attributes.transformers

# apply_image_adjustments and helpers imported from core.image_utils



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


# Voice and emotion fusion imported from attributes.emotion


# --- Rectangle-select geometric transforms (ideas/transform.md, ideas/geo-transform.md) ---
# Applied to an arbitrary user-selected sub-rectangle of the whole image, independent of face
# detection -- these operate on any region, not just faces.
GEOMETRIC_TRANSFORM_OPTIONS = ["translate", "reflect", "rotate", "scale", "shear"]

# crop_region and face_crop_bounds imported from core.image_utils



# Geometric transforms and image ops imported from attributes.transformers


# Age, gender, emotion, and alignment functions imported from attributes


# Fusion (ranking and ensembles) are imported from .fusion above.


# Race and FairFace/DeepFace prediction functions imported from attributes.race and attributes.gender


def compute_face_embedding(net, face_bgr: np.ndarray) -> np.ndarray:
    """Compute normalized VGGFace embedding for identity recognition (cosine distance)."""
    # VGGFace input: 224x224 BGR, unnormalized [0,255].
    face_resized = cv2.resize(face_bgr, (224, 224)).astype(np.float32)
    with _lock_for(net):
        emb = net(face_resized[np.newaxis, ...], training=False).numpy().flatten()
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
    """Load enrolled face embeddings from gallery/known_faces.json (VGGFace embeddings, pre-normalized)."""
    if not GALLERY_FILE.exists():
        return {}
    raw = json.loads(GALLERY_FILE.read_text())
    return {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}


def save_gallery(gallery: dict) -> None:
    """Persist enrolled face embeddings to gallery/known_faces.json (JSON format)."""
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
    """Compute a SHA256 hash of the LBPH gallery structure (file paths, sizes, mtimes) to detect changes."""
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


# Modern phone cameras commonly produce 3000-4000px-wide photos. Every downstream detector
# resizes its own input internally (SSD to 300x300, YOLO/SCRFD/RetinaFace to their own fixed
# input size), so feeding them a multi-thousand-pixel source buys no detection quality -- it
# only multiplies the cost of every full-frame op that runs BEFORE that internal resize
# (decode, color conversion, cv2.dnn's own resize, drawing overlays, content hashing for the
# detection/prediction caches). Capping the longer side here is quality-neutral for anything
# feeding those fixed-size model inputs.
MAX_UPLOAD_DIMENSION = 2000


def decode_image_bytes(file_bytes: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Decode uploaded image bytes (JPEG/PNG/WebP/etc) into BGR ndarray, downscaled if huge."""
    encoded = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("The uploaded file is empty or could not be read.")
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("The uploaded file is not a valid supported image.")
    height, width = frame.shape[:2]
    longer_side = max(height, width)
    if longer_side > MAX_UPLOAD_DIMENSION:
        scale = MAX_UPLOAD_DIMENSION / longer_side
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return frame


def predict_identity_lbph(recognizer, label_names: list[str], face_bgr: np.ndarray) -> tuple[str, float] | None:
    """LOWER LBPH confidence is a better match (opposite convention from vggface's cosine
    similarity) -- accept only below LBPH_CONFIDENCE_THRESHOLD."""
    with _lock_for(recognizer):
        label, confidence = recognizer.predict(_lbph_preprocess(face_bgr))
    return (label_names[label], confidence) if confidence < LBPH_CONFIDENCE_THRESHOLD else None


# Per-face classifier output cache, keyed on exact preprocessed pixel bytes (not identity embedding).
# Cache key invariant: identical bytes -> identical deterministic output. Embedding-based keys fail
# because adjusted/re-cropped versions of "the same" face have different bytes and may legitimately
# produce different classifications. This matters because Streamlit re-runs the entire script on any
# widget interaction, recomputing every face classification despite no image/adjustment/model changes.
# Module-level, unlocked: concurrent sessions may redundantly recompute the same key, but dict get/set
# are GIL-atomic so no corruption risk. LRU-evicted to prevent unbounded memory growth.
PREDICTION_CACHE_MAX_SIZE = 2048
_PREDICTION_CACHE: "OrderedDict[tuple, object]" = OrderedDict()


def _cached_face_predict(feature: str, model_key: str, face_bgr: np.ndarray, predict_fn, *args):
    """Memoize a predict_*(net, face, ...) call on (feature, model_key, hash(image bytes)).
    Despite the name, `face_bgr` may be a per-face crop OR a whole frame (face detection,
    hand landmarks) -- the cache key only depends on that array's bytes, not what it depicts.
    Predictors that take the whole frame plus a box (fairface) are keyed on the face crop cut
    from that same frame and box: the crop is derived from them 1:1, so its bytes identify the
    same input without hashing the entire frame.

    NOTE: only `feature`/`model_key`/`face_bgr` are part of the cache key -- predict_fn's other
    *args are NOT hashed. Any caller whose behavior also depends on another argument (e.g.
    conf_threshold for detection) MUST fold that value into `model_key` itself, or a rerun with
    a changed argument will wrongly return a stale cached result."""
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


# _sanitize_column_name and _gather_face_results imported from .fusion


def _crop_and_resize_for_eigenfaces(face_bgr: np.ndarray) -> np.ndarray:
    """Preprocess for eigenfaces: grayscale, center-square crop, resize to standard size.

    Used during save_face() and match_face_eigenfaces() to ensure training and query
    images have consistent preprocessing for PCA projection.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    side = min(h, w)
    cy, cx = h // 2, w // 2
    zoomed = gray[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2]
    return cv2.resize(zoomed, EIGEN_FACE_SIZE)


def save_face(face_bgr: np.ndarray, raw_columns: dict[str, str]) -> int:
    """Save one classified face into the database with sparse columns and image artifacts.

    Persists: (1) a DB row with sparse columns (columns created on-demand per model/feature
    that actually contributes a value), (2) color crop to faces/{id}.jpg, (3) grayscale/zoomed
    crop to eigen/{id}.jpg for eigenfaces training/matching. Returns a unique random face_id.
    Sparse column design: a model never run on any saved face doesn't create a column,
    keeping the schema flexible and compact as features are added/removed.
    """
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    face_id = None
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

        for _ in range(20):
            candidate = random.randint(1, 999_999)
            if any(path.exists() for path in (
                FACES_DIR / f"{candidate}.jpg",
                EIGEN_DIR / f"{candidate}.jpg",
            )):
                continue
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
        # Write and verify both artifacts before committing the row. A failed image write
        # must not leave a database record that the file-backed search features cannot load.
        face_path = FACES_DIR / f"{face_id}.jpg"
        eigen_path = EIGEN_DIR / f"{face_id}.jpg"
        if not cv2.imwrite(str(face_path), face_bgr):
            raise OSError(f"Could not write saved face image: {face_path}")
        if not cv2.imwrite(str(eigen_path), _crop_and_resize_for_eigenfaces(face_bgr)):
            raise OSError(f"Could not write eigenface image: {eigen_path}")
        conn.commit()
    except Exception:
        conn.rollback()
        if face_id is not None:
            for path in (FACES_DIR / f"{face_id}.jpg", EIGEN_DIR / f"{face_id}.jpg"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        raise
    finally:
        conn.close()

    return face_id


def _load_eigen_images() -> tuple[list[int], np.ndarray]:
    """Load all saved eigenfaces from eigen/ into memory as flattened vectors.

    Returns (ids, data) where data is (M, EIGEN_FACE_SIZE[0]*EIGEN_FACE_SIZE[1]) float64.
    """
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
    """Train PCA-based face subspace from saved eigenfaces (faces ever clicked SAVE on).

    Re-trains fresh on every matching attempt (no persisted model), matching eigenfaces.md.
    Uses the trick A @ A.T instead of A.T @ A for covariance (M << N*N for saved face count).
    Returns (ids, mean_face, eigenfaces, weights) or None if fewer than 2 saved faces exist.
    """
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
    """Project a query face into eigenspace and find the closest training sample (L2 distance)."""
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
    """Run MediaPipe FaceLandmarker on one face crop with thread-safe locking."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    with _lock_for(landmarker):
        with _silence_native_logs():
            return landmarker.detect(mp_image)


# Accessory and colorimetric prediction functions imported from attributes.accessories


def predict_face_landmarks_mediapipe(landmarker, face_bgr: np.ndarray, result=None) -> list[tuple[float, float]] | None:
    """MediaPipe FaceLandmarker face-mesh points used by the landmark and gaze features.
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


def _record_model_latency(metrics: dict | None, feature: str, model: str, started: float) -> None:
    """Append per-model inference latency (ms) to metrics dict for performance monitoring."""
    if metrics is None:
        return
    metrics.setdefault("model_latency_ms", {}).setdefault(f"{feature}/{model}", []).append(
        (time.perf_counter() - started) * 1000
    )


# 3D reconstruction and age progression imported from attributes.transformers


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
        with _silence_native_logs():
            result = landmarker.detect(mp_image)
    return [
        [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand]
        for hand in result.hand_landmarks
    ]


def draw_hand_landmarks(frame: np.ndarray, hands: list[list[tuple[int, int]]]) -> None:
    """Draw each hand's skeleton (joints + connecting bones) directly onto frame, same
    HUD palette consistent with the other landmark overlays."""
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
    active_race: set,
    active_recognition: set,
    gallery: dict,
    active_glasses: set,
    active_mask: set,
    active_hair_color: set,
    active_eye_color: set,
    active_face_landmarks: set,
    active_hands: set,
    active_gaze: set,
    global_adjustments: dict,
    face_adjustments: dict,
    face_detector: str = "yolo",
    metrics: dict | None = None,
    tracker: "FaceTracker | None" = None,
    liveness_tracker: "LivenessTracker | None" = None,
    active_liveness: set | None = None,
):
    """Detect faces and run inference for whichever model keys are active per feature.
    Multiple active models for the same feature (e.g. active_age = {"caffe", "fairface"})
    all run and are shown together. No Streamlit calls (safe for background threads).

    global_adjustments apply to the whole frame first, before face detection even runs --
    every output derived from this call (the annotated image, every face crop, every
    classification) sees the adjusted pixels. face_adjustments apply again, per detected
    face, to that face's own crop only, after detection but before classification -- they
    affect just that one face's thumbnail/attributes, not the shared frame or other faces.

    face_detector picks which face detection backend runs (unlike every other feature,
    exactly one runs per frame -- running two detectors and merging their boxes would just
    produce duplicate/overlapping faces, not a meaningfully combined result). "yolo"/"scrfd"/
    "retinaface" fall back to "ssd" (the always-required detector) if that model isn't loaded.

    tracker (#2) is optional and stays None for single-image callers (upload/snapshot have no
    "next frame" for an ID to persist into). When a FaceTracker is passed -- video/webcam LIVE
    mode only -- each face's dict also carries a stable "track_id" (see FaceTracker), and the
    number burned into the annotated frame is that track_id instead of this frame's
    detection-order position, so tracking is visible, not just data the caller ignores).

    liveness_tracker is only ever passed by video/webcam LIVE mode -- a single static image has
    no blink transitions to observe, so liveness is unavailable there by design (not just
    unchecked): static callers (upload/snapshot) never pass a tracker, and analyze_frame skips
    liveness entirely -- no "liveness" pairs, no LivenessResult -- whenever liveness_tracker is
    None, regardless of active_liveness. active_liveness additionally gates it off within LIVE
    mode itself (unchecked box = skipped); omitted callers default to every loaded backend."""
    if active_liveness is None:
        active_liveness = set(models.liveness_nets)
    if global_adjustments and any(global_adjustments.values()):
        frame = apply_image_adjustments(frame, global_adjustments)

    annotated_frame = frame.copy()
    yolo_net = models.yolo_face_nets.get("yolo")
    scrfd_net = models.scrfd_face_nets.get("scrfd")
    retinaface_net = models.retinaface_nets.get("retinaface")
    # Detection runs on the whole frame and is independent of which classifiers are active,
    # but Streamlit reruns this whole function on every unrelated widget interaction (a model
    # checkbox toggle, an export button) even when the frame bytes are unchanged. Route through
    # _cached_face_predict (it hashes whatever ndarray it's given, not just face crops) so a
    # rerun with the same frame + conf_threshold + detector reuses last run's boxes instead of
    # re-running the detector network. Video/webcam frames differ every call, so this is a
    # pure win there too -- worst case is one wasted hash per frame, never a wrong cache hit.
    # conf_threshold is folded into model_key (not just passed as an arg) because
    # _cached_face_predict's cache key is (feature, model_key, frame hash) -- it does not hash
    # predict_fn's *args, so a bare "yolo" key would wrongly reuse boxes from a different
    # confidence threshold.
    if face_detector == "yolo" and yolo_net is not None:
        face_boxes = _cached_face_predict("face_detection", f"yolo:{conf_threshold}", frame, detect_faces_yolo, yolo_net, frame, conf_threshold)
    elif face_detector == "scrfd" and scrfd_net is not None:
        face_boxes = _cached_face_predict("face_detection", f"scrfd:{conf_threshold}", frame, detect_faces_scrfd, scrfd_net, frame, conf_threshold)
    elif face_detector == "retinaface" and retinaface_net is not None:
        face_boxes = _cached_face_predict("face_detection", f"retinaface:{conf_threshold}", frame, detect_faces_retinaface, retinaface_net, frame, conf_threshold)
    else:
        face_boxes = _cached_face_predict("face_detection", f"ssd:{conf_threshold}", frame, detect_faces, models.face_net, frame, conf_threshold)
    track_ids = tracker.update(face_boxes) if tracker is not None else [None] * len(face_boxes)
    cropped_faces = []
    hands_detected = False
    hand_net = models.hand_nets.get("mediapipe")
    if hand_net is not None and "mediapipe" in active_hands:
        hands = _cached_face_predict("hand_landmarks", "mediapipe", frame, detect_hand_landmarks_mediapipe, hand_net, frame)
        if hands:
            hands_detected = True
            draw_hand_landmarks(annotated_frame, hands)

    need_blob227 = ("caffe" in active_age and "caffe" in models.age_nets) or \
                   ("caffe" in active_gender and "caffe" in models.gender_nets)

    eye_cascade = models.eye_color_nets.get("colorimetric")

    # Train LBPH once per frame, not per face -- prevents M x face_count redundant training calls.
    lbph_trained = train_lbph_recognizer() if "lbph" in active_recognition and models.recognition_nets.get("lbph") else None

    for idx, ((x1, y1, x2, y2), track_id) in enumerate(zip(face_boxes, track_ids), 1):
        # Straighten tilted heads (in-plane roll) before classification. Asymmetric head poses
        # can confuse some classifiers. No new model dependency -- reuses the eye cascade.
        crop_frame, (cx1, cy1, cx2, cy2) = frame, (x1, y1, x2, y2)
        if eye_cascade is not None:
            probe = frame[max(0, y1 - 20):min(y2 + 20, frame.shape[0]), max(0, x1 - 20):min(x2 + 20, frame.shape[1])]
            # Same content-addressed memoization as face detection above: identical probe
            # bytes always yield the same angle, so a rerun with an unchanged face region
            # (e.g. from a toggled model checkbox) skips the Haar cascade re-scan.
            angle = _cached_face_predict("roll_angle", "haarcascade", probe, _estimate_roll_angle, probe, eye_cascade) if probe.size else None
            if angle is not None and abs(angle) > 3:  # skip work for near-level faces
                crop_frame, (cx1, cy1, cx2, cy2) = _rotate_region(frame, (x1, y1, x2, y2), angle)

        x1_crop, y1_crop, x2_crop, y2_crop = face_crop_bounds(
            (cx1, cy1, cx2, cy2), crop_frame.shape[:2],
        )

        face = crop_frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        if face_adjustments and any(face_adjustments.values()):
            face = apply_image_adjustments(face, face_adjustments)

        # Gaze, head pose, and drawn landmarks all consume the
        # same MediaPipe FaceLandmarker result. Detect once before dispatching
        # feature tasks so the shared model is not run repeatedly per crop.
        # Liveness only makes sense with a stable track to watch blinks across frames --
        # video/webcam LIVE mode only (see analyze_frame docstring). Static callers (upload/
        # snapshot) never pass liveness_tracker, so liveness is skipped there regardless of
        # active_liveness.
        run_liveness = liveness_tracker is not None and "mediapipe" in active_liveness
        liveness_net = models.liveness_nets.get("mediapipe") if run_liveness else None
        face_landmarker = (
            liveness_net if liveness_net is not None
            else models.face_landmarks_nets.get("mediapipe")
        )
        needs_face_landmarks = (
            "mediapipe" in active_gaze
            or "mediapipe" in active_face_landmarks
            or face_landmarker is not None
        )
        landmarker_result = (
            _detect_face_landmarker(face_landmarker, face)
            if face_landmarker is not None and needs_face_landmarks
            else None
        )
        fairface_landmarks = None
        points = None
        if landmarker_result is not None and face_landmarker is not None:
            points = predict_face_landmarks_mediapipe(face_landmarker, face, landmarker_result)
            if points is not None:
                local_landmarks = fairface_landmarks_from_mediapipe(points, face.shape[1], face.shape[0])
                if local_landmarks is not None:
                    local_landmarks += np.array([x1_crop, y1_crop], dtype=np.float32)
                    fairface_landmarks = local_landmarks
        texture_score = predict_texture_artifact_score(face) if run_liveness else 0.0
        blink_score = blink_score_from_landmarker(landmarker_result) if run_liveness else None

        blob227 = None
        if need_blob227:
            blob227 = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # #19: each feature below is independent of every other feature for this face (they
        # read the same face/crop_frame/blob227 but never share mutable state with each
        # other -- _cached_face_predict's cache and _record_model_latency's metrics dict are
        # both documented/verified safe for this, see their own docstrings/comments), so they
        # run concurrently on _INFERENCE_EXECUTOR instead of one after another. Shared model
        # instances (e.g. one fairface net backing age, gender, and race, or one
        # MediaPipe landmarker backing gaze/face_landmarks is made safe for this
        # by _lock_for(), applied at each net's actual setInput/forward/predict/detect call
        # site (see the top of this file) -- concurrent calls onto the SAME net serialize
        # there, while calls onto DIFFERENT nets still overlap for real.
        # MiVOLO answers age and gender from ONE forward pass and is by far the slowest backend
        # (seconds per face), so run it once here rather than once inside each feature's task.
        mivolo_net = models.age_nets.get("mivolo") or models.gender_nets.get("mivolo")
        mivolo_wanted = "mivolo" in active_age or "mivolo" in active_gender
        mivolo_result = None
        mivolo_age_result = None
        if mivolo_net is not None and mivolo_wanted:
            if "mivolo" in active_age and "mivolo" not in active_gender:
                mivolo_age_result = _cached_face_predict(
                    "mivolo_age", "face", face, mivolo_age_estimate, mivolo_net, face,
                )
            else:
                mivolo_result = _cached_face_predict(
                    "mivolo", "face", face, mivolo_estimate, mivolo_net, face,
                )

        def _age_task():
            """Per-model age labels, plus each model's estimate in years for fusion."""
            pairs, estimates = [], {}
            for key in active_age:
                net = models.age_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "caffe":
                    probs = _cached_face_predict("age_probs", key, face, caffe_probabilities, net, blob227)
                    bucket = int(np.argmax(probs))
                    value = AGE_LIST[bucket]
                    estimates[key] = float(np.mean(AGE_LIST_RANGES[bucket]))
                elif key == "fairface":
                    probs = _cached_face_predict(
                        "age_probs", key, face, fairface_probabilities,
                        net, crop_frame, (cx1, cy1, cx2, cy2), "age_output", fairface_landmarks,
                    )
                    value = fairface_age_label(probs)
                    estimates[key] = float(np.mean(FAIRFACE_AGE_RANGES[int(np.argmax(probs))]))
                elif key == "dex":
                    dex_face = crop_face_dex(frame, (x1, y1, x2, y2))
                    if face_adjustments and any(face_adjustments.values()):
                        dex_face = apply_image_adjustments(dex_face, face_adjustments)
                    estimate = _cached_face_predict("age_estimate", key, dex_face, dex_age_estimate, net, dex_face)
                    value = format_dex_age(estimate)
                    if estimate is not None:
                        estimates[key] = estimate[0]
                elif key == "mivolo":
                    if mivolo_result is None and mivolo_age_result is None:
                        continue
                    age = mivolo_result[0] if mivolo_result is not None else mivolo_age_result
                    value = f"{age:.0f}"
                    estimates[key] = age
                pairs.append((key, value))
                _record_model_latency(metrics, "age", key, started)
            return pairs, select_age(estimates)

        def _gender_task():
            """Per-model gender labels, plus each model's P(Male) for fusion."""
            pairs, male_probabilities = [], {}
            for key in active_gender:
                net = models.gender_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "caffe":
                    probs = _cached_face_predict("gender_probs", key, face, caffe_probabilities, net, blob227)
                    value = GENDER_LIST[int(np.argmax(probs))]
                    male_probabilities[key] = float(probs[0] / (probs.sum() or 1.0))
                elif key == "deepface":
                    probs = _cached_face_predict("gender_probs", key, face, deepface_probabilities, net, face)
                    value = "Male" if np.argmax(probs) == 1 else "Female"
                    male_probabilities[key] = float(probs[1] / (probs.sum() or 1.0))
                elif key == "fairface":
                    probs = _cached_face_predict(
                        "gender_probs", key, face, fairface_probabilities,
                        net, crop_frame, (cx1, cy1, cx2, cy2), "gender_output", fairface_landmarks,
                    )
                    value = fairface_gender_label(probs)
                    male_probabilities[key] = float(probs[0])
                elif key == "mivolo":
                    if mivolo_result is None:
                        continue
                    value = mivolo_result[1]
                    # MiVOLO exposes only a label, so it votes at full confidence either way.
                    male_probabilities[key] = 1.0 if value == "Male" else 0.0
                pairs.append((key, value))
                _record_model_latency(metrics, "gender", key, started)
            return pairs, fuse_gender(male_probabilities)

        def _emotion_task():
            """Per-model emotion labels, plus the fused weighted-vote label."""
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
                    # FERPlus performs best on the tight detector crop; other models keep
                    # the shared padded crop used by their training preprocessing.
                    ferplus_face = crop_region(frame, x1, y1, x2, y2)
                    if face_adjustments and any(face_adjustments.values()):
                        ferplus_face = apply_image_adjustments(ferplus_face, face_adjustments)
                    value = _cached_face_predict(
                        "emotion", key, ferplus_face, predict_emotion_ferplus, net, ferplus_face,
                    )
                else:
                    value = _cached_face_predict("emotion", key, face, predict_emotion_hsemotion, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "emotion", key, started)
            return pairs, fuse_emotion(dict(pairs))

        def _race_task():
            """Per-model race labels, plus each model's canonical distribution for fusion."""
            pairs, distributions = [], {}
            for key in active_race:
                net = models.race_nets.get(key)
                if net is None:
                    continue
                started = time.perf_counter()
                if key == "fairface":
                    probs = _cached_face_predict(
                        "race_probs", key, face, fairface_probabilities,
                        net, crop_frame, (cx1, cy1, cx2, cy2), "race_output", fairface_landmarks,
                    )
                    value = fairface_race_label(probs)
                    distributions[key] = canonical_race_probabilities(probs, RACE_LABELS_FAIRFACE)
                else:
                    probs = _cached_face_predict("race_probs", key, face, deepface_probabilities, net, face)
                    value = _format_race_label(probs, RACE_LABELS_DEEPFACE)
                    distributions[key] = canonical_race_probabilities(probs, RACE_LABELS_DEEPFACE)
                pairs.append((key, value))
                _record_model_latency(metrics, "race", key, started)
            return pairs, fuse_race(distributions)

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
                if points is not None:
                    value = predict_eye_color_colorimetric(net, face, points)
                else:
                    value = _cached_face_predict("eye_color", key, face, predict_eye_color_colorimetric, net, face)
                pairs.append((key, value))
                _record_model_latency(metrics, "eye_color", key, started)
            return pairs

        def _liveness_task():
            if not run_liveness:
                return [], None
            started = time.perf_counter()
            if track_id is not None:
                result = liveness_tracker.update(track_id, blink_score, texture_score)
            else:
                result = assess_static_liveness(texture_score)
            _record_model_latency(metrics, "liveness", "mediapipe", started)
            return [("mediapipe", result.summary)], result

        futures = {
            "age": _INFERENCE_EXECUTOR.submit(_age_task),
            "gender": _INFERENCE_EXECUTOR.submit(_gender_task),
            "emotion": _INFERENCE_EXECUTOR.submit(_emotion_task),
            "race": _INFERENCE_EXECUTOR.submit(_race_task),
            "gaze": _INFERENCE_EXECUTOR.submit(_gaze_task),
            "head_pose": _INFERENCE_EXECUTOR.submit(_head_pose_task),
            "recognition": _INFERENCE_EXECUTOR.submit(_recognition_task),
            "glasses": _INFERENCE_EXECUTOR.submit(_glasses_task),
            "mask": _INFERENCE_EXECUTOR.submit(_mask_task),
            "hair_color": _INFERENCE_EXECUTOR.submit(_hair_color_task),
            "eye_color": _INFERENCE_EXECUTOR.submit(_eye_color_task),
            "liveness": _INFERENCE_EXECUTOR.submit(_liveness_task),
        }

        age_pairs, best_age = futures["age"].result()
        gender_pairs, fused_gender = futures["gender"].result()
        emotion_pairs, fused_emotion = futures["emotion"].result()
        race_pairs, fused_race = futures["race"].result()
        gaze_pairs = futures["gaze"].result()
        head_pose_pairs = futures["head_pose"].result()
        recognition_pairs, face_embedding = futures["recognition"].result()
        glasses_pairs = futures["glasses"].result()
        mask_pairs = futures["mask"].result()
        hair_color_pairs = futures["hair_color"].result()
        eye_color_pairs = futures["eye_color"].result()
        liveness_pairs, liveness_result = futures["liveness"].result()

        if metrics is not None and emotion_pairs:
            metrics.setdefault("emotion_samples", []).extend(
                {"model": key, "emotion": value} for key, value in emotion_pairs
            )

        # Attribute text is intentionally NOT drawn on the shared image -- with several faces
        # close together, per-face text overlaps illegibly. The box + a small index number is
        # the only thing burned into pixels; full results are returned as structured data for
        # the caller to render as separate per-face UI (see src/app.py's target cards).
        box_thickness = int(round(frame.shape[0] / 150)) or 1
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness, 8)
        display_id = track_id if track_id is not None else idx
        draw_outlined_text(annotated_frame, str(display_id), (x1, max(20, y1 - 10)), (0, 255, 255))

        landmarks_net = models.face_landmarks_nets.get("mediapipe")
        if landmarks_net is not None and "mediapipe" in active_face_landmarks:
            landmark_points = predict_face_landmarks_mediapipe(landmarks_net, face, landmarker_result)
            if landmark_points is not None:
                draw_face_landmarks(annotated_frame, landmark_points, (x1, y1, x2, y2))

        eye_contact = [f"{key}=yes" if value.startswith("center/") else f"{key}=no" for key, value in gaze_pairs]

        # Every feature that fuses leads with its fused answer here, so the saved columns, the
        # per-face card, and the one-line summary all agree on which answer is the headline.
        # AGE names its most reliable model instead of blending (see AGE_MODEL_RELIABILITY),
        # so its headline row is labelled "best (<model>)" rather than "fused".
        age_pairs = with_headline(
            age_pairs,
            f"{best_age[0]} ({best_age[1]})" if best_age else None,
            BEST_MODEL_KEY,
        )
        gender_pairs = with_headline(gender_pairs, fused_gender)
        race_pairs = with_headline(race_pairs, fused_race)
        emotion_pairs = with_headline(emotion_pairs, fused_emotion)

        raw_columns = _gather_face_results({
            "age": age_pairs, "gender": gender_pairs, "race": race_pairs, "emotion": emotion_pairs,
            "gaze": gaze_pairs, "identity": recognition_pairs,
            "eye_contact": [("derived", value) for value in eye_contact], "head_pose": head_pose_pairs,
            "glasses": glasses_pairs, "mask": mask_pairs,
            "hair_color": hair_color_pairs, "eye_color": eye_color_pairs,
            "liveness": liveness_pairs,
        })
        model_results = [
            {"Feature": feature.replace("_", " ").upper(), "Model": model, "Output": str(value)}
            for feature, pairs in {
                "age": age_pairs,
                "gender": gender_pairs, "race": race_pairs, "emotion": emotion_pairs,
                "gaze": gaze_pairs, "identity": recognition_pairs,
                "eye contact": [("derived", value) for value in eye_contact],
                "head pose": head_pose_pairs,
                "glasses": glasses_pairs,
                "mask": mask_pairs, "hair color": hair_color_pairs, "eye color": eye_color_pairs,
                "liveness": liveness_pairs,
            }.items()
            for model, value in pairs
        ]

        cropped_faces.append({
            "idx": idx,
            "track_id": track_id,
            "box": (x1, y1, x2, y2),
            "image": cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
            "age": _format_results(age_pairs),
            "headline": {"age": best_age[0] if best_age else None, "gender": fused_gender,
                         "race": fused_race, "emotion": fused_emotion},
            "gender": _format_results(gender_pairs),
            "race": _format_results(race_pairs),
            "emotion": _format_results(emotion_pairs),
            "gaze": _format_results(gaze_pairs),
            "eye_contact": eye_contact,
            "head_pose": _format_results(head_pose_pairs),
            "identity": _format_results(recognition_pairs),
            "glasses": _format_results(glasses_pairs),
            "mask": _format_results(mask_pairs),
            "hair_color": _format_results(hair_color_pairs),
            "eye_color": _format_results(eye_color_pairs),
            "liveness": [liveness_result.summary] if liveness_result is not None else [],
            "liveness_status": liveness_result.status if liveness_result is not None else None,
            "blink_count": liveness_result.blink_count if liveness_result is not None else None,
            "blink_rate": liveness_result.blink_rate if liveness_result is not None else None,
            "texture_score": liveness_result.texture_score if liveness_result is not None else None,
            "texture_artifact": liveness_result.texture_artifact if liveness_result is not None else None,
            "embedding": face_embedding.tolist() if face_embedding is not None else None,
            "raw_columns": raw_columns,
            "model_results": model_results,
        })

    return annotated_frame, cropped_faces, bool(face_boxes), hands_detected


AGGREGATE_FEATURES = ("age", "gender", "race")  # demographic breakdown scope for crowd counting


def aggregate_demographics(cropped_faces: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    """Whole-image demographic aggregate over already-computed per-face results (age/gender/race
    only) -- no new model, just a tally over cropped_faces' raw_columns. Reuses whatever
    model(s) were already active per feature; if two models are active for the same feature
    (e.g. caffe + fairface age), each gets its own independent tally since their label sets/value
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
