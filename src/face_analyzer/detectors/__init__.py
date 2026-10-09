"""Face detector backends and factory dispatch."""
from .factory import (
    available_face_detectors,
    detect_faces,
    face_detector_net,
    resolve_face_detector,
)
from .retinaface import detect_faces_retinaface, detect_faces_retinaface_landmarks
from .scrfd import detect_faces_scrfd, detect_faces_scrfd_landmarks
from .ssd import detect_faces_ssd
from .yolo import detect_faces_yolo

__all__ = [
    "detect_faces_ssd",
    "detect_faces_yolo",
    "detect_faces_scrfd",
    "detect_faces_retinaface",
    "detect_faces_scrfd_landmarks",
    "detect_faces_retinaface_landmarks",
    "detect_faces",
    "available_face_detectors",
    "face_detector_net",
    "resolve_face_detector",
]
