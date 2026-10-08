"""Face detector backends and factory dispatch."""
from .base import BaseFaceDetector
from .factory import detect_faces
from .retinaface import detect_faces_retinaface
from .scrfd import detect_faces_scrfd
from .ssd import detect_faces_ssd
from .yolo import detect_faces_yolo

__all__ = [
    "BaseFaceDetector",
    "detect_faces_ssd",
    "detect_faces_yolo",
    "detect_faces_scrfd",
    "detect_faces_retinaface",
    "detect_faces",
]
