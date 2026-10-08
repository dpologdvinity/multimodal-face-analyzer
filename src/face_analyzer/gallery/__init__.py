"""Face database persistence, eigenfaces matching, and identity search subpackage."""
from __future__ import annotations

from .database import save_face
from .eigenfaces import (
    _crop_and_resize_for_eigenfaces,
    _load_eigen_images,
    _match_against_trained,
    _train_eigenfaces,
    match_face_eigenfaces,
    match_faces_eigenfaces_batch,
)
from .search import (
    _lbph_gallery_signature,
    _lbph_preprocess,
    build_gallery_from_directory,
    compute_face_embedding,
    decode_image_bytes,
    enroll_lbph_face,
    load_gallery,
    match_face_identity,
    predict_identity_lbph,
    save_gallery,
    train_lbph_recognizer,
    validate_lbph_name,
)

__all__ = [
    "save_face",
    "_crop_and_resize_for_eigenfaces",
    "_load_eigen_images",
    "_train_eigenfaces",
    "_match_against_trained",
    "match_face_eigenfaces",
    "match_faces_eigenfaces_batch",
    "compute_face_embedding",
    "match_face_identity",
    "load_gallery",
    "save_gallery",
    "_lbph_preprocess",
    "enroll_lbph_face",
    "validate_lbph_name",
    "_lbph_gallery_signature",
    "train_lbph_recognizer",
    "decode_image_bytes",
    "predict_identity_lbph",
    "build_gallery_from_directory",
]
