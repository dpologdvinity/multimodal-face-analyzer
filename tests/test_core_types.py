import numpy as np
import pytest
from src.core.types import BoundingBox, Detection, FaceResult, Models
from src.core.constants import (
    AGE_LIST,
    GENDER_LIST,
    EMOTION_LABELS_DAN,
    EMOTION_LABELS_FERPLUS,
    EMOTION_LABELS_HSEMOTION,
    EMOTION_LABELS_MINI_XCEPTION,
    RACE_CANONICAL_LABELS,
    HAIR_COLOR_LABELS,
    EYE_COLOR_LABELS,
)
from src.core.image_utils import (
    apply_image_adjustments,
    _is_skin_hsv,
    face_crop_bounds,
    crop_region,
)
import src.inference as inference


def test_bounding_box_init_and_properties():
    box = BoundingBox(10, 20, 110, 220)
    assert box.x1 == 10
    assert box.y1 == 20
    assert box.x2 == 110
    assert box.y2 == 220
    assert box.width == 100
    assert box.height == 200
    assert box.area == 20000
    assert box.to_tuple() == (10, 20, 110, 220)
    assert box.to_list() == [10, 20, 110, 220]
    # Test unpacking and indexing
    x1, y1, x2, y2 = box
    assert (x1, y1, x2, y2) == (10, 20, 110, 220)
    assert box[0] == 10
    assert box[2] == 110
    assert len(box) == 4


def test_detection_init():
    box = BoundingBox(10, 20, 50, 60)
    det = Detection(box=box, confidence=0.92, landmarks=[(20, 30), (40, 30)], class_id=0, label="face")
    assert det.box == box
    assert det.confidence == 0.92
    assert det.landmarks == [(20, 30), (40, 30)]
    assert det.label == "face"

    # Test auto-conversion from tuple
    det_from_tuple = Detection(box=(10, 20, 50, 60), confidence=0.85)
    assert isinstance(det_from_tuple.box, BoundingBox)
    assert det_from_tuple.box.width == 40
    assert det_from_tuple.box.height == 40


def test_face_result_init():
    res = FaceResult(
        idx=1,
        box=BoundingBox(10, 20, 50, 60),
        headline={"age": "25-32", "gender": "Female", "race": "Asian", "emotion": "happy"},
    )
    assert res.idx == 1
    assert res.headline["gender"] == "Female"
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["idx"] == 1
    assert d["headline"]["race"] == "Asian"


def test_models_dataclass():
    models = Models()
    assert models.face_net is None
    assert models.loaded_feature_count == 0
    assert models.total_feature_count > 0
    assert len(models.offline_features) == models.total_feature_count

    dummy_net = object()
    models_with_net = Models(age_nets={"caffe": dummy_net})
    assert models_with_net.loaded_feature_count == 1
    assert "AGE" not in models_with_net.offline_features


def test_constants_integrity():
    assert len(AGE_LIST) > 0
    assert len(GENDER_LIST) == 2
    assert "neutral" in EMOTION_LABELS_DAN
    assert "neutral" in EMOTION_LABELS_FERPLUS
    assert "neutral" in EMOTION_LABELS_HSEMOTION
    assert "neutral" in EMOTION_LABELS_MINI_XCEPTION
    assert "white" in RACE_CANONICAL_LABELS
    assert "black" in HAIR_COLOR_LABELS
    assert "brown" in EYE_COLOR_LABELS

    # Re-export check in src.inference
    assert inference.AGE_LIST == AGE_LIST
    assert inference.GENDER_LIST == GENDER_LIST
    assert inference.RACE_CANONICAL_LABELS == RACE_CANONICAL_LABELS
    assert inference.BoundingBox is BoundingBox
    assert inference.Detection is Detection
    assert inference.FaceResult is FaceResult
    assert inference.Models is Models


def test_apply_image_adjustments():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    # No adjustments -> identical output
    adjusted = apply_image_adjustments(img, {})
    assert np.array_equal(img, adjusted)

    # Brightness adjustment
    brightened = apply_image_adjustments(img, {"brightness": 20})
    assert brightened.shape == img.shape
    assert brightened.dtype == np.uint8
    assert np.all(brightened > img)


def test_is_skin_hsv():
    # HSV pixel: H=15, S=80, V=100 is typically skin
    skin_pixel = np.array([[[15, 80, 100]]], dtype=np.uint8)
    assert _is_skin_hsv(skin_pixel)[0, 0]

    # Blue pixel: H=120, S=200, V=200 is not skin
    blue_pixel = np.array([[[120, 200, 200]]], dtype=np.uint8)
    assert not _is_skin_hsv(blue_pixel)[0, 0]


def test_crop_helpers():
    box = (20, 30, 80, 90)
    frame_shape = (200, 200)
    padded = face_crop_bounds(box, frame_shape, padding_ratio=0.1)
    assert padded[0] <= box[0]
    assert padded[1] <= box[1]
    assert padded[2] >= box[2]
    assert padded[3] >= box[3]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_region(frame, 10, 20, 50, 60)
    assert crop.shape == (40, 40, 3)
