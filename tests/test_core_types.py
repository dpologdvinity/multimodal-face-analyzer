import numpy as np

import face_analyzer.inference as inference
from face_analyzer.core.constants import (
    AGE_LIST,
    EMOTION_LABELS_DAN,
    EMOTION_LABELS_FERPLUS,
    EMOTION_LABELS_HSEMOTION,
    EMOTION_LABELS_MINI_XCEPTION,
    GENDER_LIST,
    RACE_CANONICAL_LABELS,
)
from face_analyzer.core.image_utils import (
    _is_skin_hsv,
    apply_image_adjustments,
    crop_region,
    face_crop_bounds,
)
from face_analyzer.core.types import Models


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

    # Re-export check in face_analyzer.inference
    assert inference.AGE_LIST == AGE_LIST
    assert inference.GENDER_LIST == GENDER_LIST
    assert inference.RACE_CANONICAL_LABELS == RACE_CANONICAL_LABELS
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


def test_base_dir_is_repo_root():
    """Resolve BASE_DIR to the repository root from inside the installed package."""
    from face_analyzer.core.constants import BASE_DIR

    assert (BASE_DIR / "models").is_dir() and (BASE_DIR / "pyproject.toml").is_file()

