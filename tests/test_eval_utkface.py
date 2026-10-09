"""Pin the UTKFace eval's label parsing, stratification key and padding."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
_SPEC = importlib.util.spec_from_file_location("eval_utkface", _TOOLS / "eval_utkface.py")
eval_utkface = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_utkface)


def test_labels_come_from_well_formed_file_names_only(tmp_path):
    for name in ("26_1_3_20170117175041574.jpg.chip.jpg", "61_1_20170109142408075.jpg.chip.jpg",
                 "24_0_1_20170116220224657 .jpg.chip.jpg", "1_0_2_20161219140530307.jpg.chip.jpg"):
        (tmp_path / name).write_bytes(b"")
    labels = eval_utkface.parse_labels(tmp_path)
    assert [(item["age"], item["gender"], item["race"]) for item in labels] == [(1, 0, 2), (26, 1, 3)]


def test_decades_pool_ninety_and_over():
    assert [eval_utkface.decade(age) for age in (1, 9, 10, 89, 90, 116)] == [0, 0, 1, 8, 9, 9]


def test_padding_keeps_the_face_centred_on_black():
    face = np.full((200, 200, 3), 255, np.uint8)
    padded = eval_utkface.pad_face(face)
    assert padded.shape == (466, 466, 3)
    assert padded[133:333, 133:333].min() == 255 and padded[:133].max() == 0
