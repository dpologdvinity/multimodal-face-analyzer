"""Sweep age-model preprocessing variants over the labelled benchmark faces.

Detects every labelled face once, then re-runs only the candidate preprocessing so a variant
costs seconds instead of a full analyze_frame() pass. Reports accuracy (prediction overlaps the
truth range) and MAE against each truth range's midpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import inference as inf  # noqa: E402


def labelled_faces(models) -> list[dict]:
    """Detect each ground-truth face once and return frame/box/truth tuples."""
    truth_data = json.loads((ROOT / "tools" / "ground_truth.json").read_text())
    out = []
    for name, faces in sorted(truth_data.items()):
        if name.startswith("_"):
            continue
        frame = cv2.imread(str(ROOT / "assets" / name))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        boxes = inf.detect_faces(models.face_net, frame, 0.5)
        for index, truth in enumerate(faces):
            if "age" not in truth:
                continue
            tx, ty = truth["center"]
            best, best_distance = None, 0.05
            for box in boxes:
                x1, y1, x2, y2 = box
                distance = (((x1 + x2) / 2 / width - tx) ** 2 + ((y1 + y2) / 2 / height - ty) ** 2) ** 0.5
                if distance < best_distance:
                    best, best_distance = box, distance
            if best is not None:
                out.append({"name": f"{name}#{index}", "frame": frame, "box": best, "age": truth["age"]})
    return out


def report(title: str, predictions: list[tuple[str, float, list[int]]]) -> None:
    """Print accuracy (prediction inside the truth range) and MAE vs the range midpoint."""
    hits = [low <= value <= high for _, value, (low, high) in predictions]
    errors = [abs(value - (low + high) / 2) for _, value, (low, high) in predictions]
    print(f"{title:<40} {sum(hits):>3}/{len(hits):<3} = {100 * sum(hits) / len(hits):5.1f}%  "
          f"MAE {np.mean(errors):5.1f}y")


def ssrnet_variants(net, samples: list[dict]) -> None:
    """Compare SSR-Net crops and input normalizations."""
    def run(face_bgr, normalize: str, to_rgb: bool):
        image = cv2.resize(face_bgr, (64, 64))
        if to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = image.astype(np.float32) / 255.0
        if normalize == "imagenet":
            array = (array - inf.SSRNET_MEAN) / inf.SSRNET_STD
        elif normalize == "centered":
            array = array - 0.5
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            return float(net(tensor).item())

    for crop_name, crop in CROPS.items():
        for normalize in ("imagenet", "plain", "centered"):
            for to_rgb in (True, False):
                predictions = [
                    (s["name"], run(crop(s), normalize, to_rgb), s["age"]) for s in samples
                ]
                report(f"ssrnet {crop_name}/{normalize}/{'rgb' if to_rgb else 'bgr'}", predictions)


def dex_variants(net, samples: list[dict]) -> None:
    """Compare DEX crops and decoding (expected value vs argmax)."""
    def run(face_bgr, decode: str):
        blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (224, 224), inf.DEX_MEAN_VALUES,
                                     swapRB=False, crop=False)
        net.setInput(blob)
        probs = net.forward().flatten().astype(np.float64)
        probs = probs / probs.sum()
        return float(probs.argmax()) if decode == "argmax" else float(probs @ np.arange(101))

    for crop_name, crop in CROPS.items():
        for decode in ("expected", "argmax"):
            predictions = [(s["name"], run(crop(s), decode), s["age"]) for s in samples]
            report(f"dex {crop_name}/{decode}", predictions)


CROPS = {
    "dexmargin": lambda s: inf.crop_face_dex(s["frame"], s["box"]),
    "pad10": lambda s: _bounds_crop(s, 0.1),
    "pad25": lambda s: _bounds_crop(s, 0.25),
    "tight": lambda s: _bounds_crop(s, 0.0),
    "align1.3": lambda s: inf._margin_align(s["frame"], s["box"], 224, 1.3),
    "align1.5": lambda s: inf._margin_align(s["frame"], s["box"], 224, 1.5),
}


def _bounds_crop(sample: dict, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = inf.face_crop_bounds(sample["box"], sample["frame"].shape[:2], ratio)
    return sample["frame"][y1:y2, x1:x2]


def main() -> None:
    models = inf.load_models()
    samples = labelled_faces(models)
    print(f"{len(samples)} labelled faces\n")
    if "ssrnet" in models.age_nets:
        ssrnet_variants(models.age_nets["ssrnet"], samples)
    print()
    if "dex" in models.age_nets:
        dex_variants(models.age_nets["dex"], samples)


if __name__ == "__main__":
    main()
