"""Cache every model's raw per-face output (probabilities where available) to tmp/predictions.json.

Fusion rules can then be designed and re-scored in milliseconds instead of re-running the
models for every idea. Run this again whenever a model's preprocessing changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import inference as inf  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def _face_crop(frame: np.ndarray, box, ratio: float = 0.1) -> np.ndarray:
    x1, y1, x2, y2 = inf.face_crop_bounds(box, frame.shape[:2], ratio)
    return frame[y1:y2, x1:x2]


def _fairface_alignment(models, frame: np.ndarray, box) -> np.ndarray | None:
    """Reproduce analyze_frame's MediaPipe landmarks, in frame coordinates.

    FairFace is markedly more accurate on a dlib-chip alignment than on the bbox
    approximation, so a fusion experiment that skipped this would under-rate it.
    """
    landmarker = models.face_landmarks_nets.get("mediapipe")
    if landmarker is None:
        return None
    x1, y1, x2, y2 = inf.face_crop_bounds(box, frame.shape[:2])
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    points = inf.predict_face_landmarks_mediapipe(landmarker, face)
    if points is None:
        return None
    local = inf.fairface_landmarks_from_mediapipe(points, face.shape[1], face.shape[0])
    return None if local is None else local + np.array([x1, y1], dtype=np.float32)


def mivolo_once(models, frame, box):
    """Run MiVOLO a single time per face; it is ~4s a call and answers age and gender together."""
    net = models.age_nets.get("mivolo") or models.gender_nets.get("mivolo")
    if net is None:
        return None
    return inf.mivolo_estimate(net, _face_crop(frame, box))


def age_predictions(models, frame, box, landmarks, mivolo) -> dict:
    """Continuous-year estimate per age backend, plus FairFace's full bucket distribution."""
    face = _face_crop(frame, box)
    out: dict = {}
    if "caffe" in models.age_nets:
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), inf.MODEL_MEAN_VALUES, swapRB=False)
        net = models.age_nets["caffe"]
        net.setInput(blob)
        out["caffe"] = net.forward().flatten().tolist()
    if "ssrnet" in models.age_nets and torch is not None:
        out["ssrnet"] = float(inf.predict_age_ssrnet(models.age_nets["ssrnet"], face))
    if "dex" in models.age_nets:
        net = models.age_nets["dex"]
        for name, crop in (("dex", inf.crop_face_dex(frame, box)),
                           ("dex_align", inf._margin_align(frame, box, 224, 1.3))):
            blob = cv2.dnn.blobFromImage(crop, 1.0, (224, 224), inf.DEX_MEAN_VALUES,
                                         swapRB=False, crop=False)
            net.setInput(blob)
            probs = net.forward().flatten().astype(np.float64)
            probs /= probs.sum()
            out[name] = float(probs @ np.arange(101))
            out[f"{name}_sd"] = float(np.sqrt(probs @ ((np.arange(101) - out[name]) ** 2)))
    if "fairface" in models.age_nets:
        logits = inf._fairface_forward(models.age_nets["fairface"], frame, box, "age_output", landmarks)
        out["fairface_probs"] = inf._softmax(logits).tolist()
    if "mivolo" in models.age_nets and mivolo is not None:
        out["mivolo"] = mivolo[0]
    return out


def gender_predictions(models, frame, box, landmarks, mivolo) -> dict:
    """Male-probability per gender backend (all are binary)."""
    face = _face_crop(frame, box)
    out: dict = {}
    if "caffe" in models.gender_nets:
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), inf.MODEL_MEAN_VALUES, swapRB=False)
        net = models.gender_nets["caffe"]
        net.setInput(blob)
        probs = net.forward().flatten()
        out["caffe"] = float(probs[0] / probs.sum())
    if "deepface" in models.gender_nets:
        resized = cv2.resize(face, (224, 224)).astype(np.float32)
        probs = models.gender_nets["deepface"](resized[np.newaxis, ...], training=False).numpy().flatten()
        out["deepface"] = float(probs[1] / probs.sum())
    if "fairface" in models.gender_nets:
        logits = inf._fairface_forward(models.gender_nets["fairface"], frame, box, "gender_output", landmarks)
        out["fairface"] = float(inf._softmax(logits)[0])
    if "mivolo" in models.gender_nets and mivolo is not None:
        out["mivolo"] = 1.0 if mivolo[1] == "Male" else 0.0
    return out


def race_predictions(models, frame, box, landmarks) -> dict:
    """Full class-probability vector per race backend (label sets differ; see inference.py)."""
    face = _face_crop(frame, box)
    out: dict = {}
    if "fairface" in models.race_nets:
        logits = inf._fairface_forward(models.race_nets["fairface"], frame, box, "race_output", landmarks)
        out["fairface"] = inf._softmax(logits).tolist()
    if "deepface" in models.race_nets:
        resized = cv2.resize(face, (224, 224)).astype(np.float32)
        out["deepface"] = models.race_nets["deepface"](
            resized[np.newaxis, ...], training=False).numpy().flatten().tolist()
    return out


def emotion_predictions(models, frame, box) -> dict:
    """Label per emotion backend, straight from the production predict functions."""
    face = _face_crop(frame, box)
    out: dict = {}
    for key, fn in (("dan", inf.predict_emotion_dan),
                    ("efficientnet", inf.predict_emotion_efficientnet),
                    ("mini_xception", inf.predict_emotion_mini_xception),
                    ("ferplus", inf.predict_emotion_ferplus),
                    ("hsemotion", inf.predict_emotion_hsemotion)):
        if key in models.emotion_nets:
            out[key] = fn(models.emotion_nets[key], face)
    return out


def main() -> None:
    models = inf.load_models()
    truth_data = json.loads((ROOT / "tools" / "ground_truth.json").read_text())
    records = []
    for name, faces in sorted(truth_data.items()):
        if name.startswith("_"):
            continue
        frame = cv2.imread(str(ROOT / "assets" / name))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        boxes = inf.detect_faces(models.face_net, frame, 0.5)
        for index, truth in enumerate(faces):
            tx, ty = truth["center"]
            best, best_distance = None, 0.05
            for box in boxes:
                x1, y1, x2, y2 = box
                distance = (((x1 + x2) / 2 / width - tx) ** 2
                            + ((y1 + y2) / 2 / height - ty) ** 2) ** 0.5
                if distance < best_distance:
                    best, best_distance = box, distance
            if best is None:
                print(f"!! no detection for {name}#{index}")
                continue
            landmarks = _fairface_alignment(models, frame, best)
            mivolo = mivolo_once(models, frame, best)
            records.append({
                "id": f"{name}#{index}",
                "truth": {k: v for k, v in truth.items() if k != "center"},
                "age": age_predictions(models, frame, best, landmarks, mivolo),
                "gender": gender_predictions(models, frame, best, landmarks, mivolo),
                "race": race_predictions(models, frame, best, landmarks),
                "emotion": emotion_predictions(models, frame, best),
            })
            print(f"  {records[-1]['id']}")
    out_path = ROOT / "tmp" / "predictions.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} faces to {out_path}")


if __name__ == "__main__":
    main()
