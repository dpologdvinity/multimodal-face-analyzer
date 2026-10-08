"""Held-out evaluation of the age, gender and race backends on the FairFace validation split.

Two phases, both run by default:

1. **Inference.** Draws a race-stratified sample (fixed seed), runs the real analyze_frame()
   on each image with every loaded age/gender/race/emotion backend and the SSD detector, and
   appends each image's raw per-backend outputs (probabilities where the backend exposes them)
   to data/eval_cache/predictions.jsonl. Already-cached rows are skipped, so the run resumes.
2. **Scoring.** Re-scores the cached outputs only (no model imports): per-backend accuracy,
   the app's combined answers via the shipped fuse_gender/fuse_race/select_age, fusion
   weights fitted on the fit half, and 95% bootstrap CIs on the test half.

    python tools/fetch_fairface.py
    FACE_ANALYZER_MODEL_DIR=/path/to/models python tools/eval_heldout.py [--n 2000] [--score-only]

Method, label mappings and caveats: docs/eval/heldout_fairface.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np

from face_analyzer.core.constants import (
    AGE_LIST_RANGES,
    FAIRFACE_AGE_RANGES,
    GENDER_FUSION_WEIGHTS,
    MODEL_DIR,
    RACE_CANONICAL_LABELS,
    RACE_FUSION_WEIGHTS,
    RACE_LABEL_TO_CANONICAL,
    RACE_LABELS_DEEPFACE,
    RACE_LABELS_FAIRFACE,
)
from face_analyzer.fusion import (
    canonical_race_probabilities,
    fuse_gender,
    fuse_race,
    select_age,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fairface"
CACHE_DIR = ROOT / "data" / "eval_cache"
OUT_JSON = ROOT / "docs" / "eval" / "heldout_fairface.json"

# Label order of the HuggingFaceM4/FairFace parquet (differs from the model's output order).
DATASET_RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern",
                 "Latino_Hispanic", "Southeast Asian"]
DATASET_GENDERS = ["Male", "Female"]

AGE_BACKENDS = ("mivolo", "fairface", "dex", "caffe")
GENDER_BACKENDS = ("mivolo", "fairface", "caffe", "deepface")
RACE_BACKENDS = ("fairface", "deepface")
CANONICAL_RACES = tuple(RACE_CANONICAL_LABELS)
# fuse_race displays canonical names ("Latino"), which are not all backend label spellings.
CANONICAL_BY_DISPLAY = {label: key for key, label in RACE_CANONICAL_LABELS.items()}

# Features the eval does not need; an empty selection skips loading them (see model_selection).
UNUSED_FEATURE_ENV = (
    "RECOGNITION_MODEL", "GLASSES_MODEL", "MASK_MODEL", "COLORIZATION_MODEL", "HAND_MODEL",
    "RECONSTRUCTION_3D_MODEL", "AGE_PROGRESSION_MODEL", "YOLO_FACE_MODEL", "SCRFD_FACE_MODEL",
    "RETINAFACE_MODEL", "LIVENESS_MODEL",
)


# --------------------------------------------------------------------------- sampling


def stratified_sample(strata: Sequence, n: int, seed: int) -> list[int]:
    """Draw n indices with per-stratum counts proportional to stratum size (largest remainder).

    Each stratum is permuted once, independent of n, and the sample is a prefix of each
    permutation, so a smaller sample is (up to rounding) a subset of a larger one.
    """
    groups: dict = {}
    for index, stratum in enumerate(strata):
        groups.setdefault(stratum, []).append(index)
    n = min(n, len(strata))
    keys = sorted(groups)
    quotas = {key: n * len(groups[key]) / len(strata) for key in keys}
    counts = {key: math.floor(quota) for key, quota in quotas.items()}
    by_remainder = sorted(keys, key=lambda key: (counts[key] - quotas[key], key))
    for key in by_remainder[: n - sum(counts.values())]:
        counts[key] += 1
    rng = np.random.default_rng([seed, 0])
    chosen: list[int] = []
    for key in keys:
        chosen.extend(int(i) for i in rng.permutation(groups[key])[: counts[key]])
    return sorted(chosen)


def stratified_split(indices: Sequence[int], strata: Sequence, seed: int) -> tuple[list[int], list[int]]:
    """Split indices 50/50 into (fit, test) within each stratum; an odd one out goes to test."""
    groups: dict = {}
    for index in indices:
        groups.setdefault(strata[index], []).append(index)
    rng = np.random.default_rng([seed, 1])
    fit: list[int] = []
    test: list[int] = []
    for key in sorted(groups):
        members = sorted(groups[key])
        rng.shuffle(members)
        half = len(members) // 2
        fit.extend(members[:half])
        test.extend(members[half:])
    return sorted(fit), sorted(test)


# --------------------------------------------------------------------------- label mappings


def age_to_bucket(age: float) -> int:
    """Map an age in years onto a FairFace bucket index, via the integer the app displays."""
    shown = max(0, int(f"{age:.0f}"))
    for index, (low, high) in enumerate(FAIRFACE_AGE_RANGES):
        if low <= shown <= high:
            return index
    return len(FAIRFACE_AGE_RANGES) - 1


def caffe_to_fairface_bucket(caffe_index: int) -> int:
    """Map a Caffe age bucket onto FairFace's buckets via its midpoint, as select_age reads it."""
    return age_to_bucket(float(np.mean(AGE_LIST_RANGES[caffe_index])))


def canonical_race(label: str) -> str | None:
    """Map any backend or dataset race label onto the app's shared canonical race key."""
    return RACE_LABEL_TO_CANONICAL.get(label.strip().lower())


def largest_in_frame(boxes: Sequence[Sequence[int]], height: int, width: int) -> list[int] | None:
    """Pick the box with the largest area inside the frame; None when no box overlaps it.

    Ranking by clipped area matters because SSD can return boxes that lie mostly or wholly
    outside the image, and those carry no face pixels to classify.
    """
    best, best_area = None, 0
    for box in boxes:
        x1, y1, x2, y2 = box
        area = max(0, min(x2, width) - max(x1, 0)) * max(0, min(y2, height) - max(y1, 0))
        if area > best_area:
            best, best_area = [int(v) for v in box], area
    return best


def shown_top1(label: str) -> str:
    """Take the first class of a "White (52%)/Black (47%)" style label: strict top-1 scoring."""
    return label.split("/")[0].split("(")[0].strip()


# --------------------------------------------------------------------------- per-backend answers
# Each mirrors pipeline/face_tasks.py, applied to the probabilities recorded during the run.


def gender_male_probabilities(raw: dict) -> dict[str, float]:
    """Rebuild the per-backend P(Male) that _gender_task feeds to fuse_gender."""
    out: dict[str, float] = {}
    if "gender_probs/caffe" in raw:
        probs = np.asarray(raw["gender_probs/caffe"], dtype=float)
        out["caffe"] = float(probs[0] / (probs.sum() or 1.0))
    if "gender_probs/deepface" in raw:
        probs = np.asarray(raw["gender_probs/deepface"], dtype=float)
        out["deepface"] = float(probs[1] / (probs.sum() or 1.0))
    if "gender_probs/fairface" in raw:
        out["fairface"] = float(raw["gender_probs/fairface"][0])
    if raw.get("mivolo/face") is not None:
        out["mivolo"] = 1.0 if raw["mivolo/face"][1] == "Male" else 0.0
    return out


def gender_labels(raw: dict) -> dict[str, str]:
    """Each backend's displayed gender label (argmax, as _gender_task shows it)."""
    out: dict[str, str] = {}
    if "gender_probs/caffe" in raw:
        out["caffe"] = "Male" if int(np.argmax(raw["gender_probs/caffe"])) == 0 else "Female"
    if "gender_probs/deepface" in raw:
        out["deepface"] = "Male" if int(np.argmax(raw["gender_probs/deepface"])) == 1 else "Female"
    if "gender_probs/fairface" in raw:
        out["fairface"] = "Male" if int(np.argmax(raw["gender_probs/fairface"])) == 0 else "Female"
    if raw.get("mivolo/face") is not None:
        out["mivolo"] = raw["mivolo/face"][1]
    return out


def race_distributions(raw: dict) -> dict[str, dict[str, float]]:
    """Rebuild the per-backend canonical race distributions that _race_task feeds to fuse_race."""
    out = {}
    if "race_probs/fairface" in raw:
        out["fairface"] = canonical_race_probabilities(
            np.asarray(raw["race_probs/fairface"]), RACE_LABELS_FAIRFACE)
    if "race_probs/deepface" in raw:
        out["deepface"] = canonical_race_probabilities(
            np.asarray(raw["race_probs/deepface"]), RACE_LABELS_DEEPFACE)
    return out


def race_labels(raw: dict) -> dict[str, str | None]:
    """Each backend's top-1 displayed class (FairFace's 7 or DeepFace's 6), mapped to canonical."""
    out = {}
    for key, labels in (("fairface", RACE_LABELS_FAIRFACE), ("deepface", RACE_LABELS_DEEPFACE)):
        probs = raw.get(f"race_probs/{key}")
        if probs is not None:
            out[key] = canonical_race(labels[int(np.argmax(probs))])
    return out


def age_estimates(raw: dict) -> dict[str, float]:
    """Rebuild the per-backend year estimates that _age_task feeds to select_age."""
    out: dict[str, float] = {}
    if "age_probs/caffe" in raw:
        out["caffe"] = float(np.mean(AGE_LIST_RANGES[int(np.argmax(raw["age_probs/caffe"]))]))
    if "age_probs/fairface" in raw:
        out["fairface"] = float(np.mean(FAIRFACE_AGE_RANGES[int(np.argmax(raw["age_probs/fairface"]))]))
    if raw.get("age_estimate/dex") is not None:
        out["dex"] = float(raw["age_estimate/dex"][0])
    if raw.get("mivolo/face") is not None:
        out["mivolo"] = float(raw["mivolo/face"][0])
    return out


def age_buckets(raw: dict) -> dict[str, int]:
    """Each backend's answer as a FairFace bucket index (see the mapping in the eval doc)."""
    out: dict[str, int] = {}
    if "age_probs/caffe" in raw:
        out["caffe"] = caffe_to_fairface_bucket(int(np.argmax(raw["age_probs/caffe"])))
    if "age_probs/fairface" in raw:
        out["fairface"] = int(np.argmax(raw["age_probs/fairface"]))
    if raw.get("age_estimate/dex") is not None:
        out["dex"] = age_to_bucket(raw["age_estimate/dex"][0])
    if raw.get("mivolo/face") is not None:
        out["mivolo"] = age_to_bucket(raw["mivolo/face"][0])
    return out


def best_age_bucket(raw: dict) -> tuple[int, str] | None:
    """The app's headline age (select_age) as (FairFace bucket, chosen backend)."""
    chosen = select_age(age_estimates(raw))
    return None if chosen is None else (age_to_bucket(float(chosen[0])), chosen[1])


def fused_gender(raw: dict, weights: dict[str, float] | None = None) -> str | None:
    """Run the shipped fuse_gender, optionally under substitute weights."""
    if weights is None:
        return fuse_gender(gender_male_probabilities(raw))
    with mock.patch.dict(GENDER_FUSION_WEIGHTS, weights, clear=True):
        return fuse_gender(gender_male_probabilities(raw))


def fused_race(raw: dict, weights: dict[str, float] | None = None) -> str | None:
    """Run the shipped fuse_race (strict top-1, canonical key), optionally under substitute weights."""
    if weights is None:
        label = fuse_race(race_distributions(raw))
    else:
        with mock.patch.dict(RACE_FUSION_WEIGHTS, weights, clear=True):
            label = fuse_race(race_distributions(raw))
    return None if label is None else CANONICAL_BY_DISPLAY.get(shown_top1(label))


# --------------------------------------------------------------------------- statistics


def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of values (seeded, so reruns are identical)."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(n_resamples, data.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def summarize(values: Sequence[float], seed: int, n_resamples: int = 1000) -> dict:
    """Mean, 95% bootstrap CI and count, rounded for the committed JSON."""
    data = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(data, n_resamples=n_resamples, seed=seed)
    mean = float(data.mean()) if data.size else float("nan")
    return {"value": round(mean, 4), "ci95": [round(low, 4), round(high, 4)], "n": int(data.size)}


def log_odds_weights(accuracies: dict[str, float], n_classes: int) -> dict[str, float]:
    """Weight each backend by log((K-1) * acc / (1 - acc)), clipped at 0.

    This is the optimal weighted-vote weight for independent K-class voters with symmetric
    errors (Nitzan & Paroush, 1982); for K=2 it is plain log-odds. A backend at or below
    chance gets weight 0.
    """
    out = {}
    for key, accuracy in accuracies.items():
        accuracy = min(max(accuracy, 1e-3), 1 - 1e-3)
        out[key] = round(max(0.0, math.log((n_classes - 1) * accuracy / (1 - accuracy))), 3)
    if not any(out.values()):
        out = dict.fromkeys(out, 1.0)
    return out


# --------------------------------------------------------------------------- inference


def load_labels(parquet: Path) -> dict[str, list]:
    """Read the label columns (and image paths) of the FairFace parquet, without images."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["age", "gender", "race"])
    paths = pq.read_table(parquet, columns=["image"]).column("image").combine_chunks()
    return {
        "age": table.column("age").to_pylist(),
        "gender": table.column("gender").to_pylist(),
        "race": table.column("race").to_pylist(),
        "path": paths.field("path").to_pylist(),
    }


def _jsonable(value):
    """Convert numpy containers and scalars into plain JSON types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _instrument(sink: dict) -> None:
    """Record every per-face model output analyze_frame computes, without changing any of them.

    analyze_frame returns display strings only. Wrapping the memoization helper the per-face
    tasks call captures the raw probabilities behind them, so the combined answers can be
    re-scored (and re-weighted) from the cache. The detector wrapper keeps only the largest
    in-frame box: FairFace images are face crops, and the eval scores the largest detection.
    """
    from face_analyzer.pipeline import analyzer, face_tasks, stages

    original_predict = stages._cached_face_predict
    original_detect = stages._detect
    original_landmarks = stages.fairface_landmarks_from_mediapipe

    def recording_predict(feature, model_key, face_bgr, predict_fn, *args):
        started = time.perf_counter()
        value = original_predict(feature, model_key, face_bgr, predict_fn, *args)
        key = f"{feature}/{model_key}"
        sink["raw"][key] = _jsonable(value)
        sink["latency_ms"][key] = round((time.perf_counter() - started) * 1000, 1)
        return value

    def largest_detection(models, frame, config):
        boxes = original_detect(models, frame, config)
        sink["detections"] = [[int(v) for v in box] for box in boxes]
        chosen = largest_in_frame(boxes, *frame.shape[:2])
        return [] if chosen is None else [chosen]

    def recording_landmarks(*args, **kwargs):
        result = original_landmarks(*args, **kwargs)
        sink["fairface_aligned"] = result is not None
        return result

    stages._cached_face_predict = recording_predict
    face_tasks._cached_face_predict = recording_predict
    analyzer._detect = largest_detection
    stages.fairface_landmarks_from_mediapipe = recording_landmarks


def _versions() -> dict[str, str | None]:
    """Installed versions of the packages that determine the numbers."""
    from importlib import metadata

    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "opencv-python-headless", "torch", "tensorflow-cpu", "tf-keras",
                 "onnxruntime", "mediapipe", "ultralytics", "timm"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["git_commit"] = None
    return out


def run_inference(rows: list[int], labels: dict, parquet: Path, cache: Path,
                  exclude: set[str], conf: float, seed: int) -> None:
    """Run analyze_frame on every sampled row not already cached, appending to the cache."""
    for name in UNUSED_FEATURE_ENV:
        os.environ[name] = ""
    import cv2
    import pyarrow.parquet as pq

    cv2.setNumThreads(2)
    try:
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    from face_analyzer import inference

    cache.mkdir(parents=True, exist_ok=True)
    predictions = cache / "predictions.jsonl"
    done = {record["row"] for record in read_cache(predictions)}
    # Half-size sample first, so an interrupted run still leaves a complete stratified subset.
    first = set(stratified_sample(labels["race"], len(rows) // 2, seed))
    todo = sorted((row for row in rows if row not in done), key=lambda row: row not in first)
    print(f"{len(rows) - len(todo)} of {len(rows)} sampled images already cached", flush=True)
    if not todo:
        return

    models = inference.load_models()
    active = {feature: {key for key in nets if f"{feature}:{key}" not in exclude}
              for feature, nets in (("age", models.age_nets), ("gender", models.gender_nets),
                                    ("race", models.race_nets), ("emotion", models.emotion_nets))}
    config = inference.AnalysisConfig(
        conf_threshold=conf, face_detector="ssd", active_age=active["age"],
        active_gender=active["gender"], active_race=active["race"],
        active_emotion=active["emotion"],
    )
    meta = {"backends": {feature: sorted(keys) for feature, keys in active.items()},
            "excluded": sorted(exclude), "detector": "ssd", "conf_threshold": conf,
            "model_dir": str(MODEL_DIR),
            "versions": _versions(), "date": date.today().isoformat()}
    (cache / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"backends: {meta['backends']}", flush=True)

    sink: dict = {}
    _instrument(sink)
    images = pq.read_table(parquet, columns=["image"]).column("image").combine_chunks().field("bytes")
    started_all = time.perf_counter()
    with predictions.open("a") as handle:
        for count, row in enumerate(todo, 1):
            sink.clear()
            sink.update(raw={}, latency_ms={}, detections=[], fairface_aligned=None)
            frame = cv2.imdecode(np.frombuffer(images[row].as_py(), np.uint8), cv2.IMREAD_COLOR)
            started = time.perf_counter()
            error = None
            faces = []
            try:
                _, faces, _, _ = inference.analyze_frame(models, frame, config)
            except Exception as exc:  # recorded and counted, so a crash never drops a row silently
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "row": row, "path": labels["path"][row],
                "truth": {feature: labels[feature][row] for feature in ("age", "gender", "race")},
                "size": list(frame.shape[:2]), "detections": sink["detections"],
                "box": [int(v) for v in faces[0]["box"]] if faces else None,
                "raw": {key: value for key, value in sink["raw"].items()
                        if not key.startswith("face_detection/")},
                "outputs": {row_["Feature"].lower() + "/" + row_["Model"]: row_["Output"]
                            for row_ in (faces[0]["model_results"] if faces else [])
                            if row_["Feature"] in ("AGE", "GENDER", "RACE", "EMOTION")},
                "fairface_aligned": sink["fairface_aligned"],
                "latency_ms": sink["latency_ms"],
                "seconds": round(time.perf_counter() - started, 3), "error": error,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if count % 25 == 0 or count == len(todo):
                elapsed = time.perf_counter() - started_all
                eta = elapsed / count * (len(todo) - count)
                print(f"{count}/{len(todo)} images, {elapsed / count:.2f} s/image, "
                      f"ETA {eta / 60:.0f} min", flush=True)


def read_cache(path: Path) -> list[dict]:
    """Load cached per-image records (empty when the cache does not exist yet)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- scoring


def _truth(record: dict) -> dict:
    """Ground truth as (gender label, canonical race, FairFace 7-class race, age bucket)."""
    truth = record["truth"]
    race = DATASET_RACES[truth["race"]]
    return {"gender": DATASET_GENDERS[truth["gender"]], "race": canonical_race(race),
            "race7": race, "age": truth["age"]}


def _detected(record: dict) -> bool:
    """Whether the detector found a face and the pipeline produced results for it."""
    return record["box"] is not None and record["error"] is None


def score(records: list[dict], fit_rows: list[int], test_rows: list[int], seed: int,
          n_resamples: int = 1000) -> dict:
    """Score the cached outputs: per-backend and combined metrics on the test half."""
    by_row = {record["row"]: record for record in records}
    fit = [by_row[row] for row in fit_rows]
    test = [by_row[row] for row in test_rows]

    def summary(values):
        return summarize(values, seed, n_resamples)

    result: dict = {
        "detection_recall": {
            "test": summary([_detected(r) for r in test]),
            "all": summary([_detected(r) for r in fit + test]),
            "errors": sum(r["error"] is not None for r in fit + test),
        },
    }
    test_faces = [r for r in test if _detected(r)]
    fit_faces = [r for r in fit if _detected(r)]

    # Gender: every backend's displayed label, the shipped fusion, fitted fusion.
    gender: dict = {"backends": {}}
    for key in GENDER_BACKENDS:
        if any(key in gender_labels(r["raw"]) for r in test_faces):
            gender["backends"][key] = summary(
                [gender_labels(r["raw"]).get(key) == _truth(r)["gender"] for r in test_faces])
    shipped_gender = [fused_gender(r["raw"]) == _truth(r)["gender"] for r in test_faces]
    gender["fused_shipped"] = summary(shipped_gender)

    # Race: canonical top-1 per backend (+ FairFace's native 7 classes), shipped/fitted fusion.
    race: dict = {"backends": {}, "per_class_recall": {}}
    for key in RACE_BACKENDS:
        if any(key in race_labels(r["raw"]) for r in test_faces):
            race["backends"][key] = summary(
                [race_labels(r["raw"]).get(key) == _truth(r)["race"] for r in test_faces])
    if any("race_probs/fairface" in r["raw"] for r in test_faces):
        race["fairface_7class"] = summary([
            "race_probs/fairface" in r["raw"]
            and RACE_LABELS_FAIRFACE[int(np.argmax(r["raw"]["race_probs/fairface"]))]
            == _truth(r)["race7"] for r in test_faces])
    shipped_race = [fused_race(r["raw"]) == _truth(r)["race"] for r in test_faces]
    race["fused_shipped"] = summary(shipped_race)
    for canonical in CANONICAL_RACES:
        members = [r for r in test_faces if _truth(r)["race"] == canonical]
        if not members:
            continue
        row: dict[str, float] = {"n": len(members)}
        for key in race["backends"]:
            row[key] = round(float(np.mean(
                [race_labels(r["raw"]).get(key) == canonical for r in members])), 4)
        row["fused_shipped"] = round(float(np.mean(
            [fused_race(r["raw"]) == canonical for r in members])), 4)
        race["per_class_recall"][canonical] = row

    # Age: bucket accuracy and mean absolute bucket offset per backend and for the headline.
    age: dict = {"backends": {}}
    for key in AGE_BACKENDS:
        answered = [r for r in test_faces if key in age_buckets(r["raw"])]
        if not answered:
            continue
        offsets = [abs(age_buckets(r["raw"])[key] - _truth(r)["age"]) for r in answered]
        age["backends"][key] = {
            "bucket_accuracy": summary(
                [age_buckets(r["raw"]).get(key) == _truth(r)["age"] for r in test_faces]),
            "mean_bucket_offset": summary(offsets),
            "within_one_bucket": summary([offset <= 1 for offset in offsets]),
        }
    best = [(best_age_bucket(r["raw"]), _truth(r)["age"]) for r in test_faces]
    best_offsets = [abs(chosen[0] - truth) for chosen, truth in best if chosen is not None]
    age["best_shipped"] = {
        "bucket_accuracy": summary([chosen is not None and chosen[0] == truth for chosen, truth in best]),
        "mean_bucket_offset": summary(best_offsets),
        "within_one_bucket": summary([offset <= 1 for offset in best_offsets]),
        "chosen_backend": {key: sum(1 for chosen, _ in best if chosen and chosen[1] == key)
                           for key in AGE_BACKENDS},
    }
    if any(r["raw"].get("age_estimate/dex") for r in test_faces):
        dex = [r["raw"]["age_estimate/dex"] for r in test_faces if r["raw"].get("age_estimate/dex")]
        age["dex_uncertain_share"] = round(float(np.mean([sd > 10.0 for _, sd in dex])), 4)

    # Fusion weights fitted on the fit half (log-odds of fit-half accuracy), scored on test.
    fit_gender_acc = {key: float(np.mean([gender_labels(r["raw"]).get(key) == _truth(r)["gender"]
                                          for r in fit_faces])) for key in gender["backends"]}
    fit_race_acc = {key: float(np.mean([race_labels(r["raw"]).get(key) == _truth(r)["race"]
                                        for r in fit_faces])) for key in race["backends"]}
    fitted_gender_weights = log_odds_weights(fit_gender_acc, 2)
    fitted_race_weights = log_odds_weights(fit_race_acc, len(CANONICAL_RACES))
    fitted_gender = [fused_gender(r["raw"], fitted_gender_weights) == _truth(r)["gender"]
                     for r in test_faces]
    fitted_race = [fused_race(r["raw"], fitted_race_weights) == _truth(r)["race"]
                   for r in test_faces]
    gender["fused_fitted"] = summary(fitted_gender)
    race["fused_fitted"] = summary(fitted_race)
    result["fusion"] = {
        "method": "log((K-1)*acc/(1-acc)) of each backend's fit-half top-1 accuracy, clipped at 0",
        "gender": {
            "shipped_weights": {key: GENDER_FUSION_WEIGHTS.get(key, 1.0) for key in gender["backends"]},
            "fit_half_accuracy": {key: round(value, 4) for key, value in fit_gender_acc.items()},
            "fitted_weights": fitted_gender_weights,
            "test_shipped": gender["fused_shipped"], "test_fitted": gender["fused_fitted"],
            "test_fitted_minus_shipped": summary(
                np.asarray(fitted_gender, float) - np.asarray(shipped_gender, float)),
        },
        "race": {
            "shipped_weights": {key: RACE_FUSION_WEIGHTS.get(key, 1.0) for key in race["backends"]},
            "fit_half_accuracy": {key: round(value, 4) for key, value in fit_race_acc.items()},
            "fitted_weights": fitted_race_weights,
            "test_shipped": race["fused_shipped"], "test_fitted": race["fused_fitted"],
            "test_fitted_minus_shipped": summary(
                np.asarray(fitted_race, float) - np.asarray(shipped_race, float)),
        },
    }

    # The cache must reproduce what analyze_frame displayed; count any disagreement.
    mismatches = {"gender": 0, "race": 0, "best_age": 0, "fused_gender": 0, "fused_race": 0}
    for r in test_faces + fit_faces:
        out = r["outputs"]
        for key, label in gender_labels(r["raw"]).items():
            mismatches["gender"] += out.get(f"gender/{key}") != label
        for key, labels in (("fairface", RACE_LABELS_FAIRFACE), ("deepface", RACE_LABELS_DEEPFACE)):
            probs = r["raw"].get(f"race_probs/{key}")
            if probs is not None:
                mismatches["race"] += shown_top1(out.get(f"race/{key}", "")) != labels[int(np.argmax(probs))]
        chosen = select_age(age_estimates(r["raw"]))
        mismatches["best_age"] += out.get("age/best") != (f"{chosen[0]} ({chosen[1]})" if chosen else None)
        mismatches["fused_gender"] += out.get("gender/fused") != fused_gender(r["raw"])
        fused = fuse_race(race_distributions(r["raw"]))
        mismatches["fused_race"] += out.get("race/fused") != fused
    result["cache_vs_display_mismatches"] = mismatches
    result["fairface_landmark_alignment_share"] = round(float(np.mean(
        [bool(r["fairface_aligned"]) for r in test_faces + fit_faces])), 4)
    result.update(gender=gender, race=race, age=age)
    return result


def _pct(summary_: dict) -> str:
    """Format a summary as "93.1% (92.0-94.2)"."""
    low, high = summary_["ci95"]
    return f"{100 * summary_['value']:.1f}% ({100 * low:.1f}-{100 * high:.1f})"


def markdown_tables(report: dict) -> str:
    """Render the headline tables for docs/eval/heldout_fairface.md."""
    s = report["scores"]
    def offset_text(summary_: dict) -> str:
        low, high = summary_["ci95"]
        return f"{summary_['value']:.2f} ({low:.2f}-{high:.2f})"

    lines = ["| Backend | Gender | Race (6 canonical) | Age bucket | Mean bucket offset |",
             "| ------- | ------ | ------------------ | ---------- | ------------------ |"]
    keys = list(dict.fromkeys([*GENDER_BACKENDS, *RACE_BACKENDS, *AGE_BACKENDS]))
    for key in keys:
        gender = s["gender"]["backends"].get(key)
        race = s["race"]["backends"].get(key)
        age = s["age"]["backends"].get(key)
        lines.append(
            f"| `{key}` | {_pct(gender) if gender else '-'} | {_pct(race) if race else '-'} | "
            f"{_pct(age['bucket_accuracy']) if age else '-'} | "
            f"{offset_text(age['mean_bucket_offset']) if age else '-'} |")
    best = s["age"]["best_shipped"]
    lines.append(f"| **App (shipped)** | **{_pct(s['gender']['fused_shipped'])}** | "
                 f"**{_pct(s['race']['fused_shipped'])}** | **{_pct(best['bucket_accuracy'])}** | "
                 f"{offset_text(best['mean_bucket_offset'])} |")
    lines += ["", "| Fusion | Shipped weights | Fitted weights | Fitted - shipped |",
              "| ------ | --------------- | -------------- | ---------------- |"]
    for feature in ("gender", "race"):
        f = s["fusion"][feature]
        diff = f["test_fitted_minus_shipped"]
        lines.append(f"| {feature} | {_pct(f['test_shipped'])} | {_pct(f['test_fitted'])} | "
                     f"{100 * diff['value']:+.1f} pp ({100 * diff['ci95'][0]:+.1f} to "
                     f"{100 * diff['ci95'][1]:+.1f}) |")
    lines += ["", "| Canonical race | n | " + " | ".join(f"`{k}`" for k in s["race"]["backends"])
              + " | App (shipped) |",
              "| --- | --- | " + " | ".join("---" for _ in s["race"]["backends"]) + " | --- |"]
    for canonical, row in s["race"]["per_class_recall"].items():
        lines.append(f"| {RACE_CANONICAL_LABELS[canonical]} | {row['n']} | "
                     + " | ".join(f"{100 * row[k]:.1f}%" for k in s["race"]["backends"])
                     + f" | {100 * row['fused_shipped']:.1f}% |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--n", type=int, default=2000, help="images to sample (race-stratified)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument("--conf", type=float, default=0.5, help="SSD detection threshold")
    parser.add_argument("--config", default="1.25", help="FairFace padding config to evaluate")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="tools/fetch_fairface.py --dest")
    parser.add_argument("--cache", type=Path, default=None, help="default data/eval_cache/<config>")
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    parser.add_argument("--exclude", default="", help="comma-separated feature:backend to skip, "
                        "e.g. age:dex (recorded in the output)")
    parser.add_argument("--score-only", action="store_true", help="re-score the cache; run no models")
    args = parser.parse_args()

    data = args.data / args.config
    args.cache = args.cache or CACHE_DIR / args.config
    manifest = json.loads((data / "manifest.json").read_text())
    parquet = data / manifest["file"]
    labels = load_labels(parquet)
    rows = stratified_sample(labels["race"], args.n, args.seed)
    fit_rows, test_rows = stratified_split(rows, labels["race"], args.seed)
    exclude = {item for item in args.exclude.split(",") if item}

    if not args.score_only:
        run_inference(rows, labels, parquet, args.cache, exclude, args.conf, args.seed)

    records = [r for r in read_cache(args.cache / "predictions.jsonl") if r["row"] in set(rows)]
    missing = set(rows) - {r["row"] for r in records}
    if missing:
        raise SystemExit(f"{len(missing)} sampled rows are not cached yet; run without --score-only")
    meta = json.loads((args.cache / "meta.json").read_text())
    seconds = [r["seconds"] for r in records]
    report = {
        "dataset": {**manifest, "n_validation": len(labels["race"])},
        "sample": {"n": len(rows), "seed": args.seed, "fit": len(fit_rows), "test": len(test_rows),
                   "stratified_by": "race", "allocation": "proportional",
                   "bootstrap_resamples": args.resamples},
        "run": {**meta, "scored_on": date.today().isoformat(),
                "seconds_per_image_mean": round(float(np.mean(seconds)), 3),
                "seconds_per_image_median": round(float(np.median(seconds)), 3),
                "total_inference_minutes": round(sum(seconds) / 60, 1)},
        "emotion": "not evaluated: FairFace has no emotion labels",
        "scores": score(records, fit_rows, test_rows, args.seed, args.resamples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown_tables(report))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
