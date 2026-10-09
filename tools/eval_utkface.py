"""Out-of-distribution age (and gender) evaluation of every age backend on UTKFace.

UTKFace (Zhang, Song and Qi, 2017) is for non-commercial research only: it is used here for
evaluation alone, read from the untracked data/utkface/ and never redistributed. Download it
with the Kaggle CLI (dataset moritzm00/utkface-cropped) and unzip it into data/utkface/.

Draws an age-decade-stratified sample (fixed seed) of the aligned-and-cropped faces, pads each
200x200 face with black so the face fills about the same share of the frame as in the FairFace
1.25-padding images (the SSD detector misses faces that fill the frame), and runs the real
analyze_frame with every age backend and the cheap gender backends, as tools/eval_heldout.py
does. An age model that is not (yet) an app backend runs afterwards on the same detections and
aligned faces through --extra-age, exactly as in tools/eval_heldout.py. Outputs are cached in
data/eval_cache/utkface/, so the run resumes; --score-only re-scores the cache.

    FACE_ANALYZER_MODEL_DIR=$PWD/models python tools/eval_utkface.py \
        [--n 1500] [--extra-age convnext=data/kaggle_age/age_convnext_fairface.onnx] [--score-only]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_heldout import (  # noqa: E402
    AGE_BACKENDS,
    UNUSED_FEATURE_ENV,
    _instrument,
    _versions,
    age_buckets,
    age_estimates,
    age_to_bucket,
    gender_labels,
    merge_extra_age,
    read_cache,
    run_extra_age,
    shipped_age_estimates,
    stratified_sample,
    summarize,
)

from face_analyzer.core.constants import FAIRFACE_AGE_LABELS  # noqa: E402
from face_analyzer.fusion import select_age  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "utkface" / "UTKFace"
CACHE_DIR = ROOT / "data" / "eval_cache" / "utkface"
OUT_JSON = ROOT / "docs" / "eval" / "age_model.json"
HELDOUT_JSON = ROOT / "docs" / "eval" / "heldout_fairface.json"
TRAINING_JSON = ROOT / "data" / "kaggle_age" / "selection_metrics.json"

NAME = re.compile(r"^(\d+)_([01])_([0-4])_\d+\.jpg\.chip\.jpg$")
GENDERS = ("Male", "Female")  # UTKFace: 0 male, 1 female
# Black border on each side, as a share of the face image's size: a 200-pixel UTKFace face
# becomes a 466-pixel frame, about the face-to-frame ratio of the FairFace 1.25 images.
PAD_FRACTION = 2 / 3
AGE_SELECTION = "caffe,dex,fairface,mivolo"
GENDER_SELECTION = "caffe,fairface,mivolo"  # deepface skipped: about 1.7 s per face
CONTINUOUS_AGE = ("mivolo", "dex")


def parse_labels(directory: Path) -> list[dict]:
    """Read (file, age, gender, race) from UTKFace file names, skipping malformed names."""
    out = []
    for path in sorted(directory.iterdir()):
        match = NAME.match(path.name)
        if match:
            age, gender, race = (int(v) for v in match.groups())
            out.append({"file": path.name, "age": age, "gender": gender, "race": race})
    return out


def decade(age: int) -> int:
    """Return the age decade used for stratification, with 90 and over pooled."""
    return min(age // 10, 9)


def pad_face(face):
    """Surround a tightly cropped face with a black border (see PAD_FRACTION)."""
    import cv2

    pad = round(PAD_FRACTION * max(face.shape[:2]))
    return cv2.copyMakeBorder(face, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def run_inference(items: list[dict], data: Path, cache: Path) -> None:
    """Run analyze_frame with every age and the cheap gender backends on each uncached face."""
    for name in (*UNUSED_FEATURE_ENV, "EMOTION_MODEL", "RACE_MODEL"):
        os.environ[name] = ""
    os.environ["AGE_MODEL"] = AGE_SELECTION
    os.environ["GENDER_MODEL"] = GENDER_SELECTION
    import cv2

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
    done = {record["file"] for record in read_cache(predictions)}
    todo = [item for item in items if item["file"] not in done]
    print(f"{len(items) - len(todo)} of {len(items)} sampled faces already cached", flush=True)
    if not todo:
        return
    models = inference.load_models()
    config = inference.AnalysisConfig(conf_threshold=0.5, face_detector="ssd",
                                      active_age=set(models.age_nets), active_gender=set(models.gender_nets))
    meta = {"backends": {"age": sorted(config.active_age), "gender": sorted(config.active_gender)},
            "detector": "ssd", "conf_threshold": config.conf_threshold,
            "pad_fraction": PAD_FRACTION, "versions": _versions(), "date": date.today().isoformat()}
    (cache / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"backends: {meta['backends']}", flush=True)

    sink: dict = {}
    _instrument(sink)
    started_all = time.perf_counter()
    with predictions.open("a") as handle:
        for count, item in enumerate(todo, 1):
            sink.clear()
            sink.update(raw={}, latency_ms={}, detections=[], fairface_aligned=None)
            frame = pad_face(cv2.imread(str(data / item["file"]), cv2.IMREAD_COLOR))
            started = time.perf_counter()
            error, faces = None, []
            try:
                _, faces, _, _ = inference.analyze_frame(models, frame, config)
            except Exception as exc:  # recorded and counted, so a crash never drops a face silently
                error = f"{type(exc).__name__}: {exc}"
            record = {
                **item, "size": list(frame.shape[:2]), "detections": sink["detections"],
                "box": [int(v) for v in faces[0]["box"]] if faces else None,
                "raw": {k: v for k, v in sink["raw"].items() if k.split("/")[0] in (
                    "age_probs", "age_estimate", "mivolo", "gender_probs")},
                "outputs": {row["Feature"].lower() + "/" + row["Model"]: row["Output"]
                            for row in (faces[0]["model_results"] if faces else [])
                            if row["Feature"] in ("AGE", "GENDER")},
                "fairface_aligned": sink["fairface_aligned"], "latency_ms": sink["latency_ms"],
                "seconds": round(time.perf_counter() - started, 3), "error": error,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if count % 25 == 0 or count == len(todo):
                elapsed = time.perf_counter() - started_all
                print(f"{count}/{len(todo)} faces, {elapsed / count:.2f} s/face, "
                      f"ETA {elapsed / count * (len(todo) - count) / 60:.0f} min", flush=True)


def score(records: list[dict], seed: int, n_resamples: int = 1000, extra_age: Sequence[str] = ()) -> dict:
    """Bucket accuracy, within-one, MAE and gender accuracy per backend, with paired differences."""
    def summary(values):
        return summarize(values, seed, n_resamples)

    detected = [r for r in records if r["box"] is not None and r["error"] is None]
    truth_bucket = [age_to_bucket(r["age"]) for r in detected]
    present = [key for key in AGE_BACKENDS if any(key in age_buckets(r["raw"]) for r in detected)]
    result: dict = {
        "detection_recall": summary([r["box"] is not None and r["error"] is None for r in records]),
        "errors": sum(r["error"] is not None for r in records),
        "fairface_landmark_alignment_share": round(float(np.mean(
            [bool(r["fairface_aligned"]) for r in detected])), 4),
        "age": {"backends": {}}, "gender": {"backends": {}},
    }
    right, near = {}, {}
    for key in present:
        buckets = [age_buckets(r["raw"]).get(key) for r in detected]
        right[key] = np.asarray([b == t for b, t in zip(buckets, truth_bucket, strict=True)], float)
        near[key] = np.asarray([b is not None and abs(b - t) <= 1
                                for b, t in zip(buckets, truth_bucket, strict=True)], float)
        entry = {"bucket_accuracy": summary(right[key]), "within_one_bucket": summary(near[key]),
                 "mean_bucket_offset": summary([abs(b - t) for b, t in zip(buckets, truth_bucket, strict=True)
                                                if b is not None])}
        if key in CONTINUOUS_AGE:
            errors = [abs(age_estimates(r["raw"])[key] - r["age"]) for r in detected
                      if key in age_estimates(r["raw"])]
            entry["mae_years"] = summary(errors)
        result["age"]["backends"][key] = entry
    if "mivolo" in right:
        result["age"]["minus_mivolo"] = {
            key: {"bucket_accuracy": summary(right[key] - right["mivolo"]),
                  "within_one_bucket": summary(near[key] - near["mivolo"])}
            for key in present if key != "mivolo"}
    chosen = [select_age(shipped_age_estimates(r["raw"], extra_age)) for r in detected]
    headline = [None if c is None else age_to_bucket(float(c[0])) for c in chosen]
    result["age"]["best_shipped"] = {
        "bucket_accuracy": summary([h == t for h, t in zip(headline, truth_bucket, strict=True)]),
        "chosen_backend": {key: sum(1 for c in chosen if c and c[1] == key) for key in present},
    }
    result["age"]["per_true_bucket"] = {
        FAIRFACE_AGE_LABELS[b]: {"n": int(sum(t == b for t in truth_bucket)), **{
            key: round(float(np.mean([v for v, t in zip(right[key], truth_bucket, strict=True) if t == b])), 4)
            for key in present}}
        for b in sorted(set(truth_bucket))}
    for key in ("mivolo", "fairface", "caffe"):
        labels = [gender_labels(r["raw"]).get(key) for r in detected]
        if any(labels):
            result["gender"]["backends"][key] = summary(
                [label == GENDERS[r["gender"]] for label, r in zip(labels, detected, strict=True)])
    result["n_faces"] = len(detected)
    return result


def _pct(summary_: dict) -> str:
    """Format a summary as "93.1% (92.0-94.2)"."""
    low, high = summary_["ci95"]
    return f"{100 * summary_['value']:.1f}% ({100 * low:.1f}-{100 * high:.1f})"


def markdown_table(scores: dict) -> str:
    """Render the UTKFace age table for docs/eval/age_model.md."""
    lines = ["| Backend | Age bucket | Within one bucket | MAE (years) | Bucket acc. - `mivolo` | Gender |",
             "| ------- | ---------- | ----------------- | ----------- | ---------------------- | ------ |"]
    for key, entry in scores["age"]["backends"].items():
        mae = entry.get("mae_years")
        diff = scores["age"].get("minus_mivolo", {}).get(key)
        gender = scores["gender"]["backends"].get(key)
        diff_text = "-" if diff is None else (
            f"{100 * diff['bucket_accuracy']['value']:+.1f} pp "
            f"({100 * diff['bucket_accuracy']['ci95'][0]:+.1f} to {100 * diff['bucket_accuracy']['ci95'][1]:+.1f})")
        mae_text = "-" if mae is None else f"{mae['value']:.1f} ({mae['ci95'][0]:.1f}-{mae['ci95'][1]:.1f})"
        lines.append(f"| `{key}` | {_pct(entry['bucket_accuracy'])} | {_pct(entry['within_one_bucket'])} | "
                     f"{mae_text} | {diff_text} | {_pct(gender) if gender else '-'} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--n", type=int, default=1500, help="faces to sample (age-decade-stratified)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    parser.add_argument("--training", type=Path, default=TRAINING_JSON,
                        help="the training kernel's selection_metrics.json, copied into the output")
    parser.add_argument("--extra-age", action="append", default=[], metavar="KEY=MODEL.onnx",
                        help="also run an age model that is not an app backend")
    parser.add_argument("--score-only", action="store_true", help="re-score the cache; run no models")
    args = parser.parse_args()

    labels = parse_labels(args.data)
    picked = stratified_sample([decade(item["age"]) for item in labels], args.n, args.seed)
    items = [labels[i] for i in picked]
    extra = dict(item.split("=", 1) if "=" in item else (item, "") for item in args.extra_age)
    if not args.score_only:
        run_inference(items, args.data, args.cache)
        import cv2

        for key, onnx in extra.items():
            run_extra_age([item["file"] for item in items],
                          lambda name: pad_face(cv2.imread(str(args.data / name), cv2.IMREAD_COLOR)),
                          args.cache, key, Path(onnx), id_field="file")

    wanted = {item["file"] for item in items}
    records = [r for r in read_cache(args.cache / "predictions.jsonl") if r["file"] in wanted]
    if len(records) != len(items):
        raise SystemExit(f"{len(items) - len(records)} sampled faces are not cached yet")
    extra_mismatches = merge_extra_age(records, args.cache, list(extra), id_field="file")
    meta = json.loads((args.cache / "meta.json").read_text())
    seconds = [r["seconds"] for r in records]
    utkface = {
        "dataset": {"name": "UTKFace (aligned and cropped faces)", "source": "kaggle moritzm00/utkface-cropped",
                    "license": "non-commercial research only; used for evaluation, not redistributed",
                    "n_parsed": len(labels), "n_malformed_names_skipped": sum(
                        1 for _ in args.data.iterdir()) - len(labels)},
        "sample": {"n": len(items), "seed": args.seed, "stratified_by": "age decade (90+ pooled)",
                   "allocation": "proportional", "bootstrap_resamples": args.resamples},
        "extra_age": {"backends": sorted(extra), "crop_mismatches": extra_mismatches},
        "run": {**meta, "scored_on": date.today().isoformat(),
                "seconds_per_face_mean": round(float(np.mean(seconds)), 3),
                "total_inference_minutes": round(sum(seconds) / 60, 1)},
        "scores": score(records, args.seed, args.resamples, list(extra)),
    }
    report: dict = {"utkface": utkface}
    if HELDOUT_JSON.exists():
        heldout = json.loads(HELDOUT_JSON.read_text())
        report["fairface_test_half"] = {"source": "heldout_fairface.json", "age": heldout["scores"]["age"],
                                        "extra_age": heldout.get("extra_age")}
    if args.training.exists():
        report["training"] = json.loads(args.training.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown_table(utkface["scores"]))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
