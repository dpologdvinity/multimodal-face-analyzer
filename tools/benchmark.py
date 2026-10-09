"""Score every loaded age/gender/race/emotion model against tools/ground_truth.json.

Runs the real analyze_frame() pipeline over assets/, binds each detection to its hand-labelled
face by nearest normalized centre, and reports per-model accuracy plus face-detection recall.

Age is scored by RANGE OVERLAP, which favours the bucketed models: FairFace answering "30-39"
is correct if any of those ten years is plausible, while MiVOLO answering "34" must land
inside the truth range. Compare bucketed and continuous age models on that footing only --
tools/score_fusion.py scores single-year answers, which is the stricter, like-for-like view.

    FACE_ANALYZER_MODEL_DIR=/path/to/models python tools/benchmark.py [--detector ssd]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

from face_analyzer import inference

ROOT = Path(__file__).resolve().parent.parent


# A detection counts as "the labelled face" only within this normalized centre distance.
CENTER_MATCH_TOLERANCE = 0.05

# Model label -> canonical ground-truth race key. FairFace splits Asian into East/Southeast;
# both collapse to one key because the labels do not reliably distinguish them by eye.
RACE_ALIASES = {
    "white": "white",
    "black": "black",
    "east asian": "asian",
    "southeast asian": "asian",
    "asian": "asian",
    "indian": "indian",
    "latino_hispanic": "latino",
    "latino hispanic": "latino",
    "latino": "latino",  # RACE_CANONICAL_LABELS' name, kept for outputs from the old fused race answer
    "middle eastern": "middle_eastern",
}

EMOTION_ALIASES = {
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "angry": "angry", "anger": "angry",
    "surprise": "surprise", "surprised": "surprise",
    "fear": "fear", "fearful": "fear",
    "disgust": "disgust", "disgusted": "disgust",
    "neutral": "neutral", "contempt": "contempt",
}

CONFIRMED_MODELS = {
    "age": ("best", "mivolo", "fairface", "caffe", "dex"),
    "emotion": ("fused", "dan", "hsemotion", "ferplus", "mini_xception"),
    "eye color": ("colorimetric",),
}


def parse_confirmed(markdown: str) -> dict:
    """Read exact user-confirmed labels; never widen or infer labels."""
    labels = {}
    current = None
    fields = {"age": "age", "emotion": "emotion", "eyes": "eye color"}
    for line in markdown.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            if not current or Path(current).name != current or current in labels:
                raise ValueError(f"Invalid or duplicate image: {current}")
            labels[current] = {}
        elif line.startswith("* ") and "=" in line:
            name, value = (part.strip() for part in line[2:].split("=", 1))
            feature = fields.get(name.lower())
            if feature is None:
                continue
            if current is None or not value or feature in labels[current]:
                raise ValueError(f"Invalid or duplicate label: {line}")
            if feature == "age":
                age = int(value)
                if not 0 <= age <= 122:
                    raise ValueError(f"Invalid age: {value}")
                labels[current][feature] = age
            else:
                values = [part.strip().lower() for part in value.split("/")]
                if not all(values):
                    raise ValueError(f"Empty label: {line}")
                labels[current][feature] = [
                    {"dark brown": "brown", "blond": "blonde", "gray": "grey"}.get(part, part)
                    for part in values
                ]
    if not labels or any(not values for values in labels.values()):
        raise ValueError("No supported confirmed labels for an image")
    return labels


def score_confirmed(records: list[dict], roster: dict = CONFIRMED_MODELS) -> dict:
    """Score every labelled opportunity, including absent models and missed faces."""
    scores = {}
    for feature, model_names in roster.items():
        labelled = [record for record in records if feature in record["truth"]]
        scores[feature] = {}
        for model in model_names:
            bucket = {"correct": 0, "total": len(labelled), "answered": 0}
            errors = []
            if feature == "age":
                bucket.update(within_5_years=0, bucket_contains=0)
            for record in labelled:
                value = next((str(row["Output"]).strip().lower() for row in record["outputs"]
                              if row["Feature"].lower() == feature
                              and row["Model"].split(" (")[0] == model), "")
                truth = record["truth"][feature]
                if feature == "age":
                    if model == "best":
                        source = re.search(r"\((\w+)\)", value)
                        source_value = next((str(row["Output"]).strip().lower()
                                             for row in record["outputs"]
                                             if source and row["Feature"].lower() == "age"
                                             and row["Model"] == source.group(1)), "")
                        source_span = parse_age(source_value)
                        if "uncertain" in source_value or source_span is None:
                            continue
                        if source_span[0] != source_span[1]:
                            value = source_value
                    span = None if "uncertain" in value else parse_age(value)
                    if span is None:
                        continue
                    bucket["answered"] += 1
                    if span[0] != span[1]:
                        bucket["bucket_contains"] += int(span[0] <= truth <= span[1])
                        continue
                    error = abs(span[0] - truth)
                    errors.append(error)
                    bucket["correct"] += int(error == 0)
                    bucket["within_5_years"] += int(error <= 5)
                else:
                    bucket["answered"] += int(bool(value) and value != "unknown")
                    canonical = EMOTION_ALIASES.get(value, value) if feature == "emotion" else value
                    bucket["correct"] += int(canonical in truth)
            if feature == "age":
                bucket["mae_years"] = sum(errors) / len(errors) if errors else None
                bucket["numeric_answers"] = len(errors)
            scores[feature][model] = bucket
    return scores


def run_confirmed(args) -> None:
    """Evaluate real pipeline against assets/confirmed.md."""
    confirmed_path = args.assets / "confirmed.md"
    labels = parse_confirmed(confirmed_path.read_text())
    if args.image:
        labels = {args.image: labels[args.image]}
    models = inference.load_models()
    detector_models = {"yolo": models.yolo_face_nets, "scrfd": models.scrfd_face_nets,
                       "retinaface": models.retinaface_nets}
    if args.detector != "ssd" and args.detector not in detector_models[args.detector]:
        raise ValueError(f"Requested detector {args.detector} is unavailable; refusing silent SSD fallback")
    records = []
    for name, truth in labels.items():
        path = args.assets / name
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read confirmed asset: {path}")
        _, faces, _, _ = inference.analyze_frame(
            models, frame,
            inference.AnalysisConfig(
                conf_threshold=args.conf, active_age=set(models.age_nets),
                active_emotion=set(models.emotion_nets),
                active_eye_color=set(models.eye_color_nets), face_detector=args.detector,
            ),
        )
        records.append({"image": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "truth": truth, "detections": len(faces),
                        "box": list(faces[0]["box"]) if len(faces) == 1 else None,
                        "outputs": faces[0]["model_results"] if len(faces) == 1 else []})
        print(f"{name}: {len(faces)} detections", flush=True)
    report = {
        "annotation_sha256": hashlib.sha256(confirmed_path.read_bytes()).hexdigest(),
        "inference_sha256": hashlib.sha256(Path(inference.__file__).read_bytes()).hexdigest(),
        "detector": args.detector, "confidence": args.conf,
        "single_face_matches": sum(record["detections"] == 1 for record in records),
        "images": len(records), "scores": score_confirmed(records), "records": records,
        "limitations": "Development set, not held out. Exact ages; missing outputs count as errors. "
                       "Age MAE covers numeric answers only. Bucket containment is not exact accuracy. "
                       "Race and gender are not inferred or evaluated.",
    }
    print(json.dumps(report["scores"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def parse_age(text: str) -> tuple[float, float] | None:
    """Parse any age output shape into an inclusive (low, high) year range.

    Handles the Caffe buckets "(25-32)", FairFace's "30-39"/"70+", the continuous models'
    bare "31", and DEX's "uncertain (mean 40, SD 12)" (scored on the mean, as displayed).
    """
    text = text.strip()
    annotated = re.match(r"(\d+(?:\.\d+)?)\s*\(", text)  # consensus: "31 (2 models agree ...)"
    if annotated:
        value = float(annotated.group(1))
        return (value, value)
    mean = re.search(r"mean\s+(\d+)", text)
    if mean:
        value = float(mean.group(1))
        return (value, value)
    span = re.fullmatch(r"\(?(\d+)\s*-\s*(\d+)\)?", text)
    if span:
        return (float(span.group(1)), float(span.group(2)))
    open_ended = re.fullmatch(r"\(?(\d+)\+\)?", text)
    if open_ended:
        return (float(open_ended.group(1)), 120.0)
    single = re.fullmatch(r"\(?(\d+(?:\.\d+)?)\)?", text)
    if single:
        value = float(single.group(1))
        return (value, value)
    return None


def parse_race(text: str) -> set[str]:
    """Canonical race keys from a "White (52%)/Black (47%)" style label (top-2 counts as either)."""
    keys = set()
    for part in text.split("/"):
        name = part.split("(")[0].strip().lower()
        if name in RACE_ALIASES:
            keys.add(RACE_ALIASES[name])
    return keys


def score_face(truth: dict, outputs: list[dict], tally: dict) -> None:
    """Accumulate correct/total counts for one labelled face into tally[feature][model]."""
    for row in outputs:
        feature = row["Feature"].lower()
        model = row["Model"]
        value = str(row["Output"])
        if feature == "gender" and "gender" in truth:
            # A headline row reads "Male (mivolo)"; score only the label.
            correct = value.split(" (")[0].strip().lower() == truth["gender"].lower()
        elif feature == "race" and "race" in truth:
            correct = bool(parse_race(value) & set(truth["race"]))
        elif feature == "emotion" and "emotion" in truth:
            correct = EMOTION_ALIASES.get(value.strip().lower()) in truth["emotion"]
        elif feature == "age" and "age" in truth:
            span = parse_age(value)
            low, high = truth["age"]
            correct = span is not None and span[0] <= high and span[1] >= low
        else:
            continue
        bucket = tally[feature][model]
        bucket["total"] += 1
        bucket["correct"] += int(correct)
        if not correct:
            bucket["misses"].append(f"{truth['_image']}#{truth['_index']}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="ssd", choices=inference.FACE_DETECTOR_OPTIONS)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--misses", action="store_true", help="print every wrong prediction")
    parser.add_argument("--image", default=None, help="restrict to one asset filename")
    parser.add_argument("--confirmed", action="store_true", help="score exact labels from assets/confirmed.md")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, help="save confirmed evaluation and predictions as JSON")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be in (0, 1]")
    if args.output and not args.confirmed:
        parser.error("--output requires --confirmed")
    if args.confirmed:
        run_confirmed(args)
        return

    truth_data = json.loads((ROOT / "tools" / "ground_truth.json").read_text())
    models = inference.load_models()
    every = lambda nets: set(nets)  # noqa: E731 -- run every loaded model of each feature

    tally: dict = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0, "misses": []}))
    detected_total = labelled_total = 0

    for name, faces in sorted(truth_data.items()):
        if name.startswith("_") or (args.image and name != args.image):
            continue
        frame = cv2.imread(str(ROOT / "assets" / name))
        if frame is None:
            print(f"!! missing asset {name}")
            continue
        height, width = frame.shape[:2]
        _, cropped, _, _ = inference.analyze_frame(
            models, frame,
            inference.AnalysisConfig(
                conf_threshold=args.conf, active_age=every(models.age_nets),
                active_gender=every(models.gender_nets), active_emotion=every(models.emotion_nets),
                active_race=every(models.race_nets), face_detector=args.detector,
            ),
        )
        labelled_total += len(faces)
        matched = set()
        for index, truth in enumerate(faces):
            tx, ty = truth["center"]
            best, best_distance = None, CENTER_MATCH_TOLERANCE
            for face in cropped:
                x1, y1, x2, y2 = face["box"]
                cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
                distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if distance < best_distance:
                    best, best_distance = face, distance
            if best is None:
                continue
            matched.add(id(best))
            detected_total += 1
            score_face({**truth, "_image": name, "_index": index}, best["model_results"], tally)
        print(f"{name}: matched {len(matched)}/{len(faces)} labelled faces "
              f"({len(cropped)} detections)")

    print(f"\nFACE DETECTION recall: {detected_total}/{labelled_total} "
          f"= {100 * detected_total / max(1, labelled_total):.1f}%")
    for feature in sorted(tally):
        print(f"\n{feature.upper()}")
        rows = sorted(tally[feature].items(), key=lambda kv: -kv[1]["correct"] / max(1, kv[1]["total"]))
        for model, bucket in rows:
            pct = 100 * bucket["correct"] / max(1, bucket["total"])
            print(f"  {model:<16} {bucket['correct']:>3}/{bucket['total']:<3} = {pct:5.1f}%")
            if args.misses:
                for miss in bucket["misses"]:
                    print(f"      MISS {miss}")


if __name__ == "__main__":
    main()
