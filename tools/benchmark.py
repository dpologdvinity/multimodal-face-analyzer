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
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import inference  # noqa: E402

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
    "latino": "latino",  # inference.RACE_CANONICAL_LABELS' name, used by the fused answer
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
            correct = value.strip().lower() == truth["gender"].lower()
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
    args = parser.parse_args()

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
            models, frame, args.conf,
            every(models.age_nets), every(models.gender_nets), every(models.emotion_nets),
            every(models.race_nets), set(), {}, set(), set(), set(), set(), set(), set(), set(),
            {}, {}, face_detector=args.detector,
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
