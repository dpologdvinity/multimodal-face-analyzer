"""Score inference.py's combined answers (fuse_* and select_age) against tmp/predictions.json.

Uses the shipped fusion functions, not a copy of them, so a rule that scores well here is the
rule the app actually runs. Re-run tools/dump_predictions.py first if a model changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import inference as inf  # noqa: E402

RACE_TRUTH_TO_LABEL = {key: label for key, label in inf.RACE_CANONICAL_LABELS.items()}


def age_estimates(record: dict) -> dict[str, float]:
    """Rebuild the per-model year estimates analyze_frame feeds to fuse_age."""
    raw = record["age"]
    estimates: dict[str, float] = {}
    if "caffe" in raw:
        bucket = int(np.argmax(raw["caffe"]))
        estimates["caffe"] = float(np.mean(inf.AGE_LIST_RANGES[bucket]))
    if "ssrnet" in raw:
        estimates["ssrnet"] = raw["ssrnet"]
    if "dex" in raw:
        estimates["dex"] = raw["dex"]
    if "fairface_probs" in raw:
        bucket = int(np.argmax(raw["fairface_probs"]))
        estimates["fairface"] = float(np.mean(inf.FAIRFACE_AGE_RANGES[bucket]))
    if "mivolo" in raw:
        estimates["mivolo"] = raw["mivolo"]
    return estimates


def race_distributions(record: dict) -> dict[str, dict[str, float]]:
    """Rebuild the per-model canonical race distributions analyze_frame feeds to fuse_race."""
    out = {}
    if "fairface" in record["race"]:
        out["fairface"] = inf.canonical_race_probabilities(
            np.array(record["race"]["fairface"]), inf.RACE_LABELS_FAIRFACE)
    if "deepface" in record["race"]:
        out["deepface"] = inf.canonical_race_probabilities(
            np.array(record["race"]["deepface"]), inf.RACE_LABELS_DEEPFACE)
    return out


def race_label_matches(label: str, accepted: list[str]) -> bool:
    """A race label (possibly "A (52%)/B (47%)") counts as correct if any shown class is accepted."""
    shown = {part.split("(")[0].strip() for part in label.split("/")}
    return any(RACE_TRUTH_TO_LABEL[key] in shown for key in accepted)


def main() -> None:
    records = json.loads((ROOT / "tmp" / "predictions.json").read_text())
    scores = {feature: [0, 0, []] for feature in ("age", "gender", "race", "emotion")}

    for record in records:
        truth = record["truth"]
        if "age" in truth:
            chosen = inf.select_age(age_estimates(record))
            fused = None if chosen is None else f"{chosen[0]} ({chosen[1]})"
            low, high = truth["age"]
            correct = chosen is not None and low <= float(chosen[0]) <= high
            scores["age"][1] += 1
            scores["age"][0] += int(correct)
            if not correct:
                scores["age"][2].append(f"{record['id']}={fused} want {low}-{high}")
        if "gender" in truth:
            fused = inf.fuse_gender(record["gender"])
            correct = fused == truth["gender"]
            scores["gender"][1] += 1
            scores["gender"][0] += int(correct)
            if not correct:
                scores["gender"][2].append(f"{record['id']}={fused} want {truth['gender']}")
        if "race" in truth and record["race"]:
            fused = inf.fuse_race(race_distributions(record))
            correct = fused is not None and race_label_matches(fused, truth["race"])
            scores["race"][1] += 1
            scores["race"][0] += int(correct)
            if not correct:
                scores["race"][2].append(f"{record['id']}={fused} want {truth['race']}")
        if "emotion" in truth:
            fused = inf.fuse_emotion(record["emotion"])
            correct = fused in truth["emotion"]
            scores["emotion"][1] += 1
            scores["emotion"][0] += int(correct)
            if not correct:
                scores["emotion"][2].append(f"{record['id']}={fused} want {truth['emotion']}")

    for feature, (correct, total, misses) in scores.items():
        print(f"{feature.upper():<8} fused {correct}/{total} = {100 * correct / max(1, total):5.1f}%")
        for miss in misses:
            print(f"    MISS {miss}")


if __name__ == "__main__":
    main()
