"""Fit and test a learned stacker (L2 logistic regression) for gender, race and age bucket.

Works from the held-out eval's cache only (no model is run): the same 2,000 FairFace
validation faces, the same seeded fit/test split, scoring and bootstrap as
tools/eval_heldout.py. For each attribute it fits a logistic regression on every backend's
output on the fit half, picks the L2 strength C by 5-fold CV inside the fit half, scores the
test half once, and compares it with the shipped headline and the best single backend by
paired bootstrap. Needs scikit-learn (install it into the eval venv only):

    .venv-eval/bin/python tools/fit_stacker.py

Writes docs/eval/stacker_fairface.json: the fitted models as plain JSON (feature spec,
standardisation, coefficients, intercepts, class order), CV and test results, and the
adoption decision (ship only if the paired CI against the shipped answer is above zero).
predict_stacked below is the pure-numpy inference an adopted model would ship with; the
run asserts it reproduces scikit-learn on every test face. `--fixture` writes a few of those
faces for tests/test_fit_stacker.py. Method and results: docs/eval/heldout_fairface.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_heldout as heldout  # noqa: E402  (a sibling script, not a package)

from face_analyzer.core.constants import (  # noqa: E402
    RACE_LABEL_TO_CANONICAL,
    RACE_LABELS_DEEPFACE,
    RACE_LABELS_FAIRFACE,
)

OUT_JSON = heldout.ROOT / "docs" / "eval" / "stacker_fairface.json"
C_GRID = [float(c) for c in np.logspace(-3, 3, 13)]
CV_FOLDS = 5
N_AGE_BUCKETS = 9
# Probabilities are clipped before log/logit so a saturated backend's exact 0 or 1 stays finite.
PROBABILITY_EPS = 1e-6
FIXTURE_FACES = 4


# --------------------------------------------------------------------------- numpy inference
# Pure numpy on the exported JSON, so the fitted models could ship without scikit-learn.


def _feature_value(feature: Mapping, source) -> float:
    """Compute one feature from its backend output, following the feature's op."""
    op = feature["op"]
    if op in ("log", "logit"):
        probs = np.asarray(source, dtype=float)
        probability = float(probs[feature["indices"]].sum() / (probs.sum() or 1.0))
        if op == "log":
            return math.log(max(probability, PROBABILITY_EPS))
        probability = min(max(probability, PROBABILITY_EPS), 1 - PROBABILITY_EPS)
        return math.log(probability / (1 - probability))
    value = source[feature["index"]]
    if op == "value":
        return float(value)
    if op == "log1p":
        return math.log1p(max(float(value), 0.0))
    if op == "bucket":
        return float(heldout.age_to_bucket(float(value)) == feature["bucket"])
    if op == "equals":
        return float(value == feature["value"])
    raise ValueError(f"unknown stacker feature op {op!r}")


def stacker_features(spec: Sequence[Mapping], raw: Mapping) -> np.ndarray | None:
    """Build the feature vector from per-backend outputs; None if any source is missing."""
    values = []
    for feature in spec:
        source = raw.get(feature["source"])
        if source is None:
            return None
        values.append(_feature_value(feature, source))
    return np.array(values, dtype=float)


def predict_stacked(model: Mapping, raw: Mapping) -> tuple | None:
    """Return (label, class probabilities), or None when a feature backend gave no output."""
    features = stacker_features(model["features"], raw)
    if features is None:
        return None
    standardized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = np.asarray(model["coef"]) @ standardized + np.asarray(model["intercept"])
    if logits.size == 1:  # binary: one logit for the second class against the first
        logits = np.array([0.0, logits[0]])
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return model["classes"][int(np.argmax(probs))], probs


def stacked_or_shipped(model: Mapping, raw: Mapping, attribute: str):
    """The stacker's answer, or the shipped best-model answer when a feature backend is missing."""
    stacked = predict_stacked(model, raw)
    return shipped_answer(raw, attribute) if stacked is None else stacked[0]


# --------------------------------------------------------------------------- fitting


def _canonical_groups(labels: list[str]) -> list[list[int]]:
    """Indices of a backend's classes that make up each canonical race, in canonical order."""
    return [[i for i, label in enumerate(labels) if RACE_LABEL_TO_CANONICAL[label.lower()] == key]
            for key in heldout.CANONICAL_RACES]


def _continuous_age(source: str, index: int, name: str) -> list[dict]:
    """Age in years, its log1p and the FairFace bucket the app's displayed integer falls in."""
    return ([{"name": f"{name}_age", "source": source, "op": "value", "index": index},
             {"name": f"{name}_log1p_age", "source": source, "op": "log1p", "index": index}]
            + [{"name": f"{name}_bucket_{k}", "source": source, "op": "bucket", "index": index,
                "bucket": k} for k in range(N_AGE_BUCKETS)])


def feature_spec(attribute: str) -> list[dict]:
    """Every backend output available for the attribute, as JSON feature definitions."""
    if attribute == "gender":
        return [
            {"name": "fairface_male_logit", "source": "gender_probs/fairface", "op": "logit", "indices": [0]},
            {"name": "caffe_male_logit", "source": "gender_probs/caffe", "op": "logit", "indices": [0]},
            {"name": "deepface_male_logit", "source": "gender_probs/deepface", "op": "logit", "indices": [1]},
            {"name": "mivolo_is_male", "source": "mivolo/face", "op": "equals", "index": 1, "value": "Male"},
        ]
    if attribute == "race":
        spec = []
        for backend, labels in (("fairface", RACE_LABELS_FAIRFACE), ("deepface", RACE_LABELS_DEEPFACE)):
            for key, group in zip(heldout.CANONICAL_RACES, _canonical_groups(labels), strict=True):
                spec.append({"name": f"{backend}_log_p_{key}", "source": f"race_probs/{backend}",
                             "op": "log", "indices": group})
        return spec
    if attribute == "age":
        spec = [{"name": f"fairface_log_p_{k}", "source": "age_probs/fairface", "op": "log",
                 "indices": [k]} for k in range(N_AGE_BUCKETS)]
        spec += [{"name": f"caffe_log_p_{k}", "source": "age_probs/caffe", "op": "log",
                  "indices": [k]} for k in range(8)]
        spec += _continuous_age("age_estimate/dex", 0, "dex")
        spec.append({"name": "dex_sd", "source": "age_estimate/dex", "op": "value", "index": 1})
        spec += _continuous_age("mivolo/face", 0, "mivolo")
        return spec
    raise ValueError(attribute)


def target(record: dict, attribute: str):
    """The record's ground truth for an attribute, as the eval scores it."""
    return heldout._truth(record)[attribute]


def single_answers(raw: dict, attribute: str) -> dict:
    """Each backend's own answer, mapped the way the eval scores it."""
    if attribute == "gender":
        return heldout.gender_labels(raw)
    if attribute == "race":
        return heldout.race_labels(raw)
    return heldout.age_buckets(raw)


def shipped_answer(raw: Mapping, attribute: str):
    """The app's currently shipped headline answer for an attribute (None when it shows none)."""
    if attribute == "gender":
        chosen = heldout.best_gender(raw)
    elif attribute == "race":
        chosen = heldout.best_race(raw)
    else:
        chosen = heldout.best_age_bucket(raw)
    return None if chosen is None else chosen[0]


def cross_validate(features: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Pick C by stratified 5-fold CV accuracy on the fit half (ties go to the smaller C)."""
    from sklearn.model_selection import StratifiedKFold

    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(features, y))
    scores = {}
    for c in C_GRID:
        accuracies = []
        for train, valid in folds:
            model = _fit(features[train], y[train], c)
            accuracies.append(float(np.mean(model.predict(features[valid]) == y[valid])))
        scores[c] = float(np.mean(accuracies))
    best = max(C_GRID, key=lambda c: (scores[c], -c))
    return {"C": best, "cv_accuracy": round(scores[best], 4),
            "grid": {f"{c:g}": round(score, 4) for c, score in scores.items()}}


def _fit(features: np.ndarray, y: np.ndarray, c: float):
    """Standardise, then fit an L2 logistic regression (multinomial when there are >2 classes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=10000)).fit(features, y)


def export(model, attribute: str, spec: list[dict], cv: dict) -> dict:
    """The fitted pipeline as plain JSON: everything predict_stacked needs."""
    scaler, logistic = model[0], model[1]
    classes = [_plain(c) for c in logistic.classes_]
    return {"attribute": attribute, "classes": classes, "features": spec,
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": logistic.coef_.tolist(), "intercept": logistic.intercept_.tolist(),
            "C": cv["C"], "cv_accuracy": cv["cv_accuracy"]}


def _plain(value):
    """Unwrap a numpy scalar into the plain Python value JSON stores."""
    return value.item() if isinstance(value, np.generic) else value


# --------------------------------------------------------------------------- report


def markdown_table(report: dict) -> str:
    """Render the test-half stacker results for docs/eval/heldout_fairface.md."""
    def diff(summary_: dict) -> str:
        low, high = summary_["ci95"]
        return f"{100 * summary_['value']:+.1f} pp ({100 * low:+.1f} to {100 * high:+.1f})"

    lines = ["| Attribute | C | Fit-half CV | Stacked (test) | Shipped (test) | Stacked - shipped "
             "| Stacked - best single | Adopted |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for attribute, entry in report["attributes"].items():
        r = entry["result"]
        lines.append(
            f"| {attribute} | {r['C']:g} | {100 * r['cv_accuracy']:.1f}% | "
            f"{heldout._pct(r['test_stacked'])} | {heldout._pct(r['test_shipped'])} | "
            f"{diff(r['stacked_minus_shipped'])} | {diff(r['stacked_minus_best_single'])} "
            f"(`{r['best_single']}`) | {'yes' if r['adopted'] else 'no'} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--config", default="1.25")
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    parser.add_argument("--fixture", type=Path, default=None,
                        help="also write a few test faces with scikit-learn's predictions, for tests")
    args = parser.parse_args()

    data = heldout.DATA_DIR / args.config
    manifest = json.loads((data / "manifest.json").read_text())
    labels = heldout.load_labels(data / manifest["file"])
    rows = heldout.stratified_sample(labels["race"], args.n, args.seed)
    fit_rows, test_rows = heldout.stratified_split(rows, labels["race"], args.seed)
    by_row = {r["row"]: r for r in heldout.read_cache(heldout.CACHE_DIR / args.config / "predictions.jsonl")}
    fit = [by_row[row] for row in fit_rows if heldout._detected(by_row[row])]
    test = [by_row[row] for row in test_rows if heldout._detected(by_row[row])]

    def summary(values):
        return heldout.summarize(values, args.seed, args.resamples)

    report: dict = {"method": {
        "model": "L2 logistic regression on standardised features (scikit-learn, lbfgs)",
        "C_grid": C_GRID, "cv": f"stratified {CV_FOLDS}-fold on the fit half, seed {args.seed}",
        "fit_faces": len(fit), "test_faces": len(test), "date": date.today().isoformat(),
        "adoption_rule": "ship only if the paired-bootstrap 95% CI lower bound of "
                         "(stacked - shipped) on the test half is above 0"}, "attributes": {}}
    fixture = [{"row": r["row"], "raw": {}, "expected": {}} for r in test[:FIXTURE_FACES]]
    for attribute in ("gender", "race", "age"):
        spec = feature_spec(attribute)
        complete = [r for r in fit if stacker_features(spec, r["raw"]) is not None]
        x = np.array([stacker_features(spec, r["raw"]) for r in complete])
        y = np.array([target(r, attribute) for r in complete])
        cv = cross_validate(x, y, args.seed)
        model = _fit(x, y, cv["C"])
        exported = export(model, attribute, spec, cv)

        # The numpy inference on the exported JSON must reproduce scikit-learn on every face.
        test_x = [stacker_features(spec, r["raw"]) for r in test]
        sk_labels = [None if f is None else _plain(model.predict(f[None])[0]) for f in test_x]
        np_labels = [None if f is None else predict_stacked(exported, r["raw"])[0]
                     for f, r in zip(test_x, test, strict=True)]
        assert sk_labels == np_labels, "numpy stacker disagrees with scikit-learn"
        for entry, r, f in zip(fixture, test, test_x, strict=False):
            if f is None:  # a missing backend: this face falls back, so it has no stacker answer
                continue
            entry["raw"].update({feature["source"]: r["raw"][feature["source"]] for feature in spec})
            entry["expected"][attribute] = {"label": _plain(model.predict(f[None])[0]),
                                            "probs": model.predict_proba(f[None])[0].tolist()}

        truth = [target(r, attribute) for r in test]
        shipped = [shipped_answer(r["raw"], attribute) for r in test]
        stacked = [stacked_or_shipped(exported, r["raw"], attribute) for r in test]
        fit_single = {}
        for r in fit:
            for key, value in single_answers(r["raw"], attribute).items():
                fit_single.setdefault(key, []).append(value == target(r, attribute))
        fit_single_accuracy = {key: round(float(np.mean(v)), 4) for key, v in fit_single.items()}
        best_key = max(fit_single_accuracy, key=fit_single_accuracy.get)
        single = [single_answers(r["raw"], attribute).get(best_key) for r in test]

        correct = {name: np.array([p == t for p, t in zip(preds, truth, strict=True)], float)
                   for name, preds in (("stacked", stacked), ("shipped", shipped), ("single", single))}
        # Ceiling for any rule that picks or combines these backends' top-1 answers.
        any_right = [any(v == t for v in single_answers(r["raw"], attribute).values())
                     for r, t in zip(test, truth, strict=True)]
        result = {
            "fit_half_single_accuracy": fit_single_accuracy, "best_single": best_key,
            "C": cv["C"], "cv_accuracy": cv["cv_accuracy"], "cv_grid": cv["grid"],
            "fallback_faces": sum(v is None for v in np_labels),
            "test_stacked": summary(correct["stacked"]),
            "test_shipped": summary(correct["shipped"]),
            "test_best_single": summary(correct["single"]),
            "stacked_minus_shipped": summary(correct["stacked"] - correct["shipped"]),
            "stacked_minus_best_single": summary(correct["stacked"] - correct["single"]),
            "any_backend_right": summary(any_right),
        }
        if attribute == "age":
            for name, preds in (("stacked", stacked), ("shipped", shipped), ("single", single)):
                result[f"within_one_{name}"] = summary(
                    [p is not None and abs(p - t) <= 1 for p, t in zip(preds, truth, strict=True)])
        result["adopted"] = result["stacked_minus_shipped"]["ci95"][0] > 0
        report["attributes"][attribute] = {"result": result, "model": exported}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.fixture:
        args.fixture.write_text(json.dumps(fixture, indent=1) + "\n")
        print(f"wrote {args.fixture}")
    print(markdown_table(report))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
