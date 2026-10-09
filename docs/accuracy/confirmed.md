# Confirmed-data evaluation

`assets/confirmed.md` is authoritative. The evaluator does not widen ages or
invent labels. Missing detections, unavailable models, and unknown outputs remain
in denominators. Age buckets get containment counts, not exact-age credit; exact
age, numeric MAE, and within-five-years are separate metrics.

Run from the repository root. `tools/benchmark.py` imports the `face_analyzer` package, so
install it first with `pip install --no-deps -e .` (after `requirements.txt`) or
`pip install -r requirements-dev.txt`:

```bash
GENDER_MODEL='' RACE_MODEL='' RECOGNITION_MODEL='' \
  .venv/bin/python tools/benchmark.py --confirmed --detector yolo \
  --output outputs/accuracy/confirmed.json
```

Confirmed mode evaluates age, expression, and eyes only. Race and gender are
not inferred or evaluated from faces. This three-image development set cannot
establish general 95% accuracy.

## Baseline at 91aa12f

All three faces detected. Headline age was 67, 6, 4 versus confirmed 77, 5, 3;
within-five-years was 2/3. Fused expression was 3/3. Eye color was
1/3 before iris sampling.

## Current Benchmark on `feat/modular-architecture`

Evaluated with `--confirmed --detector yolo` against `assets/confirmed.md` ground truth:

### Accuracy Summary

| Task | Backend / Model | Baseline (91aa12f) | Current (feat/modular-architecture) | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Face Detection** | YOLO (ONNX) | 3/3 (100.0%) | **3/3 (100.0%)** | Maintained |
| **Emotion** | Fused Ensemble | 3/3 (100.0%) | **3/3 (100.0%)** | Maintained |
| **Eye Color** | Colorimetric (Haar + Iris) | 1/3 (33.3%) | **2/3 (66.7%)** | **+33.4%** |
| **Age** | Best (MiVOLO continuous) | 2/3 within 5yr (MAE 4.0) | **2/3 within 5yr (MAE 4.0)** | Maintained |

### Per-Image Verified Outputs

1. **`joe-biden.jpg`** (ground truth: age 77, happy, blue eyes):
   - Face Detection: 1 detection (box: `[311, 108, 552, 457]`)
   - Headline Age: 67 (MiVOLO); Caffe bucket `(60-100)` contains truth
   - Emotion: `happy` (Fused 3/3 correct across backends)
   - Eye Color: `brown`

2. **`5yr-asian-girl-happy.jpg`** (ground truth: age 5, happy, brown/black eyes):
   - Face Detection: 1 detection (box: `[1034, 352, 1426, 826]`)
   - Headline Age: 6 (MiVOLO, within 1 year); FairFace `3-9` & Caffe `(4-6)` contain truth
   - Emotion: `happy` (**Correct**)
   - Eye Color: `brown` (**Correct**)

3. **`3yr-white-girl-sad.jpg`** (ground truth: age 3, sad, blue eyes):
   - Face Detection: 1 detection (box: `[526, 79, 654, 257]`)
   - Headline Age: 4 (MiVOLO, within 1 year); FairFace `0-2`
   - Emotion: `sad` (**Correct**)
   - Eye Color: `blue` (**Correct**)
