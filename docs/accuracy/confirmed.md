# Confirmed-data evaluation

`assets/confirmed.md` is authoritative. The evaluator does not widen ages or
invent labels. Missing detections, unavailable models, and unknown outputs remain
in denominators. Age buckets get containment counts, not exact-age credit; exact
age, numeric MAE, and within-five-years are separate metrics.

Run:

```bash
GENDER_MODEL='' RACE_MODEL='' RECOGNITION_MODEL='' \
  .venv/bin/python tools/benchmark.py --confirmed --detector yolo \
  --output outputs/accuracy/confirmed.json
```

Confirmed mode evaluates age, expression, hair, and eyes only. Race and gender are
not inferred or evaluated from faces. This three-image development set cannot
establish general 95% accuracy.

## Baseline at 91aa12f

All three faces detected. Headline age was 67, 6, 4 versus confirmed 77, 5, 3;
within-five-years was 2/3. Fused expression was 3/3. Hair was 0/3. Eye color was
1/3 before iris sampling.
