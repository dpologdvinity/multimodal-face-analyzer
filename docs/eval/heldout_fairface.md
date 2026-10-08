# Held-out evaluation: FairFace validation

Age, gender and race accuracy of every backend, and of the app's combined answers, on a public,
labelled dataset that none of the app's settings were tuned on. The table numbers come from
[heldout_fairface.json](heldout_fairface.json), together with the sample, seed, dataset revision,
package versions and date; the diagnostic figures (latencies, saturation statistics, the
200-crop padding check) were measured separately during the run and are not in the JSON.
The older 75-face benchmark is in [../benchmark.md](../benchmark.md).

## Results

2,000 FairFace validation images (seed 0), stratified by race and split into a 998-image fit
half and a 1,002-image test half. Every figure is on the test half, with a 95% bootstrap interval
in brackets. Run on 2026-10-08 against package commit `e99c6da`, SSD detector, two CPU cores
(4.7 s per image with every age, gender, race and emotion backend active, 156 minutes in total;
`mivolo` alone takes a median 2.7 s per face, and the two `deepface` models about 1.7 s each).

**Detection recall:** 1,002 of 1,002 test images (100%); 1,997 of 2,000 over the whole sample
(99.85%). Accuracies below are over detected faces.

| Backend | Gender | Race (6 canonical) | Age bucket | Mean bucket offset |
| ------- | ------ | ------------------ | ---------- | ------------------ |
| `mivolo` | 96.7% (95.5-97.7) | - | 62.3% (59.2-65.1) | 0.42 (0.38-0.45) |
| `fairface` | 93.1% (91.5-94.6) | 76.5% (74.0-79.2) | 60.3% (57.2-63.3) | 0.47 (0.42-0.51) |
| `caffe` | 73.5% (70.7-76.1) | - | 26.8% (23.8-29.5) | 1.43 (1.35-1.51) |
| `deepface` | 77.5% (74.8-80.0) | 63.0% (60.1-66.0) | - | - |
| `dex` | - | - | 39.1% (36.2-41.9) | 0.79 (0.75-0.84) |
| **App (shipped)** | **96.7% (95.5-97.7)** | **63.3% (60.5-66.3)** | **62.3% (59.2-65.1)** | 0.42 (0.38-0.45) |

"App (shipped)" is what the app shows as its headline: the `fused` gender and race rows, and the
`best` age row. `select_age` picked `mivolo` for 1,001 of the 1,002 test faces, so the app's age
is MiVOLO's. The shipped gender fusion is MiVOLO's answer by construction: MiVOLO's P(Male) is a
hard 0 or 1 carrying weight 3 of the 6 total, so the weighted mean is at least 0.5 whenever MiVOLO
says Male and at most 0.5 whenever it says Female, and the other three backends can never
overrule it (short of an exact 0.5 tie).

`fairface`'s native seven-class race accuracy (East and Southeast Asian kept apart) is 70.6%
(67.9-73.5). Within one age bucket: `mivolo` 97.0%, `fairface` 95.4%, `dex` 84.9%, `caffe` 62.0%.
`dex` reports a spread above 10 years (shown as `uncertain` in the app) on 87% of faces.

### Race by class

Recall per canonical class on the test half:

| Canonical race | n | `fairface` | `deepface` | App (shipped) |
| --- | --- | --- | --- | --- |
| White | 191 | 74.9% | 68.6% | 68.6% |
| Black | 142 | 89.4% | 81.7% | 81.7% |
| Asian | 271 | 90.0% | 73.4% | 74.2% |
| Indian | 139 | 69.8% | 68.3% | 68.3% |
| Latino | 148 | 56.8% | 42.6% | 43.2% |
| Middle Eastern | 111 | 64.9% | 24.3% | 24.3% |

Latino and Middle Eastern are the weakest classes for both backends, and `deepface` is far weaker
than `fairface` on Middle Eastern.

### Why the race fusion is worse than `fairface` alone

`deepface`'s race probabilities are saturated: its top class has a mean probability of 0.999,
and is above 0.9 on 99.6% of faces, against 0.75 and 35% for `fairface`. An equal-weight blend of
the two distributions is therefore decided by `deepface` whenever they disagree, so the combined
answer tracks the weaker backend (63.3%, against 76.5% for `fairface` alone). Re-weighting only
partly helps, because a weight scales a near one-hot distribution without making it express doubt.

## What these numbers can and cannot show

- **`fairface` is in-distribution here, and so is `deepface` race.** The `fairface` backend was
  trained by the FairFace authors on the FairFace training split. The validation images are
  held out from that training, but they come from the same collection, label protocol and crop
  style, so its numbers favour it. DeepFace's race model is also trained on FairFace: its author
  [describes](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) training on
  the FairFace training split and using the validation split as its test set; its six classes
  are FairFace's seven with East and Southeast Asian merged.
- **The other backends were trained on other data.** `caffe` age and gender (Levi & Hassner) on
  Adience, `dex` on IMDB-WIKI, `deepface` gender on the Wikipedia part of IMDB-WIKI
  ([author's post](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/)).
  `mivolo`'s [model card](https://huggingface.co/iitolstykh/mivolo_v2) says only "proprietary
  and open-source datasets", so FairFace overlap cannot be ruled out for it.
- **Ages are buckets, not years.** FairFace labels age in nine ranges, so there is no age MAE in
  years here. Bucket accuracy also penalises a continuous model for an answer one year across a
  boundary (29 for a "30-39" face) exactly as hard as a decade off, which is why the mean bucket
  offset and the within-one-bucket rate are reported too.
- **FairFace labels are perceived attributes.** Gender, race and age were annotated by crowd
  workers from the image. Race in particular is a perceived, culturally loaded category, not a
  ground truth about a person.
- **Emotion is not evaluated.** FairFace has no emotion labels. The emotion backends still run
  (so the timing matches the app), and their outputs are cached but not scored.
- **One detector.** Only the SSD detector is used, at the app's default threshold (0.5).

## Method

1. **Data.** `tools/fetch_fairface.py` downloads the FairFace validation split (10,954 images,
   CC BY 4.0) from the Hugging Face mirror
   [`HuggingFaceM4/FairFace`](https://huggingface.co/datasets/HuggingFaceM4/FairFace) at a
   pinned revision, without a login, into the untracked `data/fairface/`.
2. **Padding.** The eval uses the 1.25-padding images (448x448, the face with its surroundings),
   not the tighter 0.25-padding crops (224x224). On the 0.25 crops the face fills the frame, and
   OpenCV's SSD detector then returns boxes that lie mostly or wholly outside the image (normalized
   coordinates above 1). In a one-off check on 200 random 0.25 crops (not part of the eval tool),
   it returned at least one box for all 200, but only 102 had a box inside the frame; on the same
   200 images at 1.25 padding, all 200 did. The 0.25 crops would therefore measure that detector failure rather than the attribute models.
   This is a real limitation of the app on very tight close-ups (it does not reject out-of-frame
   SSD boxes), noted here rather than fixed in this change.
3. **Sample.** 2,000 images drawn with seed 0, stratified by race with proportional allocation,
   then split 50/50 into a **fit** half and a **test** half, again stratified by race. All
   reported accuracies are on the test half. `--n` scales the sample; a smaller sample with the
   same seed is a subset of a larger one.
4. **Inference.** `tools/eval_heldout.py` runs the real `analyze_frame()` on each image with every
   loaded age, gender, race and emotion backend active and the SSD detector, exactly as the app
   would (roll alignment, MediaPipe landmarks for FairFace alignment, per-backend crops). Two
   pieces of instrumentation are added around it, neither of which changes any per-face output
   (keeping only the largest in-frame box does decide which face is analysed; FairFace images are
   single-subject crops): the detector's result is reduced to the box with the largest in-frame
   area, and the memoization helper the per-face tasks call is wrapped to record
   each backend's raw output (probabilities where the backend exposes them). Those are cached in
   `data/eval_cache/<config>/predictions.jsonl`, so re-scoring never re-runs a model. As a check,
   the labels and combined answers rebuilt from the cache are compared with what `analyze_frame()`
   displayed for every face; the JSON's `cache_vs_display_mismatches` are all zero.
5. **Scoring.** Accuracy is over test faces the detector found. A backend that gives no answer for
   a found face counts as wrong. A detection miss is reported through detection recall, not in the
   attribute accuracies. Every figure has a 95% percentile bootstrap interval (1,000 resamples,
   seeded); the fitted-versus-shipped comparison is bootstrapped on the paired per-face difference.

## Label mappings

**Gender.** Every backend is binary; its displayed label (argmax) is compared with FairFace's
`Male`/`Female`.

**Race.** Everything is scored on the app's six canonical classes (`RACE_LABEL_TO_CANONICAL` in
`core/constants.py`): White, Black, Asian, Indian, Latino, Middle Eastern. FairFace's East Asian
and Southeast Asian labels both become Asian, in the ground truth and in the `fairface` backend's
output. DeepFace's six classes map one-to-one onto those canonical classes:

| DeepFace | FairFace label(s) | Canonical |
| -------- | ----------------- | --------- |
| `asian` | East Asian, Southeast Asian | Asian |
| `indian` | Indian | Indian |
| `black` | Black | Black |
| `white` | White | White |
| `middle eastern` | Middle Eastern | Middle Eastern |
| `latino hispanic` | Latino_Hispanic | Latino |

So DeepFace can never be right about East versus Southeast Asian, and the merged Asian class
gives both backends an easier target than FairFace's own seven-class task. `fairface`'s native
seven-class accuracy is reported separately. A backend's answer is its top-1 class, the first
class the app displays; the close-runner-up form `White (52%)/Black (47%)` is scored on its first
class only. The combined answer is the shipped `fuse_race` blend, scored the same way.

**Age.** FairFace's buckets are 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69 and 70+.

- `fairface` predicts these buckets directly (argmax).
- `mivolo` and `dex` give a continuous age. It is rounded to the integer the app displays and
  placed in the bucket that contains it. `dex` is scored on its mean even when the app shows it
  as `uncertain` (spread above 10 years); the share of such faces is in the JSON.
- `caffe` predicts Adience's eight ranges. Each is mapped through its midpoint, the same value
  `select_age` uses: (0-2) to 0-2, (4-6) to 3-9, (8-12) and (15-20) to 10-19, (25-32) to 20-29,
  (38-43) to 40-49, (48-53) to 50-59 and (60-100) to 70+. Caffe can therefore never answer 30-39
  or 60-69, a structural handicap of its label set rather than of this mapping.
- The app's headline age (`select_age`, the most reliable backend present) is bucketed the same
  way as the backend it picks.

The mean bucket offset is the mean of |predicted bucket index - true bucket index| over faces the
backend answered.

## Fusion weights

The shipped weights in `core/constants.py` were hand-set on the 75-face set. To test them, weights
were fitted on the fit half and both sets were scored on the test half. The fitting rule is the
weighted-majority-vote optimum for independent voters with symmetric errors (Nitzan & Paroush,
1982): each backend gets `log((K - 1) * acc / (1 - acc))`, where `acc` is its fit-half top-1
accuracy and `K` the number of classes (2 for gender, 6 for race), clipped at 0. It uses one
number per backend, so it has little room to overfit the fit half. The weights then go through
the shipped `fuse_gender`/`fuse_race` unchanged.

Fit-half top-1 accuracy and the resulting weights, next to the shipped ones:

| Backend | Fit-half accuracy | Shipped weight | Fitted weight |
| ------- | ----------------- | -------------- | ------------- |
| gender `mivolo` | 97.0% | 3.0 | 3.47 |
| gender `fairface` | 93.6% | 2.0 | 2.68 |
| gender `deepface` | 77.5% | 0.5 | 1.24 |
| gender `caffe` | 71.8% | 0.5 | 0.93 |
| race `fairface` | 73.0% | 1.0 | 2.60 |
| race `deepface` | 63.6% | 1.0 | 2.17 |

Test-half accuracy of the shipped `fuse_gender`/`fuse_race` under each weight set; the last column
is the paired per-face difference:

| Fusion | Shipped weights | Fitted weights | Fitted - shipped |
| ------ | --------------- | -------------- | ---------------- |
| gender | 96.7% (95.5-97.7) | 96.3% (95.1-97.4) | -0.4 pp (-0.9 to +0.0) |
| race | 63.3% (60.5-66.3) | 70.1% (67.4-72.8) | +6.8 pp (+5.1 to +8.5) |

**Recommendation (not applied; the shipped weights are unchanged).**

- **Gender:** keep the shipped weights, or drop the fusion: the fitted ones are no better (-0.4
  points, interval -0.9 to 0.0), and the shipped ones reproduce MiVOLO alone by construction.
- **Race:** the fitted weights beat the shipped equal weights by 6.8 points (5.1 to 8.5), but
  `fairface` alone (76.5%) beats both. Because `deepface`'s probabilities are saturated, the blend
  only defers to `fairface` once `deepface`'s weight is close to zero, which amounts to using
  `fairface` alone. The options are to lead the race headline with `fairface` alone, or to
  calibrate `deepface` (for example temperature scaling fitted on the fit half) before blending. Both FairFace-trained backends are in-distribution here, so this
  ranking may not carry over to other photos; the 75-face set (in-sample) ranked `deepface`
  first.

## Reproduce

```bash
uv venv --python 3.11 .venv-eval   # then install the pins listed in docs/setup.md
.venv-eval/bin/python tools/fetch_fairface.py
FACE_ANALYZER_MODEL_DIR=$PWD/models .venv-eval/bin/python tools/eval_heldout.py --n 2000
```

The second command resumes from its cache if interrupted; `--score-only` re-scores the cache
without loading any model.
