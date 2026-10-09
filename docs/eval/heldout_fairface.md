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
in brackets. Run on 2026-10-08 against package commit `e99c6da` (re-scored from the cache, with
no model re-run, after the race and gender headline changes below), SSD detector, two CPU cores
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
| `convnext` (not an app backend) | - | - | 60.0% (56.7-63.1) | 0.44 (0.40-0.47) |
| `dex` | - | - | 39.1% (36.2-41.9) | 0.79 (0.75-0.84) |
| **App (shipped)** | **96.7% (95.5-97.7)** | **76.5% (74.0-79.2)** | **62.3% (59.2-65.1)** | 0.42 (0.38-0.45) |

"App (shipped)" is what the app shows as its headline: the `best` age, gender and race rows.
`select_age` picked `mivolo` for 1,001 of the 1,002 test faces, so the app's age is MiVOLO's;
`select_gender` picked `mivolo` for all 1,002, so the app's gender is MiVOLO's; `select_race`
picked `fairface` for all 1,002, so the app's race is FairFace's. No headline is stacked: a
learned stacker over every backend did not beat these on the test half (see
[Learned stacker](#learned-stacker)).

The gender headline used to be a weighted fusion (`fused`), but it was MiVOLO's answer by
construction: MiVOLO's P(Male) is a hard 0 or 1 carrying weight 3 of the 6 total, so the weighted
mean is at least 0.5 whenever MiVOLO says Male and at most 0.5 whenever it says Female, and the
other three backends can never overrule it (short of an exact 0.5 tie). It now names its most
reliable backend (`select_gender`, ordered `mivolo` > `fairface` > `deepface` > `caffe` by
held-out accuracy, the same order on both halves), which gives the same answer on every test
face (paired difference +0.0 pp, interval +0.0 to +0.0) without calling it a fusion.

`convnext` is a ConvNeXt-Tiny fine-tuned on FairFace's training split for this evaluation's age
buckets; it did not beat `mivolo` (-2.3 pp, -5.6 to +1.2, paired) and is not an app backend. It
was scored afterwards on the same detections and crops (`--extra-age`); see
[age_model.md](age_model.md), which also has an out-of-distribution comparison on UTKFace.

`fairface`'s native seven-class race accuracy (East and Southeast Asian kept apart) is 70.6%
(67.9-73.5). Within one age bucket: `mivolo` 97.0%, `convnext` 96.6%, `fairface` 95.4%, `dex` 84.9%, `caffe` 62.0%.
`dex` reports a spread above 10 years (shown as `uncertain` in the app) on 87% of faces.

### Race by class

Recall per canonical class on the test half:

| Canonical race | n | `fairface` | `deepface` | App (shipped) | Previous fusion |
| --- | --- | --- | --- | --- | --- |
| White | 191 | 74.9% | 68.6% | 74.9% | 68.6% |
| Black | 142 | 89.4% | 81.7% | 89.4% | 81.7% |
| Asian | 271 | 90.0% | 73.4% | 90.0% | 74.2% |
| Indian | 139 | 69.8% | 68.3% | 69.8% | 68.3% |
| Latino | 148 | 56.8% | 42.6% | 56.8% | 43.2% |
| Middle Eastern | 111 | 64.9% | 24.3% | 64.9% | 24.3% |

Latino and Middle Eastern are the weakest classes for both backends, and `deepface` is far weaker
than `fairface` on Middle Eastern.

### Why race is no longer fused

The race headline used to be an equal-weight blend of the two backends' canonical
distributions. `deepface`'s race probabilities are saturated: its top class has a mean
probability of 0.999, and is above 0.9 on 99.6% of faces, against 0.75 and 35% for `fairface`.
The blend was therefore decided by `deepface` whenever they disagreed, so the combined answer
tracked the weaker backend. Re-weighting only partly helps, because a weight scales a near
one-hot distribution without making it express doubt. Following this evaluation the race
headline names its most reliable backend (`select_race`, ordered `fairface` > `deepface`), as
age already did. The previous fusion is reproduced in `tools/eval_heldout.py` from the cached
probabilities so the comparison stays scoreable; the last column is the paired per-face
difference:

| Race headline | Accuracy | Shipped best-model - this |
| ------------- | -------- | ------------------------- |
| Best model (shipped) | 76.5% (74.0-79.2) | - |
| Previous fusion, equal weights | 63.3% (60.5-66.3) | +13.3 pp (+10.2 to +16.4) |
| Fusion, fitted weights | 70.1% (67.4-72.8) | +6.5 pp (+3.8 to +9.3) |

The ordering was chosen from this evaluation's own fit and test halves (both rank `fairface`
first), and both race backends are in-distribution here, so this is evidence for FairFace-like
photos rather than a general guarantee; the in-sample 75-face set ranked `deepface` first.

## Demo configuration

The [public demo](../deploy.md) runs `fairface` alone for age, gender and race, detects faces with
RetinaFace and has no MediaPipe, so it shows `fairface`'s own answer (with one backend there is no
`best` headline row). Same sample, seed and test half as above, `fairface` backend only:

| Configuration | Gender | Race (6 canonical) | Age bucket | Aligned crops (test half) |
| ------------- | ------ | ------------------ | ---------- | ------------------------- |
| Demo before `7bb4639`: RetinaFace, unaligned box crop | 88.1% (86.1-90.0) | 61.1% (58.0-64.2) | 46.3% (43.1-49.5) | 0% |
| Demo at `7bb4639`: RetinaFace, aligned on its landmarks | 92.3% (90.6-94.0) | 75.3% (72.9-78.1) | 59.1% (56.3-62.1) | 99.5% |
| **Demo (shipped): as above, near-profiles on the box crop** | **92.2% (90.5-93.9)** | **76.0% (73.6-78.6)** | **59.4% (56.4-62.4)** | 95.6% |
| Full app: SSD, aligned on MediaPipe landmarks (table above) | 93.1% (91.5-94.6) | 76.5% (74.0-79.2) | 60.3% (57.2-63.3) | 91.2% |

Paired per-face differences on the 1,002 test faces (RetinaFace and SSD both found all of them):
aligning on RetinaFace's landmarks (`7bb4639`) gained +4.2 pp gender (+1.9 to +6.3), +14.3 pp
race (+11.2 to +17.5) and +12.8 pp age (+9.4 to +16.4) over the unaligned demo. Keeping
near-profiles on the box crop (`d838a8c`) changed the input of 39 test faces and moved the
results by -0.1 pp gender (-0.7 to +0.5), +0.7 pp race (+0.0 to +1.4) and +0.3 pp age (-0.6 to
+1.2), so it is not worse. Against the full app's MediaPipe-aligned `fairface`, the shipped demo
is -0.9 pp gender (-2.3 to +0.4), -0.5 pp race (-2.5 to +1.5) and -0.9 pp age (-3.3 to +1.2),
within the noise. The demo still trails the full app's headline gender and age (96.7% and
62.3%), which are MiVOLO's, a model too heavy for the demo host.

RetinaFace predicts the eye centers and nose tip, while FairFace was trained on crops aligned to
dlib's four eye corners and nose. The demo aligns on the eyes and nose only, with each pair of
reference corners merged into its center. It falls back to the box crop when the eyes come out
swapped, or closer than 0.15 of the box width as on near-profiles (the remaining 4.4%). Before
this, a face without MediaPipe landmarks got a 1.5x box crop with only the Haar-cascade roll
leveling, so its position and scale followed the detector's box rather than the face, which is
why the unaligned row is so much lower.
The shipped row is [heldout_fairface_demo.json](heldout_fairface_demo.json); its "App (shipped)"
fields read 0% only because the app picks no headline from a single backend. The unaligned row
was run the same way at `a932ca9`, and the paired differences were computed from the runs'
caches with the tool's seeded bootstrap; the unaligned run's cache and the paired script are not
kept.
Command for the shipped row:

```bash
FACE_ANALYZER_DEMO=1 FACE_ANALYZER_MODEL_DIR=$PWD/models .venv-eval/bin/python tools/eval_heldout.py \
    --n 2000 --detector retinaface --no-mediapipe --cache data/eval_cache/demo-profile015 \
    --output docs/eval/heldout_fairface_demo.json
```

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
- **One detector per run.** The main tables use the SSD detector, at the app's default threshold
  (0.5); the demo configuration uses RetinaFace at the same threshold.

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
   200 images at 1.25 padding, all 200 did. The 0.25 crops would
   therefore measure that detector failure rather than the attribute models.
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
   displayed for every face; the JSON's `cache_vs_display_mismatches` are all zero. The cache was
   recorded before the race and gender headline changes, so its race and gender headlines are
   the previous fusions, which the reproductions match (`previous_fused_race`,
   `previous_fused_gender`); the new headlines are `select_race` and `select_gender` over the
   per-backend labels, which are themselves checked.
5. **Scoring.** Accuracy is over test faces the detector found. A backend that gives no answer for
   a found face counts as wrong. A detection miss is reported through detection recall, not in the
   attribute accuracies. Every figure has a 95% percentile bootstrap interval (1,000 resamples,
   seeded); the fusion and race-headline comparisons are bootstrapped on the paired per-face
   difference.

## Label mappings

**Gender.** Every backend is binary; its displayed label (argmax) is compared with FairFace's
`Male`/`Female`. The app's gender headline is the label of the backend `select_gender` picks.

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
class only. The app's race headline is the label of the backend `select_race` picks, scored the
same way.

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

The gender weights and the equal race weights the app used before the gender and race headline
changes were hand-set on the 75-face set. To test them, weights were
fitted on the fit half and both sets were scored on the test half. The fitting rule is the
weighted-majority-vote optimum for independent voters with symmetric errors (Nitzan & Paroush,
1982): each backend gets `log((K - 1) * acc / (1 - acc))`, where `acc` is its fit-half top-1
accuracy and `K` the number of classes (2 for gender, 6 for race), clipped at 0. It uses one
number per backend, so it has little room to overfit the fit half. The weights then go through
the reproduced previous gender and race fusions unchanged.

Fit-half top-1 accuracy and the resulting weights, next to the previous ones:

| Backend | Fit-half accuracy | Previous weight | Fitted weight |
| ------- | ----------------- | -------------- | ------------- |
| gender `mivolo` | 97.0% | 3.0 | 3.47 |
| gender `fairface` | 93.6% | 2.0 | 2.68 |
| gender `deepface` | 77.5% | 0.5 | 1.24 |
| gender `caffe` | 71.8% | 0.5 | 0.93 |
| race `fairface` | 73.0% | 1.0 | 2.60 |
| race `deepface` | 63.6% | 1.0 | 2.17 |

Test-half accuracy of each fusion under each weight set; the last column is the paired per-face
difference:

| Fusion | Previous weights | Fitted weights | Difference |
| ------ | ---------------- | -------------- | ---------- |
| gender | 96.7% (95.5-97.7) | 96.3% (95.1-97.4) | -0.4 pp (-0.9 to +0.0) |
| race | 63.3% (60.5-66.3) | 70.1% (67.4-72.8) | +6.8 pp (+5.1 to +8.5) |

- **Gender (fusion dropped: the headline now names `mivolo`):** the fitted weights are no better
  than the previous ones (-0.4 points, interval -0.9 to 0.0), and the previous ones reproduce
  MiVOLO alone by construction, so the fusion was replaced by `select_gender`, which gives the
  same answers under an honest name.
- **Race (applied: the headline now names `fairface`):** the fitted weights beat the previous
  equal weights by 6.8 points (5.1 to 8.5), but `fairface` alone (76.5%) beats both, by 6.5
  points (3.8 to 9.3) over the fitted weights. Because `deepface`'s probabilities are saturated,
  the blend only defers to `fairface` once `deepface`'s weight is close to zero, which amounts to
  using `fairface` alone. Calibrating `deepface` (for example temperature scaling fitted on the
  fit half) before blending is the untried alternative. Both FairFace-trained backends are
  in-distribution here, so this ranking may not carry over to other photos; the 75-face set
  (in-sample) ranked `deepface` first.

## Learned stacker

A learned stacker was tried as the last way to beat the best single backend: one L2-regularised
logistic regression per attribute, fitted by `tools/fit_stacker.py` on the cached outputs of the
same fit half and scored once on the same test half, with the same seeded bootstrap. The rule
set in advance: ship an attribute's stacker only if the paired-bootstrap 95% interval of
(stacked - shipped) on the test half lies above zero. **No attribute passed, so none ships**;
the fitted models are kept as plain JSON in [stacker_fairface.json](stacker_fairface.json).

**Features** (every backend output for the attribute, standardised on the fit half; probabilities
clipped at 1e-6 before taking logs):

- Gender: the log-odds of P(Male) from `fairface`, `caffe` and `deepface`, and MiVOLO's hard
  label as 0/1 (MiVOLO exposes no probability).
- Race: the log-probability of each of the six canonical classes from `fairface` (East and
  Southeast Asian summed) and from `deepface` (12 features).
- Age bucket: `fairface`'s nine and `caffe`'s eight log-probabilities; for `mivolo` and `dex`, the
  age in years, its log1p and a one-hot of the FairFace bucket the displayed age falls in; and
  `dex`'s spread (40 features).

Cross-attribute signals were tried ad hoc, in 5-fold CV on the fit half only (MiVOLO's age
and `fairface`'s race probabilities for gender, MiVOLO's age and `fairface`'s P(Male) for race,
`fairface`'s P(Male) for age), and did not raise CV accuracy, so they are not used. That check
is not part of `tools/fit_stacker.py` and is not reproduced by it. **C** (inverse L2 strength) was chosen from 13 values between 0.001 and 1000 by stratified
5-fold CV inside the fit half (995 detected faces), ties going to the stronger penalty.

| Attribute | C | Fit-half CV | Stacked (test) | Shipped (test) | Stacked - shipped | Stacked - best single | Adopted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gender | 0.0316 | 97.1% | 96.7% (95.5-97.7) | 96.7% (95.5-97.7) | +0.0 pp (+0.0 to +0.0) | +0.0 pp (+0.0 to +0.0) (`mivolo`) | no |
| race | 0.316 | 74.3% | 75.1% (72.5-77.8) | 76.5% (74.0-79.2) | -1.4 pp (-3.4 to +0.7) | -1.4 pp (-3.4 to +0.7) (`fairface`) | no |
| age | 0.0316 | 60.5% | 62.8% (59.7-65.7) | 62.3% (59.2-65.1) | +0.5 pp (-1.8 to +2.8) | +0.5 pp (-1.8 to +2.8) (`mivolo`) | no |

The best single backend (chosen on the fit half) is the shipped headline for all three, so both
comparisons coincide. Within one age bucket the stacker scores 96.5% (95.2-97.5) against 97.0%
(95.9-98.0) for the shipped `mivolo` answer.

- **Gender:** the stacker learned to follow MiVOLO and gave its answer on every test face.
- **Race:** it disagreed with `fairface` on 128 test faces and was right on 42 of them against
  56 for `fairface`; `deepface`'s saturated probabilities again add more noise than signal.
- **Age:** it disagreed with `mivolo` on 166 faces (right on 74 against 69), a gain well inside
  the noise, and slightly worse within one bucket.

**Headroom.** A face where at least one backend's top-1 answer is right bounds what any rule
that selects among these backends could reach: 98.6% for gender, 83.9% for race and 86.8% for
age (test half), against 96.7%, 76.5% and 62.3% shipped. The bound is loose for age: with four
backends spread over neighbouring buckets, one of them often lands in the right bucket, and
nothing in their outputs tells the stacker which one.

**Caveat.** The stacker is fitted on FairFace data, so its FairFace numbers are in-distribution
for it (on top of `fairface` and `deepface` race already being in-distribution), and it may not
transfer to other photo sources. A stacker that cleared the bar here would still need checking on
another dataset before it could be called better in general.

## Reproduce

```bash
uv venv --python 3.11 .venv-eval   # then install the pins listed in docs/setup.md
.venv-eval/bin/python tools/fetch_fairface.py
FACE_ANALYZER_MODEL_DIR=$PWD/models .venv-eval/bin/python tools/eval_heldout.py --n 2000
uv pip install --python .venv-eval/bin/python scikit-learn   # the eval venv only
.venv-eval/bin/python tools/fit_stacker.py   # learned stacker, from the cache only
```

The second command resumes from its cache if interrupted; `--score-only` re-scores the cache
without loading any model.
