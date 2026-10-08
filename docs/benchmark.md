# Benchmark

How the per-model and combined accuracy figures were measured, what they do and do not show,
and how the combined answers are formed. Per-model details are in [models.md](models.md).

There are two evaluations. The **held-out FairFace evaluation** below is the one to quote: 1,002
public, labelled test faces that none of the settings were tuned on. The older **75-face set**
is in-sample (the fusion weights were chosen on it) and is kept as a wiring check.

## Held-out evaluation (FairFace validation)

`tools/eval_heldout.py` runs the real `analyze_frame()` over 2,000 race-stratified images from
the FairFace validation split (seed 0), fits fusion weights on one half and scores everything on
the other. Test half, 95% bootstrap intervals in brackets; detection recall 100% on the test
half (1,997 of 2,000 overall):

| Feature | App (shipped) | Best single model | Weakest model |
| ------- | ------------- | ----------------- | ------------- |
| Gender | **96.7%** (95.5-97.7) | `mivolo` 96.7% (95.5-97.7) | `caffe` 73.5% (70.7-76.1) |
| Race (6 classes) | **63.3%** (60.5-66.3) | `fairface` 76.5% (74.0-79.2) | `deepface` 63.0% (60.1-66.0) |
| Age bucket (9) | **62.3%** (59.2-65.1) | `mivolo` 62.3% (59.2-65.1) | `caffe` 26.8% (23.8-29.5) |

Main caveats: `fairface` and `deepface` race were both trained on FairFace's training split, so
they are in-distribution here; ages are scored as FairFace's nine buckets, not years; emotion is
not labelled in FairFace and is not evaluated. The combined race answer is worse than `fairface`
alone because `deepface`'s near one-hot probabilities dominate the blend; weights fitted on the
fit half recover 6.8 points but still trail `fairface` alone. Full tables, mappings, the fitted
weights and every caveat: [eval/heldout_fairface.md](eval/heldout_fairface.md); raw numbers:
[eval/heldout_fairface.json](eval/heldout_fairface.json).

## In-sample 75-face set

### Method

`tools/benchmark.py` runs the real `analyze_frame()` pipeline over a set of images and scores
every loaded age, gender, race, and emotion backend against `tools/ground_truth.json`. Each
hand-labelled face is bound to its label by its normalized face centre, so the labels survive
detector changes. `tools/dump_predictions.py` and `tools/score_fusion.py` re-score the combined
answers from saved per-model outputs, using the shipped fusion functions rather than a copy.

The labelled set:

- 75 faces from 11 images, 51 labelled female and 24 male.
- Race and emotion labels are lists of acceptable answers, because some faces are ambiguous.
  Race uses six canonical classes: white, black, asian, indian, latino, middle_eastern.
- Age labels are inclusive `[min, max]` year ranges. A prediction counts as correct when its
  range (or its single value) overlaps the labelled range. Only three faces have externally
  confirmed ages; the other ranges are hand estimates, so age accuracy is measured against
  judgement, not documented fact.
- Emotion is labelled on 42 of the 75 faces, and 38 of those 42 are labelled `happy` only.
  The emotion figures therefore mostly measure smile detection.
- The 11 images are third-party photos and are not distributed with the repository, so these
  figures cannot be reproduced from a clone.

### Results

Detection recall was 100% on all 75 faces.

| Feature | Combined | Best single model | Weakest active model |
| ------- | -------- | ----------------- | -------------------- |
| Gender  | **100%** | `mivolo` 100% | `caffe` / `deepface` 86.7% |
| Emotion | **100%** | `dan` / `hsemotion` / `ferplus` 100% | `mini_xception` 95.2% |
| Race    | **96.0%** | `deepface` 94.7% | `fairface` 93.3% |
| Age     | **93.3%** (= `mivolo`) | `mivolo` 93.3% | `caffe` 62.7% |

Race is scored strictly: only the top class counts, even though the UI also shows a close
runner-up (`White (52%)/Black (47%)`) when two classes are within 10 percentage points.
Counting either shown class as correct would read 97.3% combined and 96.0% for `fairface`, but
only 2 of the 75 combined answers show a runner-up at all, so the strict number is the one
reported.

These figures were recorded at commit `91aa12f`. Two preprocessing changes landed after it
(FairFace five-point alignment in `fed152b` and the DEX crop geometry in `0622b46`) and have not
been re-scored on this set.

### Limits

- **In-sample, no held-out split.** The fusion weights below were chosen by looking at results on these
  same 75 faces, so the combined figures are in-sample and optimistic.
- **Small corpus.** One or two faces is about 1.3 percentage points, which is noise at this size.
- **Skewed labels.** Emotion is almost all `happy`; gender is two-thirds female.

Treat these numbers as a sanity check of the wiring and the fusion rules, not as a measure of
accuracy in general use; the held-out evaluation above is that measure. Several backends score
far lower there (for example `deepface` race, 94.7% here and 63.0% on FairFace).

## Combined answers

With more than one model active for a feature, the per-face card leads with a single combined
answer and lists every individual model underneath. Gender, race, and emotion are fused; age
instead names its best available model.

| Feature | Combined row | How |
| ------- | ------------ | --- |
| Gender  | `fused` | Weighted mean of each model's P(Male) |
| Race    | `fused` | Weighted blend of class probabilities, re-expressed over shared canonical classes (FairFace's East and Southeast Asian collapse into one `Asian` class, since the two backends' label sets differ) |
| Emotion | `fused` | Weighted vote over canonical emotion names (`happy` and `happiness` are one vote, not two) |
| Age     | `best (<model>)` | The most reliable backend present: `mivolo` > `fairface` > `dex` > `caffe` |

The weights live in `src/face_analyzer/core/constants.py`:

| Feature | Weights |
| ------- | ------- |
| Gender  | `mivolo` 3.0, `fairface` 2.0, `caffe` 0.5, `deepface` 0.5 |
| Emotion | `dan` 3.0, `hsemotion` 3.0, `ferplus` 2.0, `mini_xception` 1.0 |
| Race    | `fairface` 1.0, `deepface` 1.0 |
| Age order | `mivolo`, `fairface`, `dex`, `caffe` (`AGE_MODEL_RELIABILITY`) |

The weights are hand-set, not fitted. They are ordered by each model's accuracy on the 75-face
set, so a weaker backend contributes less to the combined answer; race uses equal weights. They
were chosen on the same set they are scored on, with no held-out split, which is why the
75-face combined figures are in-sample. The held-out evaluation tests them against weights fitted
on a separate half; see [eval/heldout_fairface.md](eval/heldout_fairface.md#fusion-weights) for
that comparison and a recommendation (the shipped weights are unchanged). Re-run `tools/benchmark.py` after changing any model or
its preprocessing, and revisit the weights if the ordering moves.

### Why age is not fused

Every fusion rule tried for age scored at or below MiVOLO alone: weighted median and weighted
mean across a range of weights, clipping MiVOLO into FairFace's predicted decade, and overriding
MiVOLO only when both other backends disagreed with it. The age backends fail on the same faces
(elderly faces read young in all of them), so averaging moves the answer without correcting it.
The headline therefore names the most reliable model present instead of blending.

## Confirmed-age development set

A separate three-image set with externally confirmed ages is used for preprocessing changes;
see [accuracy/confirmed.md](accuracy/confirmed.md) and
[accuracy/color-sampling.md](accuracy/color-sampling.md). Three faces cannot establish general
accuracy either.

## Runtime performance

- **Threaded per-face inference.** Within each face, age, gender, emotion, race, gaze, head
  pose, recognition, glasses, mask, hair color, eye color, and liveness run concurrently on a
  shared thread pool (`src/face_analyzer/pipeline/analyzer.py`).
- **Prediction cache.** Most per-face classifier calls, plus face detection, roll estimation,
  and hand landmarks, are memoized on a hash of the exact input bytes fed to that model. Identical
  input bytes always produce identical deterministic output, so the cache carries no accuracy
  risk; it mainly helps Streamlit, which reruns the whole script on any widget change. FairFace
  (keyed on the full frame and box rather than an isolated crop) and the identity match step
  (the gallery can change between calls) are not cached. The cache is LRU-bounded at 2048
  entries and shared process-wide.
- **Classifier frame skip (webcam LIVE mode).** A `Classifier frame skip` slider (1-10, default
  1) runs the per-face classifiers every Nth frame. Face detection and the landmark overlays
  still run every frame, so the video stays smooth. Classifier outputs are not drawn onto the
  LIVE video, so skipping only reduces CPU load.
- **Live metrics.** LIVE mode reports recent FPS and average latency for each active
  feature/model pair.
