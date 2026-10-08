# Multimodal Face Analyzer

Real-time multi-model face analysis: 16 per-face and whole-frame features, four pluggable face
detectors, and benchmarked fusion across interchangeable backends.

[![CI](https://github.com/dpologdvinity/multimodal-face-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/dpologdvinity/multimodal-face-analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<!-- DEMO: record 5-10s of LIVE mode (age/gender/emotion/landmarks on, change expression once),
     export ≤5 MB, ~800px wide → docs/media/demo.gif; full-UI screenshot → docs/media/screenshot.png -->

A terminal-styled Streamlit app, with upload and live-webcam tabs, that detects faces in an uploaded image or webcam feed and runs any combination
of attribute models on each face: age, gender, race, emotion, gaze, liveness, recognition, and
more. Most features have two or more backends that can run side by side, and the app combines
their answers.

## Highlights

- **Interchangeable face detectors.** SSD, YOLOv8-Face, SCRFD, and RetinaFace share a common call
  shape in `src/face_analyzer/detectors/`, selected per frame by a factory with SSD as the fallback.
- **Fusion across label taxonomies.** Gender and emotion backends are fused, by weighted mean
  of P(Male) and by weighted vote. Emotion labels are first mapped onto shared canonical classes
  (`happy` and `happiness` are one vote).
- **Fusion rejected where the evidence says so.** Age and race name their most reliable model
  instead of fusing, each chosen by held-out evidence: every age fusion rule tried scored at or
  below MiVOLO alone, and on held-out FairFace faces the race blend trailed `fairface` alone.
- **Concurrent, cached inference.** Per-face attribute models run in parallel on a thread pool,
  outputs are memoized on a hash of the exact model input, and LIVE mode shows FPS and
  per-model latency.
- **Build-time model selection, graceful runtime degradation.** Docker build arguments choose
  which weights and dependencies go into the image; at runtime any missing model is skipped and
  listed as unavailable rather than crashing the app.

## Architecture

```mermaid
flowchart LR
    subgraph UI["ui/ (results.py, live.py)"]
        A[Frame] --> C["Colorization (grayscale input only)"]
        K[Face cards and result tables]
    end
    subgraph P["pipeline/: analyze_frame()"]
        B["Global adjustments (optional)"] --> D["Face detector: YOLO / SCRFD / RetinaFace, SSD fallback"]
        D --> E[Face tracking]
        E --> H["Hand landmarks (whole frame)"]
        E --> F["Per face: roll alignment, crop, face adjustments, face landmarks"]
        F --> G["Attribute backends in parallel"]
        G --> I["Fusion and best-model selection"]
        I --> J[Annotated overlay]
        H --> J
    end
    C --> B
    J --> K
```

The source is the installable `face_analyzer` package; paths below are relative to
`src/face_analyzer/`. Colorization runs in the UI layer before `analyze_frame()`: in
`ui/results.py` for uploads and snapshots, and in `ui/live.py` for live webcam frames.
`pipeline/analyzer.py` orchestrates the remaining stages, configured by one `AnalysisConfig`
(`pipeline/config.py`); the stages live in `pipeline/stages.py` (per-feature face tasks in
`pipeline/face_tasks.py`, the prediction cache and thread pool in `pipeline/cache.py`). Model
loading is in `pipeline/loader.py`, fusion in `fusion/`, and vendored third-party architectures
in `nets/`; `inference.py` is a facade that re-exports the public names the UI, tools and
tests use. The Streamlit entry point `app.py` only wires these together: the theme
(`ui/theme.py` plus `theme.css`), the sidebar (`ui/sidebar.py`), the image adjustment sliders
(`ui/adjustments.py`), the per-image results view (`ui/results.py`), and the Live webcam tab
(`ui/live.py`).

## Results

Held-out, on 1,002 test faces from the public [FairFace](https://github.com/joojs/fairface)
validation split (race-stratified sample, seed 0; 95% bootstrap intervals; detection recall 100%):

| Feature | App (shipped) | Best single model |
| ------- | ------------- | ----------------- |
| Gender  | **96.7%** (95.5-97.7) | `mivolo` 96.7% |
| Race, 6 classes | **76.5%** (74.0-79.2) | `fairface` 76.5% (74.0-79.2) |
| Age, 9 buckets | **62.3%** (59.2-65.1) | `mivolo` 62.3% (97.0% within one bucket) |

- Gender and age: the headline numbers are MiVOLO's, and the shipped gender fusion always returns
  MiVOLO's answer. MiVOLO's training data is not published, so overlap with FairFace cannot be
  ruled out.
- Race: the headline names its most reliable model (`fairface`) instead of fusing, because this
  evaluation showed the previous equal-weight fusion trailing `fairface` alone (63.3% against
  76.5%): `deepface`'s overconfident probabilities dominated the blend. `fairface` and `deepface`
  race were both trained on FairFace's training split, so these numbers are in-distribution and
  the ranking may not carry over to other photos.
- Age: FairFace labels age in nine ranges (mostly decades), so this is bucket accuracy, not
  error in years.
- Emotion is not evaluated (FairFace has no emotion labels).

Fusion weights fitted on the other half of the sample match the shipped gender ones; for race,
fitted weights (70.1%) still trailed `fairface` alone. Full tables, label mappings and
caveats: [docs/eval/heldout_fairface.md](docs/eval/heldout_fairface.md). The earlier 75-face,
in-sample benchmark (the set the weights were hand-set on) is kept in
[docs/benchmark.md](docs/benchmark.md).

## Quick start

No git-lfs needed. The model weights (about 2.9 GB for every downloadable model) are fetched on
demand by `tools/fetch_models.py`, which downloads only the models you select and checks each
file's sha256 against [models/manifest.json](models/manifest.json). Permissively licensed
weights come from the project's Hugging Face mirror; the rest come from their original upstream
source, and two (`dan` emotion and `deep3d` 3D reconstruction) are bring-your-own. See
[docs/setup.md](docs/setup.md#model-weights) and [MODEL_LICENSES.md](MODEL_LICENSES.md).

**Upgrading an existing clone?** Pulling this change deletes the git-lfs weights from disk, including
the bring-your-own `dan` and `deep3d` files that cannot be downloaded again. Run
`cp -al models models.bak` before `git pull`; see
[docs/setup.md](docs/setup.md#upgrading-a-clone-that-has-the-git-lfs-weights).

Native, with the guided installer (Debian/Ubuntu), which fetches the weights you pick:

```bash
git clone https://github.com/dpologdvinity/multimodal-face-analyzer.git
cd multimodal-face-analyzer
./install-and-run.sh          # pick backends, then open http://localhost:8501
```

Docker, which downloads the selected weights during the build:

```bash
git clone https://github.com/dpologdvinity/multimodal-face-analyzer.git
cd multimodal-face-analyzer
docker build -t face-analyzer .          # or ./build-and-run.sh for guided model selection
docker run -d -p 127.0.0.1:8501:8501 --name face_analyzer_container face-analyzer
```

The image runs `streamlit run src/face_analyzer/app.py`. For a manual install, add
`pip install --no-deps -e .` after the requirements so the `face_analyzer` package is importable, then
fetch weights with `python tools/fetch_models.py --keys AGE_MODEL=fairface EMOTION_MODEL=hsemotion`
(or `--all`; `--list` shows every file and its source). Build
arguments, manual setup, and the remote-access notes (the app has no authentication) are in
[docs/setup.md](docs/setup.md).

## Features

| Feature | Backends | Notes |
| ------- | -------- | ----- |
| Face detection | `ssd` (required), `yolo`, `scrfd`, `retinaface` | One detector per frame |
| Age | `mivolo`, `fairface`, `dex`, `caffe` | Headline names the most reliable model present |
| Gender | `mivolo`, `fairface`, `caffe`, `deepface` | Fused |
| Race | `fairface`, `deepface` | Headline names the most reliable model present; close runner-up shown |
| Emotion | `hsemotion`, `dan`, `ferplus`, `mini_xception` | Fused vote over canonical labels |
| Gaze and head pose | `mediapipe` | Coarse direction, eye contact, yaw and pitch |
| Liveness | `mediapipe` + texture heuristic | Blink and replay cues; a screening signal only |
| Recognition | `vggface`, `lbph` | Against a gallery you enroll yourself |
| Glasses | `mobilenet` | |
| Mask | `mobilenetv2` | |
| Hair color | `colorimetric` | Heuristic, not a trained model |
| Eye color | `colorimetric` | Heuristic, not a trained model |
| Face landmarks | `mediapipe` | 468-point mesh |
| Hand landmarks | `mediapipe` | Whole frame, up to two hands |
| Colorization | `eccv16` | Whole frame, grayscale input only, before detection |
| Age progression | `franunet` | Non-commercial use only |
| 3D reconstruction | `deep3d` | **No working default weights**: needs the registration-gated Basel Face Model |

The 16 rows after face detection are the analysis features in the `Models` container
(`src/face_analyzer/core/types.py`). Fifteen work with weights `tools/fetch_models.py` downloads; 3D
reconstruction is wired but stays unavailable until you supply its checkpoint and the Basel Face
Model file yourself. The `dan` emotion backend is bring-your-own as well.

The app also has tools that need no extra model: identity search against your own reference
photos, SAVE to a local SQLite database with eigenfaces matching, a scan that labels every face
recognized or unrecognized, an opt-in crowd count, image adjustment sliders, a mouse
crop, per-face image operations, and an opt-in voice and face arousal check in
LIVE mode. Per-model details: [docs/models.md](docs/models.md).

## Responsible use

- Predicted age, gender, race, and emotion are statistical guesses about how a face *appears*
  in one image. They are not a person's identity, and the app cannot know how anyone identifies.
- The categories are coarse and culturally contingent: a fixed list of six or seven race labels,
  binary gender, and a handful of emotion words do not describe real people well.
- The underlying models were trained on datasets (FairFace, AffectNet, IMDB-WIKI, VGGFace and
  others) with known demographic imbalances, so error rates differ across groups.
- Do not use this software to make decisions about individuals, such as hiring, policing,
  access control, credit, or surveillance.
- Run recognition and identity search only on people who have consented to it.

## Licensing

The code is MIT-licensed; see [LICENSE](LICENSE). Vendored architectures in `src/face_analyzer/nets/` keep
their upstream licenses. The model weights are third-party files under their own terms, several
of them non-commercial or with no stated license; see [MODEL_LICENSES.md](MODEL_LICENSES.md).
