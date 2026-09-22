# MULTIMODAL FACE ANALYZER

![Python](https://img.shields.io/badge/Python-3.11-00ff66?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-00ff66?style=flat-square&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-00ff66?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-00ff66?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-00ff66?style=flat-square&logo=docker&logoColor=white)

> Computer vision pipeline for face detection with age, gender, race, emotion, gaze, glasses, mask, and hair/eye color inference. A containerized Streamlit web app.

---

## Key Features

- **Multi-model face analysis:** face detection, age, gender, race, emotion, gaze, glasses, face mask, colorimetric hair/eye color, face landmarks, plus whole-frame auto-colorization and hand landmarks -- most features have 2+ selectable model backends.
- **Image adjustments:** 11 Lightroom-style sliders (exposure, contrast, shadows/highlights, saturation/vibrance, sharpness, noise reduction, etc.) for the whole image, all face crops before classification, and each detected face's preview separately.
- **Face details on hover:** Hover or focus a detected face box in the annotated image to see its analysis results; full face cards remain below the image.
- **Identity search:** SEARCH button per detected face, matching against bundled reference photos (`known_people/`, a few famous people out of the box) plus an optional user-specified directory. Local matching only, no live internet search.
- **Save & eigenfaces:** SAVE button per detected face writes to a sparse-column SQLite database plus `faces/`/`eigen/`; SEARCH also checks the eigenfaces (PCA) algorithm against every previously-saved face. A whole-image SCAN ALL FACES button labels every detected face Recognized/Unrecognized in one pass.
- **3D reconstruction:** 3D RECON button per detected face (Deep3DFaceRecon_pytorch: ResNet50 + Basel Face Model), downloads a `.obj` mesh. Ships no working weights out of the box -- both the checkpoint and the Basel Face Model data are gated (Google Drive / university license registration); see README.
- **YOLO / SCRFD / RetinaFace face detectors:** selectable per-frame via a sidebar dropdown; YOLO is the default when loaded, with the required SSD/ResNet-10 detector as the fallback.
- **LBPH recognition:** `cv2.face.LBPHFaceRecognizer`-based alternative to VGGFace, trains from scratch on your own enrolled photos -- no pretrained weights to source. The trained recognizer is cached by the enrolled gallery's file fingerprint and retrains only after enrollment or gallery-file changes.
- **Age progression/regression:** AGE PROGRESSION button per detected face (U-Net re-aging network, [timroelofs123/face_reaging](https://github.com/timroelofs123/face_reaging)), outputs a before/after image pair for a chosen source/target age. **Non-commercial use only** -- see below.
- **Docker-packaged Streamlit app:** `src/app.py`, all features including race and gaze.
- **Build-time feature toggles:** disable any model at Docker build time to shrink the image (see [Docker](#docker-web-app)).
- **Graceful degradation:** any model missing at runtime (file or dependency not present) is skipped, not a crash -- the rest of the pipeline keeps working.
- **Input validation:** empty or invalid image uploads show a clear error, and LBPH enrollment names are restricted to safe single directory names.
- **Cyberpunk web interface:** terminal-styled drag-and-drop Streamlit UI, plus snapshot/live webcam tabs.
- **Live performance metrics:** webcam LIVE mode reports recent FPS and average latency for each active feature/model pair.
- **Liveness / anti-spoofing:** webcam LIVE mode tracks blink transitions per face and all modes score regular high-frequency texture as a replay/screen cue. Results show `LIVE`, `SUSPECTED SPOOF`, or `INCONCLUSIVE`.
- **Crowd count / demographics:** opt-in (off by default), whole-image age/gender/race breakdown aggregated from per-face results, no new model.
- **Performance controls:** LIVE-mode classifier frame skip, plus content-hash caching of per-face classifier outputs (see [Performance](#performance)).

---

## Models

Face detection is required; age, gender, race, and emotion are each independently optional -- if a model's file(s) or dependencies aren't present, that feature is skipped and the rest still runs.

### Face Detection

| Backend         | Framework            | Output                  |
| --------------- | -------------------- | ----------------------- |
| SSD / ResNet-10 | TensorFlow (cv2.dnn) | bounding box (required) |
| `yolo` (web app only) | ONNX (onnxruntime) | bounding box            |
| `scrfd` (web app only) | ONNX (onnxruntime) | bounding box            |
| `retinaface` (web app only) | ONNX (onnxruntime) | bounding box            |

SSD/ResNet-10 is the original detector and is always required as the fallback. `yolo` (YOLOv8-Face, `models/yolov8n_face.onnx`, [yakhyo/yolov8-face-onnx-inference](https://github.com/yakhyo/yolov8-face-onnx-inference), no explicit upstream license -- same treatment as DAN), `scrfd` (SCRFD, `models/scrfd_2.5g_bnkps.onnx`, [deepinsight/insightface](https://github.com/deepinsight/insightface/tree/master/detection/scrfd), 2.5GF `bnkps` checkpoint, **non-commercial research-only weights**), and `retinaface` (RetinaFace, `models/retinaface_mobilenet0.25.onnx`, [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)'s mobilenet0.25 backbone -- MIT-licensed, re-exported by [AMD's Ryzen AI model zoo](https://huggingface.co/amd/retinaface) under Apache 2.0, the only unambiguously permissive face-detector option in this repo) are selectable in the web app, with YOLO first whenever its model is loaded. Exactly one detector runs per frame; running two and merging their boxes would just produce duplicate/overlapping faces, not a meaningfully combined result. All three verified with a real photo (`known_people/Barack_Obama.jpg`): correctly detect and localize the face.

**All three need `onnxruntime`, not this repo's usual `cv2.dnn` ONNX path.** For `yolo`, verified directly: this specific ONNX export fails to load under `cv2.dnn` on both OpenCV 4.10 and 5.0 (`Mixed input data types` error in its DFL box-decode subgraph -- an ONNX importer limitation, not a version-pin issue). The decode math (DFL softmax + sigmoid + NMS) is otherwise a faithful port of upstream's own `models/yolov8.py`, using `cv2.dnn.NMSBoxes` in place of their `torchvision.ops.nms` to avoid pulling in `torchvision` just for this. `scrfd` is run through onnxruntime too, for one consistent non-cv2.dnn detector code path rather than mixing conventions -- its own multi-output (score/bbox/kps per stride) anchor format is decoded via straightforward distance-to-bbox regression (no DFL needed, this checkpoint regresses distances directly), matching upstream's own `tools/scrfd.py`. `retinaface` likewise -- unlike `yolo`/`scrfd`'s dynamic square input, this checkpoint takes a fixed 608x640 NHWC input; boxes are decoded against precomputed anchor priors using the same variance-scaled regression as upstream's own `utils/box_utils.py`.

### Age

| Backend       | Framework         | Output                         |
| ------------- | ----------------- | ------------------------------ |
| `fairface`    | ONNX (cv2.dnn)    | bucketed range, e.g. `20-29` (9 buckets) |
| `caffe`       | Caffe (cv2.dnn)   | bucketed range, e.g. `(25-32)` |
| `dex`         | Caffe (cv2.dnn)   | continuous age, e.g. `31` (expected value over 101 classes) |
| `mivolo`      | PyTorch/timm ViT  | continuous age, e.g. `31`      |

Checkbox per built model in the web app sidebar. Default: `fairface`.

With two or more age backends active the UI leads with a `best (<model>)` row: the estimate
from the most accurate backend present, ranked `mivolo` > `fairface` > `dex` > `caffe`.
Age is the one feature that is **not** fused, because fusing it measured worse -- see
[Combined answers](#combined-answers).

FairFace age, gender, and race use the existing MediaPipe landmarks when available, mapped
to the four eye corners and nose used by [dlib's five-point face chip](https://github.com/davisking/dlib/blob/master/dlib/image_transforms/interpolation.h).
The similarity crop uses [FairFace's padding of 0.25](https://github.com/dchen236/FairFace/blob/master/predict.py).
MediaPipe is an approximation of the original dlib landmark detector, not an identical
replacement. Missing or degenerate landmarks retain the bbox crop. Geometry and pipeline
tests verify the contract; accuracy gains still require a labeled photo benchmark.

`dex` (Deep EXpectation, Rothe et al. ICCV 2015) is a VGG-16 trained on IMDB-WIKI, a heavy age option (513MB caffemodel). **Research/academic-use license** (ETH Zurich, IMDB-WIKI-derived) -- not for commercial deployments without independent licensing.

DEX uses the original detector box with 40% margins on each axis and replicated image edges,
matching the authors' [crop extraction](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/extractSubImage.m).
The input remains 224x224 BGR with the existing ImageNet channel-mean approximation; it does
not use landmark warping or Haar roll correction. Age is the normalized expectation over
101 probabilities, as specified by the [authors](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
Invalid distributions return `unknown`. Distributions with standard deviation above 10 years
show `uncertain (mean ..., SD ...)` instead of a bare age. This is an uncalibrated display
heuristic, not a confidence interval or proof that narrower predictions are correct.

`mivolo` (MiVOLO: Multi-input Transformer for Age/Gender, Apache 2.0, WildChlamydia/MiVOLO) is a vision transformer that runs in face-only mode (no body context in this pipeline), sharing one checkpoint with the gender backend. Accuracy is ~4.24 years age MAE (face-only mode); heaviest option (~110MB checkpoint plus ultralytics/timm dependencies). Web app only.

### Gender

| Backend            | Framework            | Output            |
| ------------------ | -------------------- | ----------------- |
| Levi & Hassner CNN | Caffe (cv2.dnn)       | `Male` / `Female` |
| `deepface`         | Keras/TensorFlow      | `Male` / `Female` |
| `fairface`         | ONNX (cv2.dnn)        | `Male` / `Female` |
| `mivolo`           | PyTorch/timm ViT      | `Male` / `Female` |

`deepface` shares its VGGFace backbone code (`src/nets/deepface_common.py`) with the deepface race backend, but is a separate 537MB weight file (`models/deepface_gender.h5`) and needs TensorFlow like deepface race does. `fairface` shares one ONNX file (`models/fairface_7class.onnx`) across all three of age, gender, and race -- one model, three feature outputs (named `age_output`/`gender_output`/`race_output` in the same graph).

`mivolo` (Apache 2.0, WildChlamydia/MiVOLO) shares its 110MB checkpoint with the mivolo age backend -- one model, two feature outputs. Face-only mode (no body context). Web app only.

### Race (web app only)

| Backend    | Framework        | Output                                                                                        |
| ---------- | ---------------- | --------------------------------------------------------------------------------------------- |
| `fairface` | ONNX (cv2.dnn)   | 7 classes: White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern |
| `deepface` | Keras/TensorFlow | 6 classes: asian, indian, black, white, middle eastern, latino hispanic                       |

If the top-2 predicted classes are within 10 percentage points of each other, both are shown together (e.g. `White (52%)/Black (47%)`) instead of just the top class. `deepface` is by far the heaviest option in the repo: a 513MB weight file plus TensorFlow itself (~200-400MB) -- only pulled into the image if requested.

### Emotion

| Backend                  | Framework         | Output                                                         |
| ------------------------ | ----------------- | -------------------------------------------------------------- |
| `ferplus`                | ONNX (cv2.dnn)    | 8 classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt |
| `mini_xception`          | Keras/TensorFlow  | 7 classes: angry, disgust, fear, happy, sad, surprise, neutral |
| `dan` (DAN, AffectNet-7) | PyTorch           | 7 classes: neutral, happy, sad, surprise, fear, disgust, anger |
| `hsemotion` (enet_b0_8_best_vgaf, AffectNet-8) | ONNX (cv2.dnn) | 8 classes: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise |

Default: `hsemotion`. Note the class label order (and count) differs between backends -- each is tracked as a separate constant, never assumed to match. `mini_xception` is tiny (853KB, oarriaga/face_classification, MIT) but needs TensorFlow like the deepface models. `ferplus` is the official ONNX Model Zoo emotion model (MIT, 35MB, no extra framework -- pure cv2.dnn ONNX), used in place of a third-party PyTorch checkpoint for security reasons (no untrusted pickle deserialization). `hsemotion` (HSE-asavchenko/EmotiEffLib, Apache-2.0 code, 16MB EfficientNet-B0 backbone) is pretrained on VGGFace2 and fine-tuned on AffectNet-8 -- same AffectNet-derived weight provenance as `dan` (research/educational use, no explicit commercial weight license); pure cv2.dnn ONNX, no extra framework, no torch needed.

### Combined answers

With more than one model active for a feature, the per-face card leads with a single combined
answer and lists every individual model underneath. Gender, race, and emotion are **fused**;
age instead names its **best** available model.

| Feature | Combined row | How |
| ------- | ------------ | --- |
| Gender  | `fused`      | Weighted mean of each model's P(Male) |
| Race    | `fused`      | Weighted blend of class probabilities, re-expressed over shared canonical classes (FairFace's East/Southeast Asian collapse into one `Asian` class, since the two backends' label sets differ) |
| Emotion | `fused`      | Weighted vote over canonical emotion names (`happy`/`happiness` are one vote, not two) |
| Age     | `best (<model>)` | The most accurate backend present: `mivolo` > `fairface` > `dex` > `caffe` |

Weights are measurements, not taste. `tools/benchmark.py` scores every backend against
`tools/ground_truth.json` -- 75 hand-labelled faces across the `assets/` images, each bound to
its label by normalized face centre so the labels survive detector changes -- and the weights
in `src/inference.py` track those accuracies, so a weaker backend contributes proportionally
less instead of dragging the combined answer toward its own error.

Measured on those 75 faces (detection recall 100%):

| Feature | Combined | Best single model | Weakest active model |
| ------- | -------- | ----------------- | -------------------- |
| Gender  | **100%** | `mivolo` 100%     | `caffe`/`deepface` 86.7% |
| Emotion | **100%** | `dan`/`hsemotion`/`ferplus` 100% | `mini_xception` 95.2% |
| Race    | **96.0%** | `fairface` 93.3% | `deepface` 94.7% |
| Age     | **93.3%** (= `mivolo`) | `mivolo` 93.3% | `caffe` 62.7% |

Age is deliberately not fused: every combination tried -- weighted median and weighted mean
across a range of weights, clipping MiVOLO into FairFace's predicted decade, and overriding
MiVOLO only when both other backends disagreed with it -- scored at or below MiVOLO alone,
because the age backends fail on the same faces (elderly faces read young in all of them).
Averaging moves that answer without correcting it, so the headline names a winner instead.

Race is scored strictly here: only the top class counts, even though the UI also shows a
close runner-up (`White (52%)/Black (47%)`) when two classes are within 10 percentage points.
Counting either shown class as correct would read 97.3% fused / 96.0% `fairface`, but only 2
of the 75 fused answers show a runner-up at all, so the strict number is the honest one.

Caveats worth stating plainly: 75 faces is a small corpus, so differences of one or two faces
(about 1.3 percentage points) are noise, and only three of those faces have externally
confirmed ages -- the rest of the age ranges are careful visual estimates, so age accuracy is
measured against judgement, not documented fact. Re-run `tools/benchmark.py` after changing
any model or its preprocessing, and re-derive the weights if the ordering moves.

### Gaze (web app only)

| Backend | Framework | Output |
| ------- | --------- | ------ |
| `mediapipe` | MediaPipe Face Landmarker | coarse direction such as `left/level` or `center/down` |

Gaze reuses the same Face Landmarker model as Face Landmarks. It estimates
coarse direction from iris and eye geometry within each face crop; the result is an attention
cue, not a calibrated eye tracker.

### Recognition (web app only)

| Backend    | Framework        | Output                                             |
| ---------- | ---------------- | --------------------------------------------------- |
| `vggface`  | Keras/TensorFlow | enrolled identity name + similarity, e.g. `Alice (82%)`, or `UNKNOWN` |
| `lbph`     | OpenCV (`cv2.face`) | enrolled identity name + LBPH confidence, e.g. `Bob (34)`, or `UNKNOWN` |

Reuses the same VGGFace backbone as the deepface race/gender heads (`src/nets/deepface_common.py`), truncated to its 4096-d penultimate layer as a face embedding (`src/nets/deepface_recognition.py`) instead of a classification head -- same represent+verify shape as the original DeepFace paper. Embeddings are L2-normalized; identity is decided by cosine similarity against every enrolled face in the gallery, with a match only reported above `RECOGNITION_COSINE_THRESHOLD = 0.68` (deepface's own default VGG-Face verification threshold). Needs TensorFlow, like deepface race/gender.

Enrollment happens in the web app: under any detected face with a computed embedding, enter a name and click ENROLL. The gallery is stored as `gallery/known_faces.json` (one L2-normalized 4096-d vector per name), created on first enrollment and gitignored as runtime user data. It survives app restarts; Docker users should volume-mount `gallery/` (e.g. `-v $(pwd)/gallery:/app/gallery`) to persist enrollments across container restarts. Sidebar has a GALLERY section listing enrolled names with a delete button per entry.

**Known issue:** `models/deepface_vgg.h5` (the `vggface` recognition/identity-search weight file) is not currently present in this repo -- it was never committed. `vggface` recognition and Identity Search's vggface-based matching are wired and ready but non-functional until that file is sourced and added. **This does not affect `lbph`** (see below), which has no pretrained weight file at all.

`lbph` (Local Binary Patterns Histogram, `ideas/lbph.md`) is architecturally different from `vggface`: instead of comparing embeddings, `cv2.face.LBPHFaceRecognizer` trains directly on raw enrolled face images and has no pretrained weights to ship -- like this app's eigenfaces feature, it trains fresh on demand (once per analyzed frame, not once per face, to bound the cost). ENROLL saves the actual grayscale face crop to `gallery/lbph/<name>/NNNN.png` (accumulating -- more photos per person generally improves accuracy) rather than a single embedding vector; the ENROLL button appears whenever either `vggface` or `lbph` is available, and enrolls into whichever backend(s) are active. `LBPH_CONFIDENCE_THRESHOLD = 80.0` (lower confidence is a better match, the opposite convention from `vggface`'s cosine similarity) is within OpenCV's commonly-cited "reasonably confident" range, but -- like `EIGENFACE_DISTANCE_THRESHOLD` -- has no published reference value the way `RECOGNITION_COSINE_THRESHOLD` does; expect it to need tuning for your own enrolled faces. **Needs `opencv-contrib-python-headless`** instead of this repo's default `opencv-python-headless` (see `RECOGNITION_MODEL` in the Docker ARG table) -- `cv2.face` only ships in the contrib build.

### Identity Search (web app only)

A SEARCH button next to ENROLL on every detected face's card. Unlike Recognition's gallery (which you build yourself via ENROLL), Identity Search matches against a small set of *bundled* reference photos in `known_people/` -- shipped with the repo so it works out of the box for a few famous people (currently: Barack Obama, Joe Biden, Donald Trump, all public-domain official White House photos). Clicking SEARCH runs the same cosine-similarity match as Recognition (`RECOGNITION_COSINE_THRESHOLD`), against the bundled photos plus, optionally, a second directory you point it at via the sidebar's "SEARCH DIRECTORY" field.

To add more known people, drop a photo with one clear face into `known_people/` named `First_Last.jpg` (underscores become the displayed name, e.g. `Ada_Lovelace.jpg` -> "Ada Lovelace"). Each photo is face-detected and embedded on demand (cached for the bundled directory; a custom directory is rescanned on each search since its contents can change between runs).

**This does not search the internet.** There is no live reverse-image-search or web-scraping component -- matching is strictly against local image files (bundled or user-specified directory). Needs the same `vggface` recognition model as Recognition above (and inherits its "not currently functional" known issue -- see above).

SEARCH also independently checks **eigenfaces** (see below) against every previously-SAVEd face, regardless of whether the `vggface` model is available -- the two methods run together and either can report a match.

### Save & Eigenfaces Search (web app only, no model)

Every detected face's card also has SAVE and (now dual-purpose) SEARCH buttons:

- **SAVE** writes one row to a small SQLite database (`db/faces.db`), the face's color crop to `faces/{id}.jpg`, and a grayscale, tighter-cropped ("zoomed in") version to `eigen/{id}.jpg`. `id` is a random integer 1-999999, retried on collision. The database schema is **sparse and lazy**: there's no fixed column list -- a column (e.g. `age_caffe`) is only created the first time some SAVEd face actually has a value for that (feature, model) pair. A model that was never run, or never active, never gets a column. Every SAVE call independently extends the schema as needed (`ALTER TABLE ... ADD COLUMN`).
- **SEARCH**'s eigenfaces half runs Turk & Pentland's PCA algorithm (`ideas/eigenfaces.md`) fresh against every image in `eigen/` -- there's no persisted/trained model file, it retrains on the fly each time (cheap at the scale this is meant for: a personal collection of previously-saved faces, not a large dataset). Faces are normalized to a fixed 100x100 grayscale size; the query face goes through the exact same crop/resize pipeline as SAVE's `eigen/` output so the two are comparable. A match is reported by saved-face **ID** (there's no name at this layer -- look up `db/faces.db` by ID for whatever attributes were saved with it).

**`EIGENFACE_DISTANCE_THRESHOLD` is an untuned heuristic.** Unlike `RECOGNITION_COSINE_THRESHOLD` (deepface's own published default), there's no established reference value for raw-pixel eigenspace L2 distance at this face size -- it was verified to behave correctly (an unmodified saved face matches itself with near-zero distance; unrelated random images produce much larger distances) but the cutoff itself will need real-world tuning against your own saved faces. Per the algorithm's own known limitations (see `ideas/eigenfaces.md`): sensitive to lighting, pose, and scale -- front-facing, consistently-lit photos work best.

### Recognized / Unrecognized Scan (web app only, no model)

A `SCAN ALL FACES: RECOGNIZED / UNRECOGNIZED` button above the per-face cards, following `ideas/recognition.md`'s Recognized/Unrecognized labeling convention. Unlike SEARCH (one face, on demand), this checks **every** face detected in the image in a single pass: it eigenfaces-matches each one against `eigen/` (previously-SAVEd faces), then redraws the image with a green box + "Recognized" label per matched face or a red box + "Unrecognized" label otherwise, plus a text summary listing each face's matched saved-face ID where applicable. `match_faces_eigenfaces_batch()` trains PCA once for the whole image rather than once per face (`match_face_eigenfaces()`, used by the single-face SEARCH button, retrains on every call -- fine for one face, wasteful for N). Same untuned-threshold caveat as above.

### Crowd Count / Demographics (web app only, no model)

An opt-in `CROWD COUNT / DEMOGRAPHICS` sidebar checkbox, **off by default** (privacy-sensitive: it turns per-face results into an aggregate statistic about everyone in the image at once). Enabling it shows a caption explaining the tradeoff, and adds a `CROWD COUNT: N face(s) detected` expander below each analyzed image with a total count plus a bar-chart breakdown per currently active age/gender/race model. No new model or Docker build argument -- it's a pure tally (`aggregate_demographics()` in `src/inference.py`) over the age/gender/race outputs `analyze_frame()` already computed for that image; if two models are active for the same feature (e.g. `caffe` + `fairface` age), each gets its own independent breakdown rather than being merged, for the same reason DAN and FERPlus emotion labels aren't mixed (different label sets/granularity). Image upload and webcam SNAPSHOT only -- not webcam LIVE mode, which has no per-frame result panel to attach this to.

### 3D Reconstruction (web app only, ships no working weights)

| Backend   | Framework       | Output                                      |
| --------- | --------------- | -------------------------------------------- |
| `deep3d`  | PyTorch (ResNet50 + BFM) | downloadable `.obj` mesh (vertices + faces + per-vertex color) |

A `3D RECON` button per detected face, wired to [sicxu/Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) (MIT-licensed code) -- predicts 257 3D Morphable Model coefficients (80 identity + 64 expression + 80 texture + 3 pose angle + 27 spherical-harmonic lighting + 3 translation) via a ResNet50, then reconstructs a textured/lit mesh from Basel Face Model (BFM) basis vectors. Output is a Wavefront `.obj` (per-vertex color, no rendering/rasterization) rather than a rendered 2D image -- see Known Limitation below for why.

**Two files this feature needs are gated and NOT bundled or auto-downloadable:**

1. `models/deep3d_recon_resnet50.pth` -- the fine-tuned coefficient-regression checkpoint. Distributed only via a Google Drive folder linked from the upstream repo's README; no scriptable/direct URL exists.
2. `models/BFM/BFM_model_front.mat` -- derived from the Basel Face Model (BFM09), which is under Basel University's own non-commercial research license and requires registering at [faces.dmi.unibas.ch/bfm](https://faces.dmi.unibas.ch/bfm/) to obtain, plus an expression basis (`Exp_Pca.bin`, also Google-Drive-gated) to convert it via upstream's own `transferBFM09()` script. Neither this conversion step nor the raw BFM09 file is vendored here.

`models/BFM/similarity_Lm3D_all.mat` (a small ~1KB landmark-alignment template) **is** bundled -- it's from the same MIT-licensed upstream repo and isn't derived from BFM09 itself.

Without both gated files present, this feature shows as offline (`RECONSTRUCTION_3D` in the sidebar's OFFLINE list). If you have legitimate access to both (e.g. you're a BFM09 registrant with the converted `.mat` file, and you've downloaded the checkpoint), drop them into `models/` at the paths above and the feature activates automatically -- no rebuild needed if using the dev-mount option in `build-and-run.sh`.

**Known limitation -- no rendered preview, mesh only:** upstream's own rendering step uses [nvdiffrast](https://github.com/NVlabs/nvdiffrast), NVIDIA's differentiable rasterizer, which is GPU/CUDA-only with no CPU fallback -- incompatible with this repo's CPU-only design (same constraint documented for every other PyTorch/TensorFlow feature here). This integration skips rendering entirely and stops at mesh export, which needs no GPU: the coefficient regression and BFM linear-algebra reconstruction are both plain ResNet50 forward passes and matrix math, verified CPU-only. Open the downloaded `.obj` in Blender, MeshLab, or any online viewer to inspect it.

**Known limitation -- landmark source substituted:** upstream's own pipeline expects 5-point face landmarks from an external MTCNN-based tool (not bundled in their repo either). This integration derives the same 5 points from this app's existing MediaPipe FaceLandmarker instead (`landmarks_5pt_from_mediapipe` in `src/nets/deep3d_recon.py`) -- a reasonable approximation, not a re-implementation of their exact landmark source.

**Verified against real weights:** the real upstream checkpoint (`epoch_20.pth`, its `net_recon` sub-state-dict) and real BFM09 data (converted via upstream's own `util/load_mats.py transferBFM09()`, combined with the Guo et al. expression basis) were obtained and run end-to-end -- produced a valid 35709-vertex/70789-face mesh with plausible coordinate/color ranges from a real test image. It's a faithful line-for-line port of the published source (`models/networks.py`, `models/bfm.py`, `util/preprocess.py`), now confirmed correct against the real pipeline, not just synthetic weights.

### Age Progression / Regression (web app only, non-commercial use only)

| Backend    | Framework | Output                          |
| ---------- | --------- | -------------------------------- |
| `franunet` | PyTorch U-Net | before/after face-crop image pair (downloadable PNG) |

An `AGE PROGRESSION` button per detected face, wired to [timroelofs123/face_reaging](https://github.com/timroelofs123/face_reaging) (MIT-licensed code), a U-Net reproducing Disney Research's FRAN paper. Given a source age and target age (both user-entered, 0-100), the network predicts a residual that's added onto the face crop (resized to the model's native 512x512, then resized back), producing an aged/de-aged version of the same crop. Pretrained weights (`best_unet_model.pth` -> bundled here as `models/face_reaging_unet.pth`) are downloaded directly from [Hugging Face](https://huggingface.co/timroelofs123/face_re-aging), loadable with plain `torch.load` -- no training or gated download needed, unlike this repo's 3D Reconstruction model.

**License caveat -- non-commercial, and NOT just a training-data footnote:** the U-Net's `DownLayer`/`UpLayer` blocks use `BlurPool` (anti-aliased strided downsampling), vendored here from Adobe's [antialiased-cnns](https://github.com/adobe/antialiased-cnns) (see `src/nets/face_reaging_model.py`). antialiased-cnns is licensed **Creative Commons Attribution-NonCommercial-ShareAlike 4.0** -- non-commercial only, and share-alike (redistributions/adaptations must carry the same license). This is a *required inference-time component of the network architecture itself*, not merely a caveat about what data the weights were trained on. Separately, the pretrained weights were trained on FFHQ images re-aged via SAM (built on StyleGAN2, NVIDIA's own non-commercial research license), so the training-data provenance also traces back through a non-commercial-licensed tool. Net effect: treat this whole feature as **non-commercial/research use only**.

### Liveness / anti-spoofing

Liveness has one `mediapipe` backend. It reuses `models/face_landmarker.task`, selected by
`LIVENESS_MODEL=mediapipe`, plus two lightweight heuristics. In webcam LIVE mode,
MediaPipe eye-blink blendshapes are tracked by the existing stable face ID; a blink transition
increments the count and produces a blink rate per minute. Every face crop also receives a
regular high-frequency texture score for screen, moire, or replay artifacts. A high texture
score takes precedence and reports `SUSPECTED SPOOF`; a blink reports `LIVE`; no temporal blink
evidence on an image or new track reports `INCONCLUSIVE`.

These are screening cues, not biometric proof. Lighting, compression, makeup, camera focus, and
display hardware can produce false positives or false negatives.

### Voice + Face Fusion (web app, webcam LIVE mode only)

Enable `VOICE + FACE FUSION` in the LIVE webcam controls to request microphone access. Audio is
converted to mono and analyzed as normalized short-term RMS energy over a rolling 1.5-second
window. The video callback compares that signal with the emotion label of the largest detected
face and reports whether their coarse arousal levels are `consistent` or `inconsistent`.

This is an opt-in heuristic, not speech-emotion recognition. It does not perform frame-accurate
lip-sync, identify which person is speaking, or produce a clinically or scientifically validated
emotion measurement. With multiple faces, only the largest face is used. Uploads and webcam
snapshots do not include audio and therefore do not use this fusion feature. Browser microphone
permissions and a working WebRTC connection are required.

### Glasses (web app only)

| Backend     | Framework      | Output              |
| ----------- | -------------- | -------------------- |
| `mobilenet` | ONNX (onnxruntime) | `glasses` / `none`  |

Sorour190/Glasses-Detector, `models/glasses_detector.onnx` (MobileNetV3-Large, 224x224 RGB, normalization baked into the ONNX graph itself -- feed raw uint8 pixels). **License unstated by the source repo** -- same treatment as DAN, use at your own discretion.

### Mask (web app only)

| Backend        | Framework        | Output                        |
| --------------- | ----------------- | ------------------------------ |
| `mobilenetv2`  | Keras/TensorFlow  | `with_mask` / `without_mask`  |

chandrikadeb7/Face-Mask-Detection (MIT), MobileNetV2 backbone + AveragePooling2D/Flatten/Dense(128)/Dropout/Dense(2, softmax) head, 224x224 RGB, `mobilenet_v2.preprocess_input` scaling ([-1, 1]). Needs TensorFlow, like deepface/mini_xception/recognition.

### Hair Color (web app only -- heuristic, not ML)

| Backend         | Framework | Output                                              |
| ---------------- | --------- | ---------------------------------------------------- |
| `colorimetric`  | OpenCV    | `black` / `brown` / `blonde` / `red` / `grey` / `white` |

No model file, no dependency, always available. Samples the region above the detected face box, excludes likely-skin pixels (rough HSV skin-color range), takes the median HSV of what's left, and buckets by hue/saturation/value against fixed thresholds. **This is a plain colorimetric heuristic, not a trained classifier** -- accuracy is meaningfully lower than the model-backed attributes and is sensitive to lighting, hats, camera white-balance, and hairstyle framing. Treat results as a rough guess, not a benchmark-grade prediction.

### Eye Color (web app only -- heuristic, not ML)

| Backend         | Framework | Output                                                  |
| ---------------- | --------- | --------------------------------------------------------- |
| `colorimetric`  | OpenCV    | `brown` / `blue` / `green` / `hazel` / `grey` / `amber`  |

No model file. Reuses `haarcascade_eye.xml` when that file is present. Locates the largest detected eye box, samples its center 40% (avoiding sclera/eyelid), and buckets the median HSV against fixed thresholds. **Also a plain colorimetric heuristic, not a trained classifier** -- same lighting/pose-sensitivity caveats as Hair Color, generally the least reliable attribute in the app.

### Colorization (web app only, not a face attribute)

| Backend    | Framework       | Output                              |
| ---------- | --------------- | ------------------------------------ |
| `eccv16`   | Caffe (cv2.dnn) | Colorized BGR frame, or unchanged   |

Zhang et al.'s ECCV16 colorization model (`models/colorization_deploy_v2.prototxt` / `_release_v2.caffemodel` / `pts_in_hull.npy`, BSD-2-Clause, richzhang/colorization). Unlike every other feature above, this isn't a per-face attribute -- it's a whole-frame preprocessing step applied *before* face detection. If the uploaded/captured frame is auto-detected as grayscale (near-zero difference between its B/G/R channels), it's colorized in Lab space (predict `ab` from `L`, per `ideas/colorization.md`) before the rest of the pipeline runs, so downstream color-dependent attributes (hair color, eye color) see the colorized version too. On by default; toggle off in the sidebar (`AUTO-COLORIZE B&W`) to leave grayscale images untouched. Already-color images are left alone regardless of the toggle (the grayscale check skips them).

### Face Landmarks (web app only)

| Backend        | Framework | Output                                    |
| --------------- | --------- | -------------------------------------------- |
| `mediapipe`   | MediaPipe | 468-point face mesh overlay, drawn per face |

Uses MediaPipe's `FaceLandmarker` (`models/face_landmarker.task`) and draws 468 small dots per detected face directly onto the shared annotated image. Toggle in the sidebar (`FACE LANDMARKS`).

### Hand Landmarks (web app only)

| Backend      | Framework | Output                                        |
| ------------- | --------- | ------------------------------------------------ |
| `mediapipe`  | MediaPipe | 21-point skeleton per detected hand, up to 2 hands |

Google's MediaPipe HandLandmarker (`models/hand_landmarker.task`, Apache 2.0). Like Pose and Colorization, this is a whole-frame feature -- it runs once per frame independent of face detection, so hands are found (or not) whether or not a face is in view. If no hands are detected, nothing is drawn and nothing reported -- that's how "if hands are visible" is implemented, there's no separate hand-presence check beyond the model's own empty-result case. When hands are found, each one's 21-point skeleton is drawn directly onto the frame and a `[ HANDS DETECTED ]` caption is shown. Toggle in the sidebar (`HAND LANDMARKS`).

### Image Adjustments (web app only, not a model)

| Slider            | Range        | Effect                                                    |
| ------------------- | ------------ | ------------------------------------------------------------ |
| Exposure           | -3.0 .. 3.0  | stops (2^value gain)                                        |
| Brightness         | -100 .. 100  | additive offset                                              |
| Contrast           | -100 .. 100  | classic contrast-correction-factor curve around midtone      |
| Highlights         | -100 .. 100  | luminance-weighted lift/cut on bright tones only             |
| Shadows            | -100 .. 100  | luminance-weighted lift/cut on dark tones only               |
| Black Point        | -100 .. 100  | remaps the shadow floor (levels-style)                       |
| Saturation         | -100 .. 100  | uniform HSV saturation scale                                 |
| Vibrance           | -100 .. 100  | saturation boost weighted toward already-desaturated pixels (protects skin tones) |
| Sharpness          | 0 .. 100     | small-radius unsharp mask                                    |
| Definition         | 0 .. 100     | large-radius unsharp mask on LAB lightness only ("clarity")  |
| Noise Reduction    | 0 .. 100     | bilateral filter                                              |

Eleven Lightroom-style sliders, all defaulting to 0 (no-op). Pure OpenCV/numpy -- no model file, no build ARG, always available. There are **two independent slider panels**, applied at two different points in the pipeline:

- **GLOBAL IMAGE ADJUSTMENTS**: applied to the whole image first, before face detection even runs. Every downstream output -- the annotated image, every face crop, every classification -- sees the adjusted pixels. Useful for e.g. brightening a dark source image so face detection itself finds more faces.
- **PER-FACE IMAGE ADJUSTMENTS**: applied again, separately, to each detected face's own crop -- after detection, before any classifier runs on it. This only affects that one face's thumbnail and attribute results, not the shared frame or other faces.

Each detected face card also has its own **Edit face** sliders. They change that face's preview, edited PNG download, and input to the one-click image operations. Other faces and the classification labels stay unchanged. All sliders use the same `apply_image_adjustments()` function; defaults are a no-op.
Each slider panel has a reset button. **Reset all adjustments** clears the whole-image, shared face, and individual face sliders together.

### Select Region & Transform (web app only)

| Backend | Framework | Output |
| ------- | --------- | ------ |
| Rectangle crop and geometric transform | OpenCV | Transformed PNG |

Open **SELECT REGION & TRANSFORM** beneath an uploaded image or webcam snapshot, enter rectangle bounds, choose translate, reflect, rotate, scale, or shear, then apply and download the result. Coordinates use the image's pixel dimensions; an empty rectangle is rejected. Translation, reflection, and rotation retain the crop size, while scaling and shearing can change it. This works even when no face is detected. No model file or Docker build argument is needed.

### Per-Face Image Operations (web app only)

| Operation | Framework | Methods |
| --------- | --------- | ------- |
| Intensity | OpenCV | negative, log, gamma, contrast stretch |
| Enhance | OpenCV | brightness, contrast, luminance equalization |
| Sharpen | OpenCV | Laplacian, high boost |
| Color correct | OpenCV | LAB contrast adjustment |
| Denoise | OpenCV | Gaussian, median, non-local means |
| Bilateral filter | OpenCV | edge-preserving smoothing |
| Wavelet denoise | PyWavelets | wavelet soft thresholding |

Each detected face has an **Edit face** expander containing its sliders, **IMAGE OP** selector, and **APPLY IMAGE OP** button. The filters stay hidden until that expander opens. Operations act on that face's displayed crop; the result can be downloaded as a PNG. Intensity, sharpen, and denoise expose a method selector. These operations need no model files or Docker build arguments. CNN and GAN denoising are not included because trained weights are not supplied.

Model provenance: DAN and DeepFace's race model are vendored research code (`src/nets/`). DAN has no explicit upstream license file (research/educational use). DeepFace (race, gender, and recognition/`deepface_vgg.h5`) is MIT. FairFace's ONNX conversion is MIT (underlying dataset CC BY 4.0). Face-Mask-Detection is MIT; the glasses detector's license is unstated. HSEmotion (HSE-asavchenko/EmotiEffLib) code is Apache-2.0; its AffectNet-8 fine-tuned weight has the same research/educational provenance as DAN. The age-reaging U-Net (`franunet`) is MIT-licensed code, but its BlurPool component (vendored from Adobe's antialiased-cnns) is CC BY-NC-SA 4.0 -- non-commercial use only (see Age Progression / Regression above).

## Performance

- **Classifier frame skip (webcam LIVE mode only):** a `CLASSIFIER FRAME SKIP` slider (1-10, default 1 = every frame) above the LIVE video feed runs age/gender/emotion/race/recognition/glasses/mask/hair-color/eye-color classifiers every Nth frame instead of every frame. Face detection and the pose/hand/face-landmark overlays still run every frame, so the video itself stays smooth. This is safe to skip freely: those classifiers' outputs aren't otherwise drawn onto the LIVE video (per-face text cards only exist for Image upload / webcam SNAPSHOT), so there's no visible staleness to interpolate around -- skipping only reduces CPU load.
- **Per-face result caching:** `src/inference.py` memoizes most per-face classifier calls (age, gender, emotion, race, expression, glasses, mask, eye color, and the recognition embedding step) keyed on a hash of the exact preprocessed face-crop bytes fed to that model, not a face-identity embedding -- an embedding hash isn't a stable cache key for an adjusted or re-cropped face, but identical input bytes always produce identical deterministic output, so hashing the input itself is correct with no accuracy risk. This mainly helps Streamlit's rerun-the-whole-script-on-any-widget-change model: toggling one unrelated sidebar option no longer recomputes every classifier for every face from scratch. `fairface` (which key off the full frame + box rather than an isolated face crop) and the identity **match** step (which must stay live since the gallery can change between calls) are intentionally not cached. The cache is bounded (LRU-evicted, 2048 entries) and shared process-wide.

---

## Directory Layout

```text
multimodal-face-analyzer/
├── Dockerfile              # Container build (per-feature toggles, see below)
├── .dockerignore
├── build-and-run.sh        # Interactive build+run wrapper
├── requirements.txt        # Base dependencies (opencv/streamlit/etc.)
│
├── models/                 # Pre-trained weights & configs (data, not code)
│   ├── opencv_face_detector_uint8.pb / .pbtxt   # face detection (required)
│   ├── age_deploy.prototxt / age_net.caffemodel # age: caffe backend
│   ├── gender_deploy.prototxt / gender_net.caffemodel
│   ├── mivolo_v2.safetensors / _config.json     # age + gender: mivolo backend
│   ├── dan_affecnet7.pth                        # emotion: dan backend
│   ├── emotion_ferplus.onnx                     # emotion: ferplus backend
│   ├── mini_xception_fer.h5                     # emotion: mini_xception backend
│   ├── hsemotion_enet_b0_8_best_vgaf.onnx       # emotion: hsemotion backend
│   ├── fairface_7class.onnx                     # age + gender + race: fairface backend
│   ├── deepface_race.h5                         # race: deepface backend
│   ├── deepface_gender.h5                       # gender: deepface backend
│   ├── deepface_vgg.h5                          # recognition: vggface backend
│   ├── face_landmarker.task                     # face landmarks, gaze, and liveness
│   ├── haarcascade_eye.xml                      # eye color + roll alignment
│   ├── glasses_detector.onnx                    # glasses: mobilenet backend
│   ├── mask_detector.h5                         # mask: mobilenetv2 backend
│   ├── colorization_deploy_v2.prototxt / colorization_release_v2.caffemodel / pts_in_hull.npy  # colorization
│   ├── hand_landmarker.task                     # hand landmarks: mediapipe backend
│   ├── yolov8n_face.onnx                        # face detection: yolo backend (additive)
│   ├── scrfd_2.5g_bnkps.onnx                     # face detection: scrfd backend (additive)
│   ├── retinaface_mobilenet0.25.onnx             # face detection: retinaface backend (additive)
│   ├── BFM/
│   │   └── similarity_Lm3D_all.mat              # 3D recon: bundled landmark alignment template
│   │   # (no BFM_model_front.mat -- Basel Face Model, registration-gated, see README)
│   # (no deep3d_recon_resnet50.pth -- Google-Drive-gated checkpoint, see README)
│   # face_landmarker.task (above) is also reused for the Face Landmarks toggle
│   # (no deepface_vgg.h5 -- never committed, Recognition/Identity Search are non-functional until it's sourced)
│
├── known_people/           # bundled reference photos for Identity Search (First_Last.jpg)
├── db/                     # gitignored: faces.db (SQLite, sparse per-model columns)
├── faces/                  # gitignored: SAVEd faces' color crops, {id}.jpg
├── eigen/                  # gitignored: SAVEd faces' grayscale/zoomed crops for eigenfaces
├── gallery/                # gitignored: known_faces.json (vggface) + lbph/<name>/NNNN.png (lbph)
│
└── src/                    # Streamlit app module
    ├── app.py              # UI only (page layout, sidebar, tabs)
    ├── inference.py        # model loading + per-face prediction logic
    └── nets/               # vendored model architectures (code, not weights)
        ├── dan_model.py
        ├── deepface_common.py                   # shared VGGFace backbone
        ├── deepface_race.py
        ├── mini_xception_model.py
        ├── deepface_gender.py
        ├── deepface_recognition.py
        ├── mask_model.py                        # mask: mobilenetv2 backend
        ├── deep3d_recon.py                      # 3D recon (no working weights yet, see README)
        └── mivolo/                              # MiVOLO ViT (Apache 2.0)
            ├── __init__.py
            ├── loader.py                        # HF checkpoint adapter
            ├── inference_wrapper.py             # high-level inference API
            ├── predictor.py
            ├── structures.py
            ├── model/                           # ViT architecture
            ├── data/                            # preprocessing
            └── LICENSE_MIVOLO
```

---

## Local Setup

```bash
conda create -n vision_env python=3.11 -y
conda activate vision_env
pip install -r requirements.txt

# needed for: dan emotion, mivolo age/gender
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# needed for: deepface race and/or deepface gender (heavy -- ~200-400MB)
pip install tensorflow-cpu tf-keras
```

Without torch/tensorflow installed, the corresponding models are automatically skipped (not an error).

## Guided run scripts

Both run scripts use the same grouped prompts: FACE DETECTION, AGE, GENDER, RACE, EMOTION,
RECOGNITION, ADDITIONAL CLASSIFICATIONS, and ADDITIONAL FEATURES. In every group except FACE
DETECTION, `0) none` disables the group; FACE DETECTION uses `0) ssd` as the required fallback.
`9) all` selects every listed option. The default is marked with `*`. Multiple options may be entered as concatenated
digits (`1234`), space-separated digits (`1 2 3 4`), or comma-separated digits (`1,2,3,4`).

Headers are bold blue. Bold green options need no additional optional install; bold orange options
need an additional package. `none` and `all` are intentionally uncolored. The options are ordered
with green choices before orange choices where applicable.

### Native install and run

Run `./install-and-run.sh` for a guided native setup. It creates or reuses `.venv`, installs the
selected dependency groups, and starts the app at `http://localhost:8501`.

```bash
./install-and-run.sh
```

Useful flags:

- `-v` or `--verbose` shows up to two dependency lines beneath each model. Models retain their
  bold green/orange colors; package lines use regular-weight green/orange text.
- `-h` or `--hidden` hides choices that would add a new optional package. In hidden mode, `9) all`
  selects only the remaining visible choices.
- `-p` or `--package PACKAGE[,PACKAGE...]` marks packages you will provide yourself as already
  available for prompt coloring and hidden filtering. For example:

  ```bash
  ./install-and-run.sh --package onnxruntime,tensorflow-cpu
  ```

  The installer still installs packages required by selected backends. It accepts `onxruntime` as
  an alias for `onnxruntime`.

The native script requires `apt-get` and `sudo` on Debian/Ubuntu to install the OpenCV and MediaPipe
runtime libraries. On other systems, install the equivalent `libgl1`, `libglib2.0-0`, `libegl1`, and
`libgles2` packages yourself.

### Docker build and run

Run `./build-and-run.sh` for the equivalent grouped prompts followed by a Docker build and launch.
It stages only the selected model files into a temporary Docker build context, builds them into the
image, asks whether to live-mount `src/` for testing, and then starts the container. This avoids
sending the full models directory (currently more than 3GB) to Docker for every guided build.
While the container is running, enter `q` to stop it or `d` to stop it and remove the image and build
cache.

---

## Docker Web App

### Build

```bash
# guided prompt -- grouped numbered options; multi-select as 1234, 1 2 3 4, or 1,2,3,4
./build-and-run.sh

# or manually, selecting the model keys to bake into the image
docker build -t face-analyzer .
```

Each build ARG takes a comma-separated list of model keys for that feature, or an empty string for
none. The guided prompt defaults to the first option in each group; SSD remains the required face
detector fallback while YOLO, SCRFD, and RetinaFace are additive choices:

```bash
docker build \
  --build-arg AGE_MODEL=fairface,caffe,dex,mivolo \
  --build-arg GENDER_MODEL=fairface,caffe,deepface,mivolo \
  --build-arg EMOTION_MODEL=hsemotion,ferplus,mini_xception,dan \
  --build-arg RACE_MODEL=fairface,deepface \
  --build-arg FACE_LANDMARKS_MODEL=mediapipe \
  --build-arg LIVENESS_MODEL=mediapipe \
  --build-arg RECOGNITION_MODEL=vggface,lbph \
  --build-arg GLASSES_MODEL=mobilenet \
  --build-arg MASK_MODEL=mobilenetv2 \
  --build-arg COLORIZATION_MODEL=eccv16 \
  --build-arg HAND_MODEL=mediapipe \
  --build-arg RECONSTRUCTION_3D_MODEL=deep3d \
  --build-arg YOLO_FACE_MODEL=yolo \
  --build-arg SCRFD_FACE_MODEL=scrfd \
  --build-arg RETINAFACE_MODEL=retinaface \
  --build-arg AGE_PROGRESSION_MODEL=franunet \
  -t face-analyzer .
```

| Build arg           | Options (default first)          |
| -------------------- | -------------------------------- |
| `AGE_MODEL`        | `fairface`, `caffe`, `dex`, `mivolo` |
| `GENDER_MODEL`     | `fairface`, `caffe`, `deepface`, `mivolo` |
| `EMOTION_MODEL`    | `hsemotion`, `ferplus`, `mini_xception`, `dan` |
| `RACE_MODEL`       | `fairface`, `deepface`                   |
| `FACE_LANDMARKS_MODEL` | `mediapipe`                        |
| `LIVENESS_MODEL` | `mediapipe` (reuses `face_landmarker.task`) |
| `RECOGNITION_MODEL` | `vggface`, `lbph`                       |
| `GLASSES_MODEL`    | `mobilenet`                              |
| `MASK_MODEL`       | `mobilenetv2`                            |
| `COLORIZATION_MODEL` | `eccv16`                                |
| `HAND_MODEL`       | `mediapipe`                               |
| `RECONSTRUCTION_3D_MODEL` | `deep3d` (ships no working weights, see [3D Reconstruction](#3d-reconstruction-web-app-only-ships-no-working-weights)) |
| `YOLO_FACE_MODEL`  | `yolo` (additive -- SSD stays required/always on) |
| `SCRFD_FACE_MODEL` | `scrfd` (additive -- SSD stays required/always on) |
| `RETINAFACE_MODEL` | `retinaface` (additive -- SSD stays required/always on) |
| `AGE_PROGRESSION_MODEL` | `franunet` (non-commercial use only, see [Age Progression / Regression](#age-progression--regression-web-app-only-non-commercial-use-only)) |

Hair Color and Eye Color are colorimetric heuristics with no model file and thus no build ARG either -- they're always available in the web app (Eye Color additionally needs `haarcascade_eye.xml`). Face Landmarks uses `FACE_LANDMARKS_MODEL=mediapipe`; Liveness uses its own `LIVENESS_MODEL=mediapipe` ARG while reusing the same model file.

Multiple models per feature (e.g. `AGE_MODEL=fairface,caffe`) can be built in together -- the web app sidebar shows a checkbox per built model, and checking more than one for the same feature runs and displays all of them at once.

Disabled model files never land in an image layer. Guided builds also exclude them from the Docker
build context by staging only selected files. Direct `docker build .` retains the conditional-copy
behavior but still sends the full context. `torch`/`torchvision` (~200MB) are only installed if
`dan`, `mivolo`, `deep3d`, and/or `franunet` are requested (`scipy` is additionally
installed for `deep3d` alone, to load `.mat` files). `tensorflow-cpu`/`tf-keras` (~200-400MB, plus
deepface's 513MB weight file) are only installed if `deepface`, `mini_xception`, `vggface`, and/or
`mask` are requested -- deepface race remains by far the heaviest single option in the repo (note:
`mivolo` at ~110MB checkpoint plus ultralytics/timm dependencies is the second-heaviest, still much
lighter than deepface's full stack). `mediapipe` is only installed if face landmarks, liveness,
or hand landmarks are requested. `onnxruntime` is installed if `yolo`/`scrfd`/`retinaface` (face detector) or
`mobilenet` (glasses) is requested. `opencv-contrib-python-headless` replaces the default
`opencv-python-headless` only if `lbph` is requested (needed for `cv2.face`).

### Run

```bash
docker run -d -p 127.0.0.1:8501:8501 --name face_analyzer_container face-analyzer
```

Then open `http://localhost:8501`.

### Remote-access trust boundary

This app has no user authentication or authorization. The container listens on its
internal interface so Docker can route traffic to it, but the documented run command
publishes it on the host loopback interface only. Treat all face images, enrollments,
and saved face metadata as trusted-network data. For remote access, put the app behind
an authenticating, TLS-terminating reverse proxy or a private network/VPN, and expose
only that protected proxy; do not publish port 8501 directly to the public internet.
Streamlit XSRF protection is enabled in the container command, but it is not an
authentication boundary.
