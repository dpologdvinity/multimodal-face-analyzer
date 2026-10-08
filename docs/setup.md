# Setup

How to install, build, and run the analyzer natively or in Docker, plus the full table of
Docker build arguments. For what each model does, see [models.md](models.md).

## Model weights (git-lfs)

Every `*.h5`, `*.pth`, `*.pt`, `*.caffemodel`, `*.safetensors`, `*.onnx`, and `*.task` file in
`models/` is stored with [git-lfs](https://git-lfs.com/). The full tracked set is about 3.2 GB.
A clone without LFS contains small pointer files in their place, which the model loaders cannot
read, so pull the weights before running natively. A model whose file is absent altogether is
skipped and reported as unavailable; a pointer file is not.

```bash
git lfs install
git clone https://github.com/dpologdvinity/multimodal-face-analyzer.git
cd multimodal-face-analyzer
git lfs pull                                  # all weights
git lfs pull --include="models/fairface_7class.onnx,models/hsemotion_enet_b0_8_best_vgaf.onnx"  # or a subset
```

Set `FACE_ANALYZER_MODEL_DIR` to load weights from a directory other than `models/`.

## Local setup

The base install covers the required SSD face detector and every OpenCV-only backend:

```bash
conda create -n vision_env python=3.11 -y
conda activate vision_env
pip install -r requirements.txt
streamlit run src/app.py
```

Optional backends need heavier packages (PyTorch, TensorFlow, MediaPipe, onnxruntime). The
easiest way to get them is the [guided installer](#native-install-and-run), which installs only
the packages your selected backends need and reinstalls the pinned OpenCV last, or
[Docker](#docker-web-app), which pins every optional package. If you manage the environment
yourself, these are the versions the Dockerfile pins:

| Needed for | Packages (Dockerfile pins) |
| ---------- | -------- |
| `dan` emotion, `mivolo` age/gender, `deep3d` 3D reconstruction, `franunet` age progression | `torch==2.14.1+cpu torchvision==0.29.1+cpu` (with `--extra-index-url https://download.pytorch.org/whl/cpu`) |
| `mivolo` age/gender | `ultralytics==8.1.0 timm==0.8.13.dev0 safetensors==0.8.0 huggingface_hub==2.1.1` |
| `deepface` race/gender, `mini_xception` emotion, `vggface` recognition, `mobilenetv2` mask | `tensorflow-cpu==2.21.0 tf-keras==2.21.0` |
| `yolo`/`scrfd`/`retinaface` detectors, `mobilenet` glasses | `onnxruntime==1.30.0` |
| Face landmarks, gaze, liveness, hand landmarks | `mediapipe==1.1.0` |
| `deep3d` 3D reconstruction | `scipy==1.17.1` |
| `lbph` recognition | `opencv-contrib-python-headless==4.14.0.94` in place of `opencv-python-headless` |

If you install them by hand, reinstall OpenCV last. `ultralytics` and `mediapipe` pull in their
own OpenCV 5.x builds, which drop Caffe support and break the `caffe`/`dex` backends. Remove
every OpenCV variant, then reinstall `opencv-python-headless==4.14.0.94` (or
`opencv-contrib-python-headless==4.14.0.94` for `lbph`), as the Dockerfile and the guided
installer both do.

Without an optional package installed, the models that need it are skipped, not an error.

### Tests and lint

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
```

Tests that need a real weight file skip when it is missing or still an LFS pointer, so the suite
runs on a clone without `git lfs pull`. CI runs ruff and the test suite on Python 3.11 and 3.12
against exactly such a clone.

## Guided run scripts

Both run scripts use the same grouped prompts: FACE DETECTION, AGE, GENDER, RACE, EMOTION,
RECOGNITION, ADDITIONAL CLASSIFICATIONS, and ADDITIONAL FEATURES. In every group except FACE
DETECTION, `0) none` disables the group; FACE DETECTION uses `0) ssd` as the required fallback.
`9) all` selects every listed option. The default is marked with `*`. Multiple options may be
entered as concatenated digits (`1234`), space-separated digits (`1 2 3 4`), or comma-separated
digits (`1,2,3,4`).

Headers are bold blue. Bold green options need no additional optional install; bold orange options
need an additional package. `none` and `all` are intentionally uncolored. The options are ordered
with green choices before orange choices where applicable.

### Native install and run

Run `./install-and-run.sh` for a guided native setup. It creates or reuses `.venv`, installs the
selected dependency groups, and starts the app at `http://localhost:8501`.

```bash
./install-and-run.sh
```

The script pins OpenCV, `ultralytics`, and `timm`, but installs the latest `torch`,
`tensorflow-cpu`, `mediapipe`, `onnxruntime`, and `scipy`. For the exact versions the Docker
image is built with, use Docker or install the pinned versions from the table above.

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
sending the full models directory (more than 3 GB) to Docker for every guided build.
While the container is running, enter `q` to stop it or `d` to stop it and remove the image and build
cache.

## Docker web app

### Build

```bash
# guided prompt -- grouped numbered options; multi-select as 1234, 1 2 3 4, or 1,2,3,4
./build-and-run.sh

# or manually, with the Dockerfile defaults (one backend per feature)
docker build -t face-analyzer .
```

Each build ARG takes a comma-separated list of model keys for that feature, or an empty string for
none. SSD remains the required face detector fallback while YOLO, SCRFD, and RetinaFace are
additive choices. A build with every backend:

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

| Build arg | Options (default first) |
| --------- | ----------------------- |
| `AGE_MODEL` | `fairface`, `caffe`, `dex`, `mivolo` |
| `GENDER_MODEL` | `fairface`, `caffe`, `deepface`, `mivolo` |
| `EMOTION_MODEL` | `hsemotion`, `ferplus`, `mini_xception`, `dan` |
| `RACE_MODEL` | `fairface`, `deepface` |
| `FACE_LANDMARKS_MODEL` | `mediapipe` (gaze and head pose use the same landmarker) |
| `LIVENESS_MODEL` | `mediapipe` (reuses `face_landmarker.task`) |
| `RECOGNITION_MODEL` | `vggface`, `lbph` |
| `GLASSES_MODEL` | `mobilenet` |
| `MASK_MODEL` | `mobilenetv2` |
| `COLORIZATION_MODEL` | `eccv16` |
| `HAND_MODEL` | `mediapipe` |
| `RECONSTRUCTION_3D_MODEL` | `deep3d` (no working default weights, see [3D Reconstruction](models.md#3d-reconstruction-no-working-default-weights)) |
| `YOLO_FACE_MODEL` | `yolo` (additive -- SSD stays required/always on) |
| `SCRFD_FACE_MODEL` | `scrfd` (additive -- SSD stays required/always on) |
| `RETINAFACE_MODEL` | `retinaface` (additive -- SSD stays required/always on) |
| `AGE_PROGRESSION_MODEL` | `franunet` (non-commercial use only, see [Age Progression](models.md#age-progression--regression-non-commercial-use-only)) |

Hair Color and Eye Color are colorimetric heuristics with no model file and thus no build ARG
either -- they're always available in the web app (Eye Color additionally needs
`haarcascade_eye.xml`). Face Landmarks uses `FACE_LANDMARKS_MODEL=mediapipe`; Liveness uses its own
`LIVENESS_MODEL=mediapipe` ARG while reusing the same model file.

Multiple models per feature (e.g. `AGE_MODEL=fairface,caffe`) can be built in together -- the web
app sidebar shows a checkbox per built model, and checking more than one for the same feature runs
and displays all of them at once.

Disabled model files never land in an image layer. Guided builds also exclude them from the Docker
build context by staging only selected files. Direct `docker build .` retains the conditional-copy
behavior but still sends the full context. Every model file the Dockerfile bind-mounts must exist
in the checkout (BuildKit resolves bind-mount sources before any conditional logic runs), so run
`git lfs pull` before a direct build.

Optional packages are installed only when a backend needs them:

- `torch`/`torchvision` for `dan`, `mivolo`, `deep3d`, and/or `franunet`; `scipy` additionally
  for `deep3d` (to load `.mat` files).
- `tensorflow-cpu`/`tf-keras` (~200-400 MB) for `deepface`, `mini_xception`, `vggface`, and/or
  `mobilenetv2` (mask). The deepface weight files are 537 MB each, the heaviest options in the repo.
- `ultralytics`/`timm` for `mivolo`.
- `mediapipe` for face landmarks, liveness, or hand landmarks.
- `onnxruntime` for `yolo`/`scrfd`/`retinaface` (face detectors) or `mobilenet` (glasses).
- `opencv-contrib-python-headless` replaces the default `opencv-python-headless` only if `lbph`
  is requested (needed for `cv2.face`).

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

## Repository layout

```text
multimodal-face-analyzer/
├── Dockerfile, build-and-run.sh   # per-feature Docker build + guided wrapper
├── install-and-run.sh             # guided native install
├── requirements*.txt, pyproject.toml
├── models/                        # weights and configs (git-lfs)
├── src/
│   ├── app.py                     # Streamlit UI (layout, sidebar, tabs)
│   ├── inference.py               # facade re-exporting the modules below
│   ├── core/                      # constants (paths, fusion weights), types (Models), image utils
│   ├── detectors/                 # BaseFaceDetector + factory; SSD, YOLO, SCRFD, RetinaFace
│   ├── attributes/                # per-face predictors: age, gender, emotion, race, accessories, ...
│   ├── fusion/                    # combined answers: fuse_gender/race/emotion, select_age
│   ├── pipeline/                  # loader (load_models), analyzer (analyze_frame) + stages, drawing, tracker
│   ├── ui/                        # live webcam frame callback (make_video_frame_callback)
│   ├── gallery/                   # SQLite saved faces, eigenfaces, identity search
│   ├── liveness.py, model_selection.py
│   └── nets/                      # vendored third-party model architectures
├── tests/                         # pytest suite
├── tools/                         # benchmark.py, dump_predictions.py, score_fusion.py, ground_truth.json
└── docs/                          # this documentation
```

Runtime data is gitignored: `gallery/` (enrolled faces), `db/`, `faces/`, `eigen/` (saved faces),
and `known_people/` (identity-search reference photos).
