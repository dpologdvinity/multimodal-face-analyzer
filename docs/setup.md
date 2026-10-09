# Setup

How to install, build, and run the analyzer natively or in Docker, plus the full table of
Docker build arguments. For what each model does, see [models.md](models.md).

## Model weights

Weights are not stored in git. [`models/manifest.json`](../models/manifest.json) lists every model
file the app can load, with its size, sha256, license, the build args that select it, and where it
comes from. `tools/fetch_models.py` (standard library only) downloads the files you select into
`models/`, or into `FACE_ANALYZER_MODEL_DIR` when that is set. The complete downloadable set is
about 2.9 GB.

```bash
python tools/fetch_models.py --list                                   # every file, its source and build args
python tools/fetch_models.py --keys AGE_MODEL=fairface,caffe EMOTION_MODEL=hsemotion
python tools/fetch_models.py --keys ferplus scrfd                     # bare model keys work too
python tools/fetch_models.py --all
```

`--keys` takes the same `ARG=key[,key]` pairs as the Docker build args (an empty value selects
nothing) or bare model keys. The required files (the SSD face detector and the eye cascade) are
always included. Each file is downloaded to a `.part` file, checked against its sha256, and only
then moved into place. A file that already exists with the right hash is skipped, and an
interrupted download resumes where it stopped. A hash mismatch is an error (exit code 1) and the
download is discarded. An existing file with a different hash is left alone unless you pass
`--force`.

Each manifest entry has one of three sources, decided by its license
([MODEL_LICENSES.md](../MODEL_LICENSES.md)):

- `hf`: permissively licensed (MIT, Apache-2.0, BSD, CC BY) and mirrored, unchanged, in the
  [kaitlynbassford/face-analyzer-weights](https://huggingface.co/kaitlynbassford/face-analyzer-weights)
  Hugging Face repo, whose model card carries the attributions and license notices.
- `upstream`: restricted or unclear license, so not mirrored. It is downloaded from the original
  project's own GitHub release, raw file, Hugging Face repo, or website. SCRFD comes out of
  InsightFace's `buffalo_m.zip` release asset (`det_2.5g.onnx`), so selecting it downloads the
  276 MB archive to extract a 3 MB file.
- `manual`: no direct download exists (Google Drive or a registration form). The script prints
  where to get the file and where to put it, and the feature stays listed as unavailable until
  you do. These are `dan_affecnet7.pth` (`dan` emotion), `deep3d_recon_resnet50.pth` and
  `BFM/BFM_model_front.mat` (`deep3d` 3D reconstruction). A self-supplied file whose hash differs
  from the one this repo was tested with only triggers a warning.

A few small, permissively licensed files stay in git: `opencv_face_detector.pbtxt`,
`haarcascade_eye.xml`, `colorization_deploy_v2.prototxt`, `pts_in_hull.npy`,
`mivolo_v2_config.json`, and `BFM/similarity_Lm3D_all.mat` (the manifest's `"in_git": true`
entries). Every other file in `models/` is gitignored, including the required SSD weights
(`opencv_face_detector_uint8.pb`, 2.7 MB, no license stated upstream). The app needs at least one
face detector and SSD is the default one, so run `python tools/fetch_models.py --keys` (with no
keys it fetches only the required files) before the first start; the guided scripts and the
Docker build do this for you. Without SSD the app still starts if another detector loads (the
public demo uses RetinaFace only; see [deploy.md](deploy.md)).

Older checkouts stored the weights in git-lfs. The loader still treats an unpulled LFS pointer
file like a missing model.

### Upgrading a clone that has the git-lfs weights

**Back up `models/` before you pull this change.** Git deletes files from disk when a pull stops
tracking them, so the pull removes every weight the old commits tracked. Most can be downloaded
again with `tools/fetch_models.py`, but the bring-your-own files cannot. Make a hard-linked copy
first (no extra disk space), then copy back whatever you need:

```bash
cp -al models models.bak
git pull
cp -a models.bak/dan_affecnet7.pth models.bak/deep3d_recon_resnet50.pth models/   # whichever you have
python tools/fetch_models.py --keys AGE_MODEL=fairface ...                         # the rest
```

If you already pulled without a backup, `dan_affecnet7.pth` can be restored from the last commit
that tracked the weights, `c6fcc83` (the merge of PR #6), with git-lfs:

```bash
git lfs fetch origin c6fcc83 --include=models/dan_affecnet7.pth
git show c6fcc83:models/dan_affecnet7.pth | git lfs smudge > models/dan_affecnet7.pth
```

The same works for any other file of that commit. The `deep3d_recon_resnet50.pth` stored there is
a placeholder, not a working checkpoint (see its manifest note), so a working one only survives in
a backup. Switching an upgraded checkout back to an older commit and then forward again deletes the
weights the same way.

## Local setup

The base install covers the required SSD face detector and every OpenCV-only backend:

```bash
conda create -n vision_env python=3.11 -y
conda activate vision_env
pip install -r requirements.txt
pip install --no-deps -e .
python tools/fetch_models.py --keys    # the required SSD detector; add ARG=key pairs for more
streamlit run src/face_analyzer/app.py
```

The source is the installable `face_analyzer` package (`src/` layout); the editable install makes
it importable while keeping `models/` and the runtime data directories resolved from the
repository root.

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
pip install -r requirements-dev.txt   # also installs face_analyzer in editable mode
python -m pytest -q
ruff check .
```

Tests that need a real weight file skip when it is missing (or an old LFS pointer), so the suite
runs on a fresh clone. CI runs ruff and the test suite on Python 3.11 and 3.12 against such a
clone after fetching only the required files (`python tools/fetch_models.py --keys`), so the app
smoke tests still run. `tests/test_fetch_models.py` checks the manifest and the
fetch tool offline, including that every model path in `core/constants.py` has a manifest entry.

### Held-out evaluation

The FairFace evaluation ([eval/heldout_fairface.md](eval/heldout_fairface.md)) needs every
optional backend, so it uses its own Python 3.11 environment, `.venv-eval` (gitignored), with the
Dockerfile's pins:

```bash
uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval/bin/python -r requirements-dev.txt
uv pip install --python .venv-eval/bin/python --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match torch==2.14.1+cpu torchvision==0.29.1+cpu
uv pip install --python .venv-eval/bin/python onnxruntime==1.30.0 tensorflow-cpu==2.21.0 tf-keras==2.21.0
uv pip install --python .venv-eval/bin/python ultralytics==8.1.0 timm==0.8.13.dev0 safetensors==0.8.0 huggingface_hub==2.1.1
uv pip install --python .venv-eval/bin/python mediapipe==1.1.0
# ultralytics and mediapipe pull in OpenCV 5 builds without Caffe support; keep only the pin
uv pip uninstall --python .venv-eval/bin/python opencv-python opencv-contrib-python opencv-python-headless
uv pip install --python .venv-eval/bin/python opencv-python-headless==4.14.0.94
```

Then the two commands (the second takes about 2.5 hours at N=2,000 on two CPU cores, resumes
from its cache if interrupted, and rewrites `docs/eval/heldout_fairface.json`):

```bash
.venv-eval/bin/python tools/fetch_fairface.py
FACE_ANALYZER_MODEL_DIR=$PWD/models .venv-eval/bin/python tools/eval_heldout.py --n 2000
```

`--score-only` re-scores the cached outputs in `data/eval_cache/` without loading any model. It
still rewrites the tracked `docs/eval/heldout_fairface.json`, and `scored_on` and
`scoring_git_commit` change on every run. For checks that should leave the JSON alone, add
`--output <path>`.

`--detector` picks another face detector and `--no-mediapipe` leaves MediaPipe's landmarker
unloaded; [heldout_fairface.md](eval/heldout_fairface.md#demo-configuration) has the command for
the public demo's configuration. The eval caps ONNX Runtime at two threads per session through
`FACE_ANALYZER_ORT_THREADS`, which the app also honors; unset, ONNX Runtime starts one thread per
core and pins them, past any `taskset` limit.

`--extra-age KEY=model.onnx` also runs an age model that is not an app backend on the same
detections and aligned faces, cached separately in `data/eval_cache/<config>/extra_age_KEY.jsonl`;
the committed JSON includes `convnext` this way, so re-score it with
`--score-only --extra-age convnext`. `tools/eval_utkface.py` is the out-of-distribution
counterpart on UTKFace (non-commercial research only; download it with
`kaggle datasets download moritzm00/utkface-cropped -p data/utkface --unzip`) and writes
`docs/eval/age_model.json`; see [eval/age_model.md](eval/age_model.md).

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
selected dependency groups, fetches the selected weights with `tools/fetch_models.py`, and starts
the app at `http://localhost:8501`.

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
It passes the selections as build args (the image build downloads those weights), asks whether to
live-mount `src/` for testing, and then starts the container.
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

Eye Color is a colorimetric heuristic with no model file of its own and thus no build ARG
either -- it's available in the web app whenever `haarcascade_eye.xml` is present. Face
Landmarks uses `FACE_LANDMARKS_MODEL=mediapipe`; Liveness uses its own
`LIVENESS_MODEL=mediapipe` ARG while reusing the same model file.

Multiple models per feature (e.g. `AGE_MODEL=fairface,caffe`) can be built in together -- the web
app sidebar shows a checkbox per built model, and checking more than one for the same feature runs
and displays all of them at once.

The build downloads only the selected weights, with `tools/fetch_models.py` and the build args
as its `--keys`, so disabled models never land in an image layer and no local weights are needed.
`.dockerignore` keeps any weights you have in `models/` out of the build context, which stays a few
MB. A failed download or a sha256 mismatch fails the build. Selecting a bring-your-own model
(`dan`, `deep3d`) does not: the build prints where to get the file, the app lists the feature as
unavailable, and you can mount the file in at run time, for example
`-v "$PWD/models/dan_affecnet7.pth:/app/models/dan_affecnet7.pth:ro"`.

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
├── streamlit_app.py               # public demo entry point (demo mode + demo weights; see deploy.md)
├── models/                        # manifest.json + small configs; weights fetched on demand
├── src/face_analyzer/             # the installable face_analyzer package
│   ├── app.py                     # Streamlit entry point: page setup, model load, sidebar -> tabs wiring
│   ├── inference.py               # facade re-exporting the modules below
│   ├── core/                      # constants (paths, fusion weights, model rankings), types (Models), image utils
│   ├── detectors/                 # factory + backends: SSD, YOLO, SCRFD, RetinaFace
│   ├── attributes/                # per-face predictors: age, gender, emotion, race, accessories, ...
│   ├── fusion/                    # combined answers: fuse_emotion, select_age/gender/race
│   ├── pipeline/                  # loader (load_models), analyzer (analyze_frame), config (AnalysisConfig),
│   │                              #   stages, face_tasks, cache, landmarks, drawing, tracker
│   ├── ui/                        # Streamlit views: themes (+ theme.css tokens), sidebar, adjustments, results, live webcam tab
│   ├── gallery/                   # SQLite saved faces, eigenfaces, identity search
│   ├── liveness.py, model_selection.py, demo.py (demo-mode allowlist and limits)
│   └── nets/                      # vendored third-party model architectures
├── tests/                         # pytest suite
├── tools/                         # fetch_models.py, benchmark.py, dump_predictions.py, score_fusion.py, ground_truth.json,
│                                  #   fetch_fairface.py + eval_heldout.py (held-out FairFace eval),
│                                  #   fit_stacker.py (learned stacker, tested on that eval's cache)
│                                  #   eval_utkface.py (UTKFace age eval), kaggle/age_model/ (age model training kernels)
└── docs/                          # this documentation
```

Runtime data is gitignored: `gallery/` (enrolled faces), `db/`, `faces/`, `eigen/` (saved faces),
and `known_people/` (identity-search reference photos). The held-out evaluation's dataset and
output cache live in `data/`, and its environment in `.venv-eval/`; both are gitignored too.
