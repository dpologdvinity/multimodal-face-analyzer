# Public demo deployment

The app has a demo mode for running it as a free, public web demo. This page covers what the mode
changes, how to deploy it on Streamlit Community Cloud, and how to run it on a Docker host.

## Demo mode

Demo mode is on when the environment variable `FACE_ANALYZER_DEMO` is `1` (or `true`, `yes`,
`on`). The root `streamlit_app.py` entry point always turns it on. A Streamlit root-level secret
`FACE_ANALYZER_DEMO = "1"` also works, because Streamlit exports root-level secrets as
environment variables when the server starts. The model allowlist and limits are in
`src/face_analyzer/demo.py`.

Demo mode changes the following:

- **Nothing persists or is shared between visitors.** The saved-face SQLite database, SAVE,
  SEARCH (eigenfaces and identity search over `known_people/`), enrolled galleries and ENROLL,
  LBPH training, and SCAN ALL FACES are all off. Their sidebar sections and buttons are hidden,
  the gallery file is never read, and nothing is written to `faces/`, `db/`, `eigen/` or `gallery/`.
  As a second line of defense, `save_face`, `save_gallery` and `enroll_lbph_face` raise
  `PermissionError` in demo mode. Decoded uploads and per-face predictions are not cached either,
  so nothing derived from an image outlives the request that analyzed it.
- **No live webcam mode.** WebRTC needs a TURN server on most hosts, so the Webcam tab offers
  snapshots only, with a one-line note that live mode runs locally only.
- **A responsible-use note** at the top: the results are apparent attributes only, the models
  have known biases, the results must not be used for decisions about people, and uploads are
  processed in memory and never stored. A one-line credit for the demo's models (FairFace's
  CC BY 4.0 license requires attribution) sits at the bottom of the page.
- **No error details for visitors.** `client.showErrorDetails` is set to `none` at runtime, so a
  crash shows a generic message, while the server log keeps the traceback. This is set in code
  rather than in a `.streamlit/config.toml`, because Streamlit reads that file from the working
  directory, and a tracked one at the repo root would also hide tracebacks during local
  development and in the Docker image. `.streamlit/` stays gitignored for local settings.
- **Bounded memory.** Each upload is limited to 10 MB and the first 3 images are analyzed. Only
  JPEG, PNG and WebP files are decoded; any other file is rejected before decoding, whatever its
  extension, and so is any image whose header declares more than 50 megapixels. Images are
  downscaled to a longest side of 1600 px. Only one analysis runs at a time per process: another
  visitor's upload waits (up to 2 minutes) with a "being analyzed" note instead of adding its
  peak memory on top.
- **Only permissively licensed, lightweight models.** These are `hf` entries in
  `models/manifest.json` (mirrored on `kaitlynbassford/face-analyzer-weights`), and none needs
  torch or TensorFlow:

  | Feature | Backend | File | License |
  | ------- | ------- | ---- | ------- |
  | Face detection | `retinaface` (default detector) | `retinaface_mobilenet0.25.onnx` | MIT / Apache-2.0 |
  | Age, gender, race | `fairface` | `fairface_7class.onnx` | CC BY 4.0 |
  | Emotion | `ferplus` | `emotion_ferplus.onnx` | MIT |
  | Eye color, roll alignment | `colorimetric` | `haarcascade_eye.xml` | Intel (BSD-style) |
  | Hair color | `colorimetric` | none | - |
  | Colorization (grayscale input) | `eccv16` | `colorization_*`, `pts_in_hull.npy` | BSD-2-Clause |
  | Face landmarks, gaze | `mediapipe` | `face_landmarker.task` | Apache-2.0 |

  Face landmarks and gaze load only if the `mediapipe` package is installed. It is not in
  `requirements.txt`, so the Community Cloud demo runs without them (see
  [Dependencies](#dependencies)). The SSD detector is never loaded in demo mode, because its
  weights have no stated license.

  Without MediaPipe, FairFace aligns each face on RetinaFace's own eye and nose landmarks, which
  holds the demo within noise of the full app's `fairface` on held-out FairFace (gender 92.2%,
  race 76.0%, age bucket 59.4%; see [the demo row](eval/heldout_fairface.md#demo-configuration)).

## Streamlit Community Cloud

Community Cloud is free, gives each app about 2.7 GB of RAM and 2 CPU cores, and installs the
root `requirements.txt`. The steps below are for the repository owner.

1. Go to [share.streamlit.io](https://share.streamlit.io) and choose **Continue with GitHub**.
   Authorize Streamlit to see the `dpologdvinity` repositories when asked.
2. Click **Create app**, then choose to deploy a public app from GitHub.
3. Fill in the form:
   - **Repository:** `dpologdvinity/multimodal-face-analyzer`
   - **Branch:** `master`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** optional, a custom subdomain such as `face-analyzer-demo`.
4. Open **Advanced settings**:
   - **Python version:** `3.12`. CI tests 3.11 and 3.12, and the pins also resolve on 3.11.
   - **Secrets:** leave empty. `streamlit_app.py` turns demo mode on itself, and the weights
     come from a public Hugging Face repo, so no token is needed.
5. Click **Deploy**.
6. When the app is up, put its URL in the `LIVE DEMO` comment near the top of `README.md` as a
   visible link.

What happens on first boot:

- Community Cloud installs `requirements.txt` (with uv, falling back to pip). This usually takes
  a few minutes.
- The first page load downloads and sha256-checks the demo weights (8 files, 251 MB) into
  `models/` while a spinner is shown. On the development machine this took 25 s from Hugging
  Face. Model loading then takes a few seconds. Later page loads in the same container skip both
  steps, since a hash check of files already present takes well under a second.
- Community Cloud puts apps to sleep after a period without traffic, and a container restart
  starts again from a fresh checkout, so the next visitor after a restart waits for the
  download again.
- A push to `master` updates the checkout of the running process, but Streamlit re-executes only
  `streamlit_app.py`, and the already imported `face_analyzer` modules would keep the old code.
  On each run the entry point therefore compares the modification times of
  `src/face_analyzer/**/*.py` with those at import. When they differ, it drops the package's
  modules and clears `st.cache_resource` and `st.cache_data`, so the code and the models reload
  once. While an analysis is running it leaves the reload to a later page run, and the reloaded
  code keeps the same one-analysis slot. If the demo still behaves like the old code after a
  deploy, use **Reboot app** in the app's Community Cloud menu.

### Measured memory

Peak resident memory of the whole Streamlit process, in demo mode on Python 3.12, measured in
AppTest and confirmed against a real `streamlit run streamlit_app.py` server:

| Step | Peak RSS |
| ---- | -------- |
| Boot and load the demo models | 0.53-0.56 GB |
| Analyze a 1024 px color photo with 6 faces | 1.0 GB |
| Analyze a 3072 px grayscale photo with 6 faces (downscaled to 1600 px, then colorized) | 1.6 GB |

That leaves about 1 GB of headroom under the 2.7 GB limit. Models are loaded once per process
and shared by all sessions, and analyses run one at a time, so concurrent visitors add only
their own decoded images and rendered results, not another analysis peak. Colorization accounts
for about 0.55 GB of the peak. If the app hits the memory limit under real traffic, remove
`COLORIZATION_MODEL` from `DEMO_MODELS` in `src/face_analyzer/demo.py` first.

### Dependencies

- `requirements.txt` covers the demo set. Its one addition for the demo is `onnxruntime`, which
  RetinaFace needs. It adds only `onnxruntime` and `flatbuffers` to the resolved set, and the
  Dockerfile already installed it for the default build, so it is harmless for normal installs.
- `mediapipe` stays out. `mediapipe==1.1.0` pulls in `opencv-contrib-python` 5.x next to the
  pinned `opencv-python-headless` 4.14. Both install a `cv2` package, so whichever wins is
  arbitrary, and OpenCV 5 drops the Caffe importer that colorization needs. The Dockerfile and
  the guided installer fix this by reinstalling OpenCV last, which a single requirements file
  cannot do. mediapipe would also need extra system libraries (`packages.txt`).
- No `packages.txt` is needed: the headless OpenCV build needs no system GUI libraries.
- `streamlit_app.py` puts `src/` on `sys.path` instead of installing the package. A `.` or
  `-e .` line in `requirements.txt` resolves against the installer's working directory, which
  the Community Cloud docs do not specify. A regular (non-editable) install would also copy the
  package into site-packages, where `BASE_DIR` (and so `models/`) would no longer point at the
  checkout. The `sys.path` line works the same everywhere and needs no build step.

## Docker host

The Dockerfile's default build has onnxruntime and every demo weight except `ferplus`: its
`EMOTION_MODEL` default is `hsemotion`, which demo mode does not load. Add `ferplus` to the build,
then run the image with demo mode on:

```bash
docker build -t face-analyzer --build-arg EMOTION_MODEL=hsemotion,ferplus .
docker run -d -p 127.0.0.1:8501:8501 -e FACE_ANALYZER_DEMO=1 --name face_analyzer_demo face-analyzer
```

Demo mode loads only the allowlisted backends even if the image holds more weights. Because the
image also installs mediapipe, face landmarks and gaze are available there. For a smaller image,
build it with only the demo backends:

```bash
docker build -t face-analyzer-demo \
  --build-arg AGE_MODEL=fairface --build-arg GENDER_MODEL=fairface --build-arg RACE_MODEL=fairface \
  --build-arg EMOTION_MODEL=ferplus --build-arg RETINAFACE_MODEL=retinaface \
  --build-arg COLORIZATION_MODEL=eccv16 --build-arg FACE_LANDMARKS_MODEL=mediapipe \
  --build-arg LIVENESS_MODEL= --build-arg RECOGNITION_MODEL= --build-arg GLASSES_MODEL= \
  --build-arg MASK_MODEL= --build-arg HAND_MODEL= --build-arg RECONSTRUCTION_3D_MODEL= \
  --build-arg YOLO_FACE_MODEL= --build-arg SCRFD_FACE_MODEL= --build-arg AGE_PROGRESSION_MODEL= .
```

The build still downloads the SSD weights, because the fetch script always includes the required
files, but demo mode never loads them. Put the container behind HTTPS so browsers allow webcam
snapshots, and see the remote-access notes in [setup.md](setup.md#remote-access-trust-boundary).
