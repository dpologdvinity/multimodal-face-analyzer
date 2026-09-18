# MULTIMODAL FACE ANALYZER

![Python](https://img.shields.io/badge/Python-3.11-00ff66?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-00ff66?style=flat-square&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-00ff66?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-00ff66?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-00ff66?style=flat-square&logo=docker&logoColor=white)

> Computer vision pipeline for face detection with age, gender, emotion, and drowsiness inference. Terminal CLI and a containerized Streamlit web app share the same detection pipeline.

---

## Key Features

- **Multi-model face analysis:** face detection, age (two selectable backends), gender, emotion (7-class), and drowsiness (eyes-closed) in one pass.
- **Dual deployment:** terminal CLI (`detect.py`) or Docker-packaged Streamlit app (`src/app.py`).
- **Build-time feature toggles:** disable any model at Docker build time to shrink the image (see [Docker](#docker-web-app)).
- **Graceful degradation:** any model missing at runtime (file or dependency not present) is skipped, not a crash -- the rest of the pipeline keeps working.
- **Batch processing:** single image files or entire directories.
- **Headless friendly:** full CLI support for non-GUI environments (WSL2, remote SSH, headless CI/CD).
- **Cyberpunk web interface:** terminal-styled drag-and-drop Streamlit UI, plus snapshot/live webcam tabs.

---

## Models

Face detection is required; age, gender, emotion, and drowsiness are each independently optional -- if a model's file(s) or dependencies aren't present, that feature is skipped and the rest still runs.

### Face Detection

| Backend          | Framework             | Output                    |
| ----------------- | ---------------------- | -------------------------- |
| SSD / ResNet-10   | TensorFlow (cv2.dnn)   | bounding box (required)    |

### Age

| Backend  | Framework       | Output                          |
| -------- | ---------------- | -------------------------------- |
| `caffe`  | Caffe (cv2.dnn)  | bucketed range, e.g. `(25-32)`   |
| `ssrnet` | PyTorch          | continuous age, e.g. `31`        |

Select with `--age-model` (CLI) or the sidebar dropdown (web app). Default: `caffe`.

### Gender

| Backend             | Framework       | Output              |
| -------------------- | ---------------- | -------------------- |
| Levi & Hassner CNN   | Caffe (cv2.dnn)  | `Male` / `Female`    |

### Emotion

| Backend                                        | Framework | Output                                                                 |
| ------------------------------------------------ | ---------- | ------------------------------------------------------------------------ |
| DAN ("Distract Your Attention", AffectNet-7)     | PyTorch   | one of 7 classes: neutral, happy, sad, surprise, fear, disgust, anger    |

### Drowsiness

| Backend                     | Framework | Output              |
| ----------------------------- | ---------- | -------------------- |
| Haar cascade eye detector     | OpenCV    | `DROWSY` / `ALERT`   |

Model provenance: DAN and SSR-Net are vendored research code (`src/nets/`) with no explicit upstream license file -- used here for research/educational purposes.

---

## Directory Layout

```text
multimodal-face-analyzer/
├── Dockerfile              # Container build (per-feature toggles, see below)
├── .dockerignore
├── detect.py               # CLI entry point
├── requirements.txt        # Base dependencies (opencv/streamlit/etc.)
│
├── models/                 # Pre-trained weights & configs (data, not code)
│   ├── opencv_face_detector_uint8.pb / .pbtxt   # face detection (required)
│   ├── age_deploy.prototxt / age_net.caffemodel # age: caffe backend
│   ├── ssrnet_morph2.pth                        # age: ssrnet backend
│   ├── gender_deploy.prototxt / gender_net.caffemodel
│   ├── dan_affecnet7.pth                        # emotion
│   └── haarcascade_eye.xml                      # drowsiness
│
└── src/                    # Streamlit app module
    ├── app.py              # UI only (page layout, sidebar, tabs)
    ├── inference.py        # model loading + per-face prediction logic
    └── nets/                # vendored model architectures (code, not weights)
        ├── dan_model.py
        └── ssrnet_model.py
```

`detect.py` (CLI) and `src/app.py`+`src/inference.py` (web app) intentionally duplicate the detection pipeline rather than sharing one module -- see `CLAUDE.md` for the reasoning.

---

## Local Setup

```bash
conda create -n vision_env python=3.11 -y
conda activate vision_env
pip install -r requirements.txt

# emotion classification and the SSR-Net age backend also need torch/torchvision:
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

Without torch installed, emotion and the `ssrnet` age backend are automatically skipped (not an error).

---

## CLI Usage (`detect.py`)

```bash
# Basic single-image detection
python detect.py path/to/image.jpg

# Process an entire folder and save annotated outputs
python detect.py path/to/folder/ --save

# Run headlessly (WSL/SSH) and save cropped face targets only
python detect.py path/to/folder/ --crop --save --no-show

# Use the SSR-Net continuous age model instead of the default bucketed Caffe model
python detect.py path/to/image.jpg --age-model ssrnet

# Custom output path and confidence threshold
python detect.py path/to/image.jpg --save --out-dir ./custom_results --conf 0.8
```

### CLI Flags

| Flag          | Description                                                                       |
| ------------- | ---------------------------------------------------------------------------------- |
| `path`        | Path to target image file or directory                                             |
| `--crop`      | Display/save cropped face targets instead of full annotated frames                 |
| `--save`      | Export processed images to disk                                                    |
| `--no-show`   | Disable GUI display pop-ups (required for headless environments)                   |
| `--out-dir`   | Target directory for saved images (default: `output`)                              |
| `--conf`      | Minimum face detection confidence score (default: `0.7`)                           |
| `--age-model` | Age backend: `caffe` (bucketed ranges) or `ssrnet` (continuous, default: `caffe`)   |

---

## Docker Web App

### Build

```bash
# default: every feature included
docker build -t face-analyzer .
```

Disable specific features at build time to shrink the image (each `ARG` defaults to `true`):

```bash
docker build \
  --build-arg INCLUDE_AGE=false \
  --build-arg INCLUDE_AGE_SSRNET=false \
  --build-arg INCLUDE_DROWSINESS=false \
  --build-arg INCLUDE_EMOTION=false \
  -t face-analyzer:gender-only .
```

| Build arg            | Controls                          |
| --------------------- | ----------------------------------- |
| `INCLUDE_AGE`         | Caffe age model (bucketed ranges)   |
| `INCLUDE_AGE_SSRNET`  | SSR-Net age model (continuous)      |
| `INCLUDE_GENDER`      | Gender model                        |
| `INCLUDE_DROWSINESS`  | Haar cascade eye detector           |
| `INCLUDE_EMOTION`     | DAN emotion model                   |

Disabled model files never land in an image layer (BuildKit bind-mount + conditional copy). `torch`/`torchvision` (~200MB) are only installed if `INCLUDE_EMOTION=true` or `INCLUDE_AGE_SSRNET=true`.

### Run

```bash
docker run -d -p 8501:8501 --name face_analyzer_container face-analyzer
```

Then open `http://localhost:8501`.
