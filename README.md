# MULTIMODAL FACE ANALYZER

![Python](https://img.shields.io/badge/Python-3.11-00ff66?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-00ff66?style=flat-square&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-00ff66?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-00ff66?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-00ff66?style=flat-square&logo=docker&logoColor=white)

> Computer vision pipeline for face detection with age, gender, race, emotion, and drowsiness inference. Terminal CLI and a containerized Streamlit web app share the same detection pipeline.

---

## Key Features

- **Multi-model face analysis:** face detection, age, gender, race, emotion, and drowsiness -- most features have 2+ selectable model backends.
- **Dual deployment:** terminal CLI (`detect.py`, age/gender/emotion/drowsiness only) or Docker-packaged Streamlit app (`src/app.py`, all features including race).
- **Build-time feature toggles:** disable any model at Docker build time to shrink the image (see [Docker](#docker-web-app)).
- **Graceful degradation:** any model missing at runtime (file or dependency not present) is skipped, not a crash -- the rest of the pipeline keeps working.
- **Batch processing:** single image files or entire directories.
- **Headless friendly:** full CLI support for non-GUI environments (WSL2, remote SSH, headless CI/CD).
- **Cyberpunk web interface:** terminal-styled drag-and-drop Streamlit UI, plus snapshot/live webcam tabs.

---

## Models

Face detection is required; age, gender, race, emotion, and drowsiness are each independently optional -- if a model's file(s) or dependencies aren't present, that feature is skipped and the rest still runs. Race is web-app-only (no CLI equivalent).

### Face Detection

| Backend         | Framework            | Output                  |
| --------------- | -------------------- | ----------------------- |
| SSD / ResNet-10 | TensorFlow (cv2.dnn) | bounding box (required) |

### Age

| Backend       | Framework       | Output                         |
| ------------- | --------------- | ------------------------------ |
| `caffe`       | Caffe (cv2.dnn) | bucketed range, e.g. `(25-32)` |
| `insightface` | ONNX (cv2.dnn)  | continuous age, e.g. `31`      |
| `ssrnet`      | PyTorch         | continuous age, e.g. `31`      |

CLI: `--age-model` (`caffe` or `ssrnet` only). Web app: checkbox per built model. Default: `caffe`.

### Gender

| Backend            | Framework            | Output            |
| ------------------ | -------------------- | ----------------- |
| Levi & Hassner CNN | Caffe (cv2.dnn)       | `Male` / `Female` |
| `insightface`      | ONNX (cv2.dnn)        | `Male` / `Female` |
| `deepface`         | Keras/TensorFlow      | `Male` / `Female` |

`insightface` shares one small ONNX file (`models/insightface_genderage.onnx`) with the insightface age backend -- one model, two feature outputs. **Non-commercial research-use-only license** (CelebA-derived); not for commercial deployments. `deepface` shares its VGGFace backbone code (`src/nets/deepface_common.py`) with the deepface race backend, but is a separate 537MB weight file (`models/deepface_gender.h5`) and needs TensorFlow like deepface race does.

### Race (web app only)

| Backend    | Framework        | Output                                                                                        |
| ---------- | ---------------- | --------------------------------------------------------------------------------------------- |
| `fairface` | ONNX (cv2.dnn)   | 7 classes: White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern |
| `deepface` | Keras/TensorFlow | 6 classes: asian, indian, black, white, middle eastern, latino hispanic                       |

If the top-2 predicted classes are within 10 percentage points of each other, both are shown together (e.g. `White (52%)/Black (47%)`) instead of just the top class. `deepface` is by far the heaviest option in the repo: a 513MB weight file plus TensorFlow itself (~200-400MB) -- only pulled into the image if requested.

### Emotion

| Backend                  | Framework         | Output                                                         |
| ------------------------ | ----------------- | -------------------------------------------------------------- |
| `efficientnet`           | ONNX (cv2.dnn)    | 7 classes: angry, disgust, fear, happy, sad, surprise, neutral |
| `mini_xception`          | Keras/TensorFlow  | 7 classes: angry, disgust, fear, happy, sad, surprise, neutral |
| `dan` (DAN, AffectNet-7) | PyTorch           | 7 classes: neutral, happy, sad, surprise, fear, disgust, anger |

Note the class label order differs between backends -- each is tracked as a separate constant, never assumed to match. `mini_xception` is tiny (853KB, oarriaga/face_classification, MIT) but needs TensorFlow like the deepface models.

### Drowsiness

| Backend                   | Framework | Output             |
| ------------------------- | --------- | ------------------ |
| Haar cascade eye detector | OpenCV    | `DROWSY` / `ALERT` |

Model provenance: DAN, SSR-Net, and DeepFace's race model are vendored research code (`src/nets/`). DAN and SSR-Net have no explicit upstream license file (research/educational use). DeepFace is MIT. FairFace's ONNX conversion is MIT (underlying dataset CC BY 4.0). InsightFace's model is non-commercial research use only (see Gender above).

---

## Directory Layout

```text
multimodal-face-analyzer/
├── Dockerfile              # Container build (per-feature toggles, see below)
├── .dockerignore
├── build-and-run.sh        # Interactive build+run wrapper
├── detect.py               # CLI entry point
├── requirements.txt        # Base dependencies (opencv/streamlit/etc.)
│
├── models/                 # Pre-trained weights & configs (data, not code)
│   ├── opencv_face_detector_uint8.pb / .pbtxt   # face detection (required)
│   ├── age_deploy.prototxt / age_net.caffemodel # age: caffe backend
│   ├── ssrnet_morph2.pth                        # age: ssrnet backend
│   ├── gender_deploy.prototxt / gender_net.caffemodel
│   ├── insightface_genderage.onnx               # age + gender: insightface backend
│   ├── dan_affecnet7.pth                        # emotion: dan backend
│   ├── efficientnet_b0_fer.onnx                 # emotion: efficientnet backend
│   ├── mini_xception_fer.h5                     # emotion: mini_xception backend
│   ├── fairface_7class.onnx                     # race: fairface backend
│   ├── deepface_race.h5                         # race: deepface backend
│   ├── deepface_gender.h5                       # gender: deepface backend
│   └── haarcascade_eye.xml                      # drowsiness
│
└── src/                    # Streamlit app module
    ├── app.py              # UI only (page layout, sidebar, tabs)
    ├── inference.py        # model loading + per-face prediction logic
    └── nets/               # vendored model architectures (code, not weights)
        ├── dan_model.py
        ├── ssrnet_model.py
        ├── deepface_common.py                   # shared VGGFace backbone
        ├── deepface_race.py
        ├── mini_xception_model.py
        └── deepface_gender.py
```

`detect.py` (CLI) and `src/app.py`+`src/inference.py` (web app) intentionally duplicate the detection pipeline rather than sharing one module.

---

## Local Setup

```bash
conda create -n vision_env python=3.11 -y
conda activate vision_env
pip install -r requirements.txt

# needed for: ssrnet age, dan emotion
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# needed for: deepface race and/or deepface gender (heavy -- ~200-400MB)
pip install tensorflow-cpu tf-keras
```

Without torch/tensorflow installed, the corresponding models are automatically skipped (not an error).

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
| ------------- | --------------------------------------------------------------------------------- |
| `path`        | Path to target image file or directory                                            |
| `--crop`      | Display/save cropped face targets instead of full annotated frames                |
| `--save`      | Export processed images to disk                                                   |
| `--no-show`   | Disable GUI display pop-ups (required for headless environments)                  |
| `--out-dir`   | Target directory for saved images (default: `output`)                             |
| `--conf`      | Minimum face detection confidence score (default: `0.7`)                          |
| `--age-model` | Age backend: `caffe` (bucketed ranges) or `ssrnet` (continuous, default: `caffe`) |

The CLI does not have `insightface`, `efficientnet`, or race classification -- those are web-app-only (see Models above).

---

## Docker Web App

### Build

```bash
# guided prompt -- numbered options per feature, comma-separated multi-select
./build-and-run.sh

# or manually, default: quickest-to-build option per feature
docker build -t face-analyzer .
```

Each build ARG takes a comma-separated list of model keys for that feature, or empty string for none (each defaults to its quickest-to-build option):

```bash
docker build \
  --build-arg AGE_MODEL=caffe,insightface,ssrnet \
  --build-arg GENDER_MODEL=caffe,insightface,deepface \
  --build-arg EMOTION_MODEL=efficientnet,mini_xception,dan \
  --build-arg DROWSINESS_MODEL=haarcascade \
  --build-arg RACE_MODEL=fairface,deepface \
  -t face-analyzer .
```

| Build arg          | Options (default first)          |
| ------------------ | -------------------------------- |
| `AGE_MODEL`        | `caffe`, `insightface`, `ssrnet` |
| `GENDER_MODEL`     | `caffe`, `insightface`, `deepface` |
| `EMOTION_MODEL`    | `efficientnet`, `mini_xception`, `dan` |
| `DROWSINESS_MODEL` | `haarcascade`                    |
| `RACE_MODEL`       | `fairface`, `deepface`           |

Multiple models per feature (e.g. `AGE_MODEL=caffe,ssrnet`) can be built in together -- the web app sidebar shows a checkbox per built model, and checking more than one for the same feature runs and displays all of them at once.

Disabled model files never land in an image layer (BuildKit bind-mount + conditional copy). `torch`/`torchvision` (~200MB) are only installed if `ssrnet` and/or `dan` are requested. `tensorflow-cpu`/`tf-keras` (~200-400MB, plus deepface's 513MB weight file) are only installed if `deepface` is requested -- by far the heaviest single option in the repo.

### Run

```bash
docker run -d -p 8501:8501 --name face_analyzer_container face-analyzer
```

Then open `http://localhost:8501`.
