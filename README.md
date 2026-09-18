# MULTIMODAL FACE ANALYZER

![Python](https://img.shields.io/badge/Python-3.11-00ff66?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-00ff66?style=flat-square&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-00ff66?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-00ff66?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-00ff66?style=flat-square&logo=docker&logoColor=white)

> Computer vision pipeline for face detection with age, gender, race, emotion, expression (blendshapes), and drowsiness inference. A containerized Streamlit web app.

---

## Key Features

- **Multi-model face analysis:** face detection, age, gender, race, emotion, expression (blendshapes), and drowsiness -- most features have 2+ selectable model backends.
- **Docker-packaged Streamlit app:** `src/app.py`, all features including race and expression.
- **Build-time feature toggles:** disable any model at Docker build time to shrink the image (see [Docker](#docker-web-app)).
- **Graceful degradation:** any model missing at runtime (file or dependency not present) is skipped, not a crash -- the rest of the pipeline keeps working.
- **Cyberpunk web interface:** terminal-styled drag-and-drop Streamlit UI, plus snapshot/live webcam tabs.

---

## Models

Face detection is required; age, gender, race, emotion, expression, and drowsiness are each independently optional -- if a model's file(s) or dependencies aren't present, that feature is skipped and the rest still runs.

### Face Detection

| Backend         | Framework            | Output                  |
| --------------- | -------------------- | ----------------------- |
| SSD / ResNet-10 | TensorFlow (cv2.dnn) | bounding box (required) |

### Age

| Backend       | Framework         | Output                         |
| ------------- | ----------------- | ------------------------------ |
| `caffe`       | Caffe (cv2.dnn)   | bucketed range, e.g. `(25-32)` |
| `insightface` | ONNX (cv2.dnn)    | continuous age, e.g. `31`      |
| `ssrnet`      | PyTorch           | continuous age, e.g. `31`      |
| `fairface`    | ONNX (cv2.dnn)    | bucketed range, e.g. `20-29` (9 buckets) |
| `dex`         | Caffe (cv2.dnn)   | continuous age, e.g. `31` (expected value over 101 classes) |
| `mivolo`      | PyTorch/timm ViT  | continuous age, e.g. `31`      |

Checkbox per built model in the web app sidebar. Default: `caffe`.

`dex` (Deep EXpectation, Rothe et al. ICCV 2015) is a VGG-16 trained on IMDB-WIKI, a heavy age option (513MB caffemodel). **Research/academic-use license** (ETH Zurich, IMDB-WIKI-derived) -- not for commercial deployments without independent licensing.

`mivolo` (MiVOLO: Multi-input Transformer for Age/Gender, Apache 2.0, WildChlamydia/MiVOLO) is a vision transformer that runs in face-only mode (no body context in this pipeline), sharing one checkpoint with the gender backend. Accuracy is ~4.24 years age MAE (face-only mode); heaviest option (~110MB checkpoint plus ultralytics/timm dependencies). Web app only.

### Gender

| Backend            | Framework            | Output            |
| ------------------ | -------------------- | ----------------- |
| Levi & Hassner CNN | Caffe (cv2.dnn)       | `Male` / `Female` |
| `insightface`      | ONNX (cv2.dnn)        | `Male` / `Female` |
| `deepface`         | Keras/TensorFlow      | `Male` / `Female` |
| `fairface`         | ONNX (cv2.dnn)        | `Male` / `Female` |
| `mivolo`           | PyTorch/timm ViT      | `Male` / `Female` |

`insightface` shares one small ONNX file (`models/insightface_genderage.onnx`) with the insightface age backend -- one model, two feature outputs. **Non-commercial research-use-only license** (CelebA-derived); not for commercial deployments. `deepface` shares its VGGFace backbone code (`src/nets/deepface_common.py`) with the deepface race backend, but is a separate 537MB weight file (`models/deepface_gender.h5`) and needs TensorFlow like deepface race does. `fairface` shares one ONNX file (`models/fairface_7class.onnx`) across all three of age, gender, and race -- one model, three feature outputs (named `age_output`/`gender_output`/`race_output` in the same graph).

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
| `efficientnet`           | ONNX (cv2.dnn)    | 7 classes: angry, disgust, fear, happy, sad, surprise, neutral |
| `ferplus`                | ONNX (cv2.dnn)    | 8 classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt |
| `mini_xception`          | Keras/TensorFlow  | 7 classes: angry, disgust, fear, happy, sad, surprise, neutral |
| `dan` (DAN, AffectNet-7) | PyTorch           | 7 classes: neutral, happy, sad, surprise, fear, disgust, anger |

Note the class label order (and count) differs between backends -- each is tracked as a separate constant, never assumed to match. `mini_xception` is tiny (853KB, oarriaga/face_classification, MIT) but needs TensorFlow like the deepface models. `ferplus` is the official ONNX Model Zoo emotion model (MIT, 35MB, no extra framework -- pure cv2.dnn ONNX), used in place of a third-party PyTorch checkpoint for security reasons (no untrusted pickle deserialization).

### Expression (web app only)

| Backend        | Framework | Output                                                             |
| -------------- | --------- | ------------------------------------------------------------------ |
| `blendshapes`  | MediaPipe | Top 3 facial muscle coefficients, e.g. `mouthSmileLeft 0.82, jawOpen 0.15, browDownRight 0.09` |

Raw output of Google's MediaPipe Face Landmarker (Apache 2.0), a separate feature from Emotion. BlendShapes outputs 52 continuous facial-muscle-movement coefficients (e.g. mouthSmileLeft, browDownRight, jawOpen, eyeBlinkLeft, etc.) tracking individual facial movements, whereas Emotion backends predict discrete emotion classes (angry, happy, sad, etc.). There is no validated mapping from blendshapes to emotion labels, so Expression surfaces the raw top-N-scoring blendshape coefficients as-is.

### Recognition (web app only)

| Backend    | Framework        | Output                                             |
| ---------- | ---------------- | --------------------------------------------------- |
| `vggface`  | Keras/TensorFlow | enrolled identity name + similarity, e.g. `Alice (82%)`, or `UNKNOWN` |

Reuses the same VGGFace backbone as the deepface race/gender heads (`src/nets/deepface_common.py`), truncated to its 4096-d penultimate layer as a face embedding (`src/nets/deepface_recognition.py`) instead of a classification head -- same represent+verify shape as the original DeepFace paper. Embeddings are L2-normalized; identity is decided by cosine similarity against every enrolled face in the gallery, with a match only reported above `RECOGNITION_COSINE_THRESHOLD = 0.68` (deepface's own default VGG-Face verification threshold). Needs TensorFlow, like deepface race/gender.

Enrollment happens in the web app: under any detected face with a computed embedding, enter a name and click ENROLL. The gallery is stored as `gallery/known_faces.json` (one L2-normalized 4096-d vector per name), created on first enrollment and gitignored as runtime user data. It survives app restarts; Docker users should volume-mount `gallery/` (e.g. `-v $(pwd)/gallery:/app/gallery`) to persist enrollments across container restarts. Sidebar has a GALLERY section listing enrolled names with a delete button per entry.

### Drowsiness

| Backend                   | Framework | Output             |
| ------------------------- | --------- | ------------------ |
| Haar cascade eye detector | OpenCV    | `DROWSY` / `ALERT` |

Model provenance: DAN, SSR-Net, and DeepFace's race model are vendored research code (`src/nets/`). DAN and SSR-Net have no explicit upstream license file (research/educational use). DeepFace (race, gender, and recognition/`deepface_vgg.h5`) is MIT. FairFace's ONNX conversion is MIT (underlying dataset CC BY 4.0). InsightFace's model is non-commercial research use only (see Gender above).

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
│   ├── ssrnet_morph2.pth                        # age: ssrnet backend
│   ├── gender_deploy.prototxt / gender_net.caffemodel
│   ├── insightface_genderage.onnx               # age + gender: insightface backend
│   ├── mivolo_v2.safetensors / _config.json     # age + gender: mivolo backend
│   ├── dan_affecnet7.pth                        # emotion: dan backend
│   ├── efficientnet_b0_fer.onnx                 # emotion: efficientnet backend
│   ├── emotion_ferplus.onnx                     # emotion: ferplus backend
│   ├── mini_xception_fer.h5                     # emotion: mini_xception backend
│   ├── fairface_7class.onnx                     # age + gender + race: fairface backend
│   ├── deepface_race.h5                         # race: deepface backend
│   ├── deepface_gender.h5                       # gender: deepface backend
│   ├── deepface_vgg.h5                          # recognition: vggface backend
│   ├── face_landmarker.task                     # expression: blendshapes backend
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
        ├── deepface_gender.py
        ├── deepface_recognition.py
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

# needed for: ssrnet age, dan emotion
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# needed for: deepface race and/or deepface gender (heavy -- ~200-400MB)
pip install tensorflow-cpu tf-keras
```

Without torch/tensorflow installed, the corresponding models are automatically skipped (not an error).

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
  --build-arg AGE_MODEL=caffe,insightface,ssrnet,fairface,dex,mivolo \
  --build-arg GENDER_MODEL=caffe,insightface,deepface,fairface,mivolo \
  --build-arg EMOTION_MODEL=efficientnet,ferplus,mini_xception,dan \
  --build-arg DROWSINESS_MODEL=haarcascade \
  --build-arg RACE_MODEL=fairface,deepface \
  --build-arg EXPRESSION_MODEL=blendshapes \
  --build-arg RECOGNITION_MODEL=vggface \
  -t face-analyzer .
```

| Build arg           | Options (default first)          |
| -------------------- | -------------------------------- |
| `AGE_MODEL`        | `caffe`, `insightface`, `ssrnet`, `fairface`, `dex`, `mivolo` |
| `GENDER_MODEL`     | `caffe`, `insightface`, `deepface`, `fairface`, `mivolo` |
| `EMOTION_MODEL`    | `efficientnet`, `ferplus`, `mini_xception`, `dan` |
| `DROWSINESS_MODEL` | `haarcascade`                            |
| `RACE_MODEL`       | `fairface`, `deepface`                   |
| `EXPRESSION_MODEL` | `blendshapes`                            |
| `RECOGNITION_MODEL` | `vggface`                               |

Multiple models per feature (e.g. `AGE_MODEL=caffe,ssrnet`) can be built in together -- the web app sidebar shows a checkbox per built model, and checking more than one for the same feature runs and displays all of them at once.

Disabled model files never land in an image layer (BuildKit bind-mount + conditional copy). `torch`/`torchvision` (~200MB) are only installed if `ssrnet`, `dan`, and/or `mivolo` are requested. `tensorflow-cpu`/`tf-keras` (~200-400MB, plus deepface's 513MB weight file) are only installed if `deepface` is requested -- by far the heaviest single option in the repo (note: `mivolo` at ~110MB checkpoint plus ultralytics/timm dependencies is the second-heaviest, still much lighter than deepface's full stack). `mediapipe` is only installed if the `blendshapes` expression backend is requested.

### Run

```bash
docker run -d -p 8501:8501 --name face_analyzer_container face-analyzer
```

Then open `http://localhost:8501`.
