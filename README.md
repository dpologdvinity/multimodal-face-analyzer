# AGE & GENDER DETECTION MODEL

![Python](https://img.shields.io/badge/Python-3.11-00ff66?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-00ff66?style=flat-square&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-00ff66?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-00ff66?style=flat-square&logo=docker&logoColor=white)

> Computer vision pipeline for real-time face detection, age classification, and gender prediction. Offers a high-performance terminal CLI alongside a containerized Streamlit web application.

---

## Key Features

- **Dual Deployment:** Operates via terminal CLI (`detect.py`) or Docker-packaged Web App (`src/app.py`).
- **Batch Processing:** Handles individual image files or entire directories simultaneously.
- **Cropped Focus:** Optional output mode to isolate and display bounding-box face crops instead of full images.
- **Headless Friendly:** Full CLI support for non-GUI environments (WSL2, remote SSH servers, headless CI/CD).
- **Cyberpunk Web Interface:** Terminal-styled drag-and-drop Streamlit UI.

---

## Directory Layout

```text
age-gender-detection/
├── .dockerignore          # Docker build exclusions
├── Dockerfile             # Container configuration
├── README.md              # Documentation
├── detect.py              # CLI entry point
├── requirements.txt       # Dependencies
│
├── models/                # Pre-trained DNN weights & configs
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
│
└── src/                   # Web application module
    ├── __init__.py
    └── app.py             # Streamlit application
```

---

## Requirements & Local Setup

### 1. Conda Environment Setup

```bash
conda create -n vision_env python=3.11 -y
conda activate vision_env
pip install -r requirements.txt

```

---

## CLI Usage (`detect.py`)

Run inference directly from your terminal.

```bash
# Basic single-image detection
python detect.py path/to/image.jpg

# Process an entire folder and save annotated outputs
python detect.py path/to/folder/ --save

# Run headlessly (WSL/SSH) and save cropped face targets only
python detect.py path/to/folder/ --crop --save --no-show

# Custom output path and confidence threshold
python detect.py path/to/image.jpg --save --out-dir ./custom_results --conf 0.8

```

### CLI Flags

| Flag        | Short | Description                                                        |
| ----------- | ----- | ------------------------------------------------------------------ |
| `path`      |       | Path to target image file or directory                             |
| `--crop`    |       | Display/Save cropped face targets instead of full annotated frames |
| `--save`    |       | Export processed images to disk                                    |
| `--no-show` |       | Disable GUI display pop-ups (required for headless environments)   |
| `--out-dir` |       | Target directory for saved images (Default: `output`)              |
| `--conf`    |       | Minimum face detection confidence score (Default: `0.7`)           |

---

## Docker Web App (`src/app.py`)

Deploy the Streamlit web dashboard inside an isolated Docker container.

### 1. Build Docker Image

```bash
docker build -t age-gender-app .

```

### 2. Run Container

```bash
docker run -d -p 8501:8501 --name age_gender_container age-gender-app

```

### 3. Access Dashboard

Navigate to `http://localhost:8501` in your browser.

---

## Model Specifications

| Model                  | Framework  | Architecture                           | Input Size |
| ---------------------- | ---------- | -------------------------------------- | ---------- |
| **Face Detection**     | TensorFlow | Single Shot Detector (SSD) / ResNet-10 | 300 x 300  |
| **Age Classification** | Caffe      | Levi & Hassner CNN                     | 227 x 227  |
| **Gender Prediction**  | Caffe      | Levi & Hassner CNN                     | 227 x 227  |
