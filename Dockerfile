# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Per-feature model selection. Each ARG takes a comma-separated list of model
# keys for that feature, or an empty string for "none". Options, in
# quickest-to-build order (default is the first/quickest):
#   AGE_MODEL:        caffe, insightface, ssrnet, fairface, dex, mivolo (default: caffe)
#   GENDER_MODEL:      caffe, insightface, deepface, fairface, mivolo (default: caffe)
#   EMOTION_MODEL:     efficientnet, ferplus, mini_xception, dan (default: efficientnet)
#   DROWSINESS_MODEL:  haarcascade                  (default: haarcascade)
#   RACE_MODEL:        fairface, deepface           (default: fairface)
#   EXPRESSION_MODEL:  blendshapes                  (default: blendshapes)
# insightface's genderage.onnx provides BOTH age and gender from one file
# (non-commercial research license -- see README). deepface's race model
# needs TensorFlow (~200-400MB) and a 513MB weight file, much heavier
# than fairface -- only pulled in if requested.
# e.g. --build-arg AGE_MODEL=caffe,ssrnet builds both age backends so the web
# app can switch between them at runtime. See build-and-run.sh for a guided
# prompt instead of typing these by hand.
ARG AGE_MODEL=caffe
ARG GENDER_MODEL=caffe
ARG EMOTION_MODEL=efficientnet
ARG DROWSINESS_MODEL=haarcascade
ARG RACE_MODEL=fairface
ARG EXPRESSION_MODEL=blendshapes

# Install system dependencies for OpenCV and MediaPipe (libegl1 needed by mediapipe's
# face landmarker even in CPU-only/headless use)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages (opencv/streamlit etc. -- shared by all features)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# torch/torchvision are needed for ssrnet, dan, and/or mivolo models
RUN age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; emotion_csv=",$EMOTION_MODEL,"; need_torch=false; \
    case "$age_csv" in *,ssrnet,*) need_torch=true ;; esac; \
    case "$age_csv" in *,mivolo,*) need_torch=true ;; esac; \
    case "$gender_csv" in *,mivolo,*) need_torch=true ;; esac; \
    case "$emotion_csv" in *,dan,*) need_torch=true ;; esac; \
    if [ "$need_torch" = "true" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    fi

# tensorflow/tf-keras are only needed for the deepface race, deepface gender,
# and/or mini_xception emotion models
RUN race_csv=",$RACE_MODEL,"; gender_csv=",$GENDER_MODEL,"; emotion_csv=",$EMOTION_MODEL,"; need_tf=false; \
    case "$race_csv" in *,deepface,*) need_tf=true ;; esac; \
    case "$gender_csv" in *,deepface,*) need_tf=true ;; esac; \
    case "$emotion_csv" in *,mini_xception,*) need_tf=true ;; esac; \
    if [ "$need_tf" = "true" ]; then pip install --no-cache-dir tensorflow-cpu tf-keras; fi

# MiVOLO dependencies (ultralytics, timm) are only needed for the mivolo age and/or gender models
RUN age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; need_mivolo=false; \
    case "$age_csv" in *,mivolo,*) need_mivolo=true ;; esac; \
    case "$gender_csv" in *,mivolo,*) need_mivolo=true ;; esac; \
    if [ "$need_mivolo" = "true" ]; then \
        pip install --no-cache-dir ultralytics==8.1.0 timm==0.8.13.dev0 safetensors huggingface_hub; \
    fi

# mediapipe is only needed for the blendshapes expression model
RUN expression_csv=",$EXPRESSION_MODEL,"; need_mediapipe=false; \
    case "$expression_csv" in *,blendshapes,*) need_mediapipe=true ;; esac; \
    if [ "$need_mediapipe" = "true" ]; then \
        pip install --no-cache-dir mediapipe; \
    fi

# ultralytics (mivolo) and/or mediapipe pull in their own opencv-python build as a
# transitive dependency, silently upgrading past the <5.0.0 ceiling in requirements.txt
# and breaking Caffe model support (caffe/dex age, caffe gender, haarcascade drowsiness
# all use cv2.dnn.readNetFromCaffe/CascadeClassifier, removed in OpenCV 5.0). Re-pin last.
RUN pip install --no-cache-dir "opencv-python-headless>=4.8.0,<5.0.0"

# Application code and always-required model files (face detector)
COPY detect.py ./
COPY src/ src/
COPY models/opencv_face_detector.pbtxt models/opencv_face_detector_uint8.pb models/

# Per-feature model files: bind-mount the source so files for unselected models are never written into a layer
RUN --mount=type=bind,source=models/age_deploy.prototxt,target=/tmp/models/age_deploy.prototxt \
    --mount=type=bind,source=models/age_net.caffemodel,target=/tmp/models/age_net.caffemodel \
    --mount=type=bind,source=models/ssrnet_morph2.pth,target=/tmp/models/ssrnet_morph2.pth \
    --mount=type=bind,source=models/gender_deploy.prototxt,target=/tmp/models/gender_deploy.prototxt \
    --mount=type=bind,source=models/gender_net.caffemodel,target=/tmp/models/gender_net.caffemodel \
    --mount=type=bind,source=models/insightface_genderage.onnx,target=/tmp/models/insightface_genderage.onnx \
    --mount=type=bind,source=models/haarcascade_eye.xml,target=/tmp/models/haarcascade_eye.xml \
    --mount=type=bind,source=models/dan_affecnet7.pth,target=/tmp/models/dan_affecnet7.pth \
    --mount=type=bind,source=models/efficientnet_b0_fer.onnx,target=/tmp/models/efficientnet_b0_fer.onnx \
    --mount=type=bind,source=models/emotion_ferplus.onnx,target=/tmp/models/emotion_ferplus.onnx \
    --mount=type=bind,source=models/fairface_7class.onnx,target=/tmp/models/fairface_7class.onnx \
    --mount=type=bind,source=models/deepface_race.h5,target=/tmp/models/deepface_race.h5 \
    --mount=type=bind,source=models/deepface_gender.h5,target=/tmp/models/deepface_gender.h5 \
    --mount=type=bind,source=models/mini_xception_fer.h5,target=/tmp/models/mini_xception_fer.h5 \
    --mount=type=bind,source=models/dex_age.prototxt,target=/tmp/models/dex_age.prototxt \
    --mount=type=bind,source=models/dex_age.caffemodel,target=/tmp/models/dex_age.caffemodel \
    --mount=type=bind,source=models/mivolo_v2.safetensors,target=/tmp/models/mivolo_v2.safetensors \
    --mount=type=bind,source=models/mivolo_v2_config.json,target=/tmp/models/mivolo_v2_config.json \
    --mount=type=bind,source=models/face_landmarker.task,target=/tmp/models/face_landmarker.task \
    set -e; \
    age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; emotion_csv=",$EMOTION_MODEL,"; \
    drowsiness_csv=",$DROWSINESS_MODEL,"; race_csv=",$RACE_MODEL,"; expression_csv=",$EXPRESSION_MODEL,"; \
    case "$age_csv" in *,caffe,*) cp /tmp/models/age_deploy.prototxt /tmp/models/age_net.caffemodel models/ ;; esac; \
    case "$age_csv" in *,ssrnet,*) cp /tmp/models/ssrnet_morph2.pth models/ ;; esac; \
    case "$gender_csv" in *,caffe,*) cp /tmp/models/gender_deploy.prototxt /tmp/models/gender_net.caffemodel models/ ;; esac; \
    case "$age_csv" in *,insightface,*) cp /tmp/models/insightface_genderage.onnx models/ ;; esac; \
    case "$gender_csv" in *,insightface,*) cp /tmp/models/insightface_genderage.onnx models/ ;; esac; \
    case "$age_csv" in *,fairface,*) cp /tmp/models/fairface_7class.onnx models/ ;; esac; \
    case "$gender_csv" in *,fairface,*) cp /tmp/models/fairface_7class.onnx models/ ;; esac; \
    case "$age_csv" in *,dex,*) cp /tmp/models/dex_age.prototxt /tmp/models/dex_age.caffemodel models/ ;; esac; \
    case "$gender_csv" in *,deepface,*) cp /tmp/models/deepface_gender.h5 models/ ;; esac; \
    case "$emotion_csv" in *,dan,*) cp /tmp/models/dan_affecnet7.pth models/ ;; esac; \
    case "$emotion_csv" in *,efficientnet,*) cp /tmp/models/efficientnet_b0_fer.onnx models/ ;; esac; \
    case "$emotion_csv" in *,ferplus,*) cp /tmp/models/emotion_ferplus.onnx models/ ;; esac; \
    case "$emotion_csv" in *,mini_xception,*) cp /tmp/models/mini_xception_fer.h5 models/ ;; esac; \
    case "$drowsiness_csv" in *,haarcascade,*) cp /tmp/models/haarcascade_eye.xml models/ ;; esac; \
    case "$race_csv" in *,fairface,*) cp /tmp/models/fairface_7class.onnx models/ ;; esac; \
    case "$race_csv" in *,deepface,*) cp /tmp/models/deepface_race.h5 models/ ;; esac; \
    case "$age_csv" in *,mivolo,*) cp /tmp/models/mivolo_v2.safetensors /tmp/models/mivolo_v2_config.json models/ ;; esac; \
    case "$gender_csv" in *,mivolo,*) cp /tmp/models/mivolo_v2.safetensors /tmp/models/mivolo_v2_config.json models/ ;; esac; \
    case "$expression_csv" in *,blendshapes,*) cp /tmp/models/face_landmarker.task models/ ;; esac

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit web app from src directory
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
