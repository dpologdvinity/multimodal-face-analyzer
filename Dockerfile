# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Toggle which detection features get built in. Each defaults to on (matches
# running the app with all models present). Disabling a feature skips copying
# its model file(s) into the image (via bind-mount, so the bytes never land in
# a layer) and, for emotion, skips installing torch/torchvision.
ARG INCLUDE_AGE=true
ARG INCLUDE_AGE_SSRNET=true
ARG INCLUDE_GENDER=true
ARG INCLUDE_DROWSINESS=true
ARG INCLUDE_EMOTION=true

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages (opencv/streamlit etc. -- shared by all features)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# torch/torchvision are only needed for emotion classification and/or the SSR-Net age backend; skip the ~200MB install otherwise
RUN if [ "$INCLUDE_EMOTION" = "true" ] || [ "$INCLUDE_AGE_SSRNET" = "true" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    fi

# Application code and always-required model files (face detector)
COPY detect.py ./
COPY src/ src/
COPY models/opencv_face_detector.pbtxt models/opencv_face_detector_uint8.pb models/

# Per-feature model files: bind-mount the source so disabled files are never written into a layer
RUN --mount=type=bind,source=models/age_deploy.prototxt,target=/tmp/models/age_deploy.prototxt \
    --mount=type=bind,source=models/age_net.caffemodel,target=/tmp/models/age_net.caffemodel \
    --mount=type=bind,source=models/ssrnet_morph2.pth,target=/tmp/models/ssrnet_morph2.pth \
    --mount=type=bind,source=models/gender_deploy.prototxt,target=/tmp/models/gender_deploy.prototxt \
    --mount=type=bind,source=models/gender_net.caffemodel,target=/tmp/models/gender_net.caffemodel \
    --mount=type=bind,source=models/haarcascade_eye.xml,target=/tmp/models/haarcascade_eye.xml \
    --mount=type=bind,source=models/dan_affecnet7.pth,target=/tmp/models/dan_affecnet7.pth \
    set -e; \
    if [ "$INCLUDE_AGE" = "true" ]; then cp /tmp/models/age_deploy.prototxt /tmp/models/age_net.caffemodel models/; fi; \
    if [ "$INCLUDE_AGE_SSRNET" = "true" ]; then cp /tmp/models/ssrnet_morph2.pth models/; fi; \
    if [ "$INCLUDE_GENDER" = "true" ]; then cp /tmp/models/gender_deploy.prototxt /tmp/models/gender_net.caffemodel models/; fi; \
    if [ "$INCLUDE_DROWSINESS" = "true" ]; then cp /tmp/models/haarcascade_eye.xml models/; fi; \
    if [ "$INCLUDE_EMOTION" = "true" ]; then cp /tmp/models/dan_affecnet7.pth models/; fi

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit web app from src directory
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
