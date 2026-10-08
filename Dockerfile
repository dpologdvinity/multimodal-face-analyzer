# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Per-feature model selection. Each ARG takes a comma-separated list of model
# keys for that feature, or an empty string for "none". Options are listed
# most-accurate-first, as measured by tools/benchmark.py, and the default is
# that first (most accurate) key:
#   AGE_MODEL:        fairface, caffe, dex, mivolo (default: fairface)
#   GENDER_MODEL:      fairface, caffe, deepface, mivolo (default: fairface)
#   EMOTION_MODEL:     hsemotion, ferplus, mini_xception, dan (default: hsemotion)
#   RACE_MODEL:        fairface, deepface           (default: fairface)
#   FACE_LANDMARKS_MODEL: mediapipe                 (default: mediapipe)
#   LIVENESS_MODEL:    mediapipe                     (default: mediapipe)
#   RECOGNITION_MODEL: vggface, lbph                 (default: vggface)
#   GLASSES_MODEL:     mobilenet                     (default: mobilenet)
#   MASK_MODEL:        mobilenetv2                   (default: mobilenetv2)
#   COLORIZATION_MODEL: eccv16                        (default: eccv16)
#   HAND_MODEL:         mediapipe                       (default: mediapipe)
#   RECONSTRUCTION_3D_MODEL: deep3d                      (default: deep3d)
#   YOLO_FACE_MODEL:    yolo                             (default: yolo)
#   SCRFD_FACE_MODEL:   scrfd                            (default: scrfd)
#   RETINAFACE_MODEL:   retinaface                       (default: retinaface)
#   AGE_PROGRESSION_MODEL: franunet                       (default: franunet)
# YOLO_FACE_MODEL, SCRFD_FACE_MODEL, and RETINAFACE_MODEL are all additive, not a replacement --
# the original SSD/ResNet-10 TensorFlow detector is always required and always on; these ARGs
# only control whether the alternative YOLOv8-Face/SCRFD/RetinaFace ONNX files are ALSO built
# in, selectable at runtime via a sidebar dropdown (exactly one detector runs per frame). All
# three need onnxruntime (not this repo's usual cv2.dnn ONNX path -- cv2.dnn cannot parse the
# YOLOv8-Face export, verified against both OpenCV 4.10 and 5.0; SCRFD's and RetinaFace's own
# multi-output anchor formats are likewise handled via onnxruntime for the same "one non-cv2.dnn
# ONNX code path" consistency). SCRFD's weights (deepinsight/insightface's 2.5GF bnkps
# checkpoint) are non-commercial research-only. RetinaFace's weights (biubug6/Pytorch_Retinaface's mobilenet0.25 backbone,
# MIT-licensed, re-exported by AMD's Ryzen AI model zoo under Apache 2.0) are the only
# face-detector option in this repo with an unambiguous permissive license -- see README.
# lbph (Local Binary Patterns Histogram, opencv-contrib's cv2.face module) needs
# opencv-contrib-python-headless instead of opencv-python-headless -- see the final opencv
# reinstall step below. Unlike vggface, it has no pretrained weights: it trains from scratch
# on whatever's enrolled via the ENROLL button, same "trains fresh on demand" spirit as this
# app's eigenfaces feature.
# Face Landmarks and liveness use the same face_landmarker.task file, selected independently.
# Liveness uses LIVENESS_MODEL=mediapipe with that same file and dependency.
# RECONSTRUCTION_3D_MODEL wires the code path (torch/torchvision/scipy + the small bundled
# BFM landmark template) but ships NO working weights -- Deep3DFaceRecon_pytorch's checkpoint
# and the Basel Face Model data it needs are both gated (Google Drive / university license
# registration respectively); this ARG alone will never produce a working reconstruction.
# EMOTION_MODEL=dan is likewise bring-your-own (its checkpoint is only on Google Drive).
# tools/fetch_models.py prints where to get each such file; see docs/setup.md.
# AGE_PROGRESSION_MODEL (franunet, timroelofs123/face_reaging) needs torch. Its BlurPool
# component is vendored directly into src/face_analyzer/nets/face_reaging_model.py from Adobe's
# antialiased-cnns, which is CC BY-NC-SA 4.0 (non-commercial) -- a required inference-time
# dependency, not just a training-data provenance caveat like this repo's other NC-flagged
# models. See README.
# deepface's race model needs TensorFlow (~200-400MB) and a 513MB weight file, much heavier
# than fairface -- only pulled in if requested. mask also needs TensorFlow
# (Keras .h5 weights); glasses are plain ONNX.
# e.g. --build-arg AGE_MODEL=fairface,caffe builds both age backends so the web
# app can switch between them at runtime. See build-and-run.sh for a guided
# prompt instead of typing these by hand.
ARG AGE_MODEL=fairface
ARG GENDER_MODEL=fairface
ARG EMOTION_MODEL=hsemotion
ARG RACE_MODEL=fairface
ARG FACE_LANDMARKS_MODEL=mediapipe
ARG LIVENESS_MODEL=mediapipe
ARG RECOGNITION_MODEL=vggface
ARG GLASSES_MODEL=mobilenet
ARG MASK_MODEL=mobilenetv2
ARG COLORIZATION_MODEL=eccv16
ARG HAND_MODEL=mediapipe
ARG RECONSTRUCTION_3D_MODEL=deep3d
ARG YOLO_FACE_MODEL=yolo
ARG SCRFD_FACE_MODEL=scrfd
ARG RETINAFACE_MODEL=retinaface
ARG AGE_PROGRESSION_MODEL=franunet

# Install system dependencies for OpenCV and MediaPipe (libegl1/libgles2 needed by
# mediapipe's face landmarker even in CPU-only/headless use)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages (opencv/streamlit etc. -- shared by all features)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# torch/torchvision are needed for dan, mivolo, deep3d, and/or franunet models
RUN --mount=type=cache,target=/root/.cache/pip \
    age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; emotion_csv=",$EMOTION_MODEL,"; \
    recon3d_csv=",$RECONSTRUCTION_3D_MODEL,"; need_torch=false; \
    case "$age_csv" in *,mivolo,*) need_torch=true ;; esac; \
    case "$gender_csv" in *,mivolo,*) need_torch=true ;; esac; \
    case "$emotion_csv" in *,dan,*) need_torch=true ;; esac; \
    case "$recon3d_csv" in *,deep3d,*) need_torch=true ;; esac; \
    case "$AGE_PROGRESSION_MODEL" in *franunet*) need_torch=true ;; esac; \
    if [ "$need_torch" = "true" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu; \
    fi

# scipy is only needed for RECONSTRUCTION_3D_MODEL=deep3d (loading .mat files)
RUN --mount=type=cache,target=/root/.cache/pip \
    recon3d_csv=",$RECONSTRUCTION_3D_MODEL,"; \
    case "$recon3d_csv" in *,deep3d,*) pip install scipy==1.17.1 ;; esac

# YOLO's ONNX export cannot load in cv2.dnn; the glasses export loads but fails
# during inference there; SCRFD's and RetinaFace's multi-output anchor formats use
# onnxruntime too for consistency with the other non-cv2.dnn detector. All four use onnxruntime.
RUN --mount=type=cache,target=/root/.cache/pip \
    yolo_face_csv=",$YOLO_FACE_MODEL,"; glasses_csv=",$GLASSES_MODEL,"; scrfd_face_csv=",$SCRFD_FACE_MODEL,"; retinaface_csv=",$RETINAFACE_MODEL,"; \
    case "$yolo_face_csv:$glasses_csv:$scrfd_face_csv:$retinaface_csv" in *yolo*|*mobilenet*|*scrfd*|*retinaface*) pip install onnxruntime==1.30.0 ;; esac

# tensorflow/tf-keras are only needed for the deepface race, deepface gender,
# and/or mini_xception emotion models
RUN --mount=type=cache,target=/root/.cache/pip \
    race_csv=",$RACE_MODEL,"; gender_csv=",$GENDER_MODEL,"; emotion_csv=",$EMOTION_MODEL,"; recognition_csv=",$RECOGNITION_MODEL,"; \
    mask_csv=",$MASK_MODEL,"; need_tf=false; \
    case "$race_csv" in *,deepface,*) need_tf=true ;; esac; \
    case "$gender_csv" in *,deepface,*) need_tf=true ;; esac; \
    case "$emotion_csv" in *,mini_xception,*) need_tf=true ;; esac; \
    case "$recognition_csv" in *,vggface,*) need_tf=true ;; esac; \
    case "$mask_csv" in *,mobilenetv2,*) need_tf=true ;; esac; \
    if [ "$need_tf" = "true" ]; then pip install tensorflow-cpu==2.21.0 tf-keras==2.21.0; fi

# MiVOLO dependencies (ultralytics, timm) are only needed for the mivolo age and/or gender models
RUN --mount=type=cache,target=/root/.cache/pip \
    age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; need_mivolo=false; \
    case "$age_csv" in *,mivolo,*) need_mivolo=true ;; esac; \
    case "$gender_csv" in *,mivolo,*) need_mivolo=true ;; esac; \
    if [ "$need_mivolo" = "true" ]; then \
        pip install ultralytics==8.1.0 timm==0.8.13.dev0 safetensors==0.8.0 huggingface_hub==2.1.1; \
    fi

# mediapipe is needed for face landmarks, liveness, or hand landmarks
RUN --mount=type=cache,target=/root/.cache/pip \
    face_landmarks_csv=",$FACE_LANDMARKS_MODEL,"; liveness_csv=",$LIVENESS_MODEL,"; hand_csv=",$HAND_MODEL,"; need_mediapipe=false; \
    case "$face_landmarks_csv" in *,mediapipe,*) need_mediapipe=true ;; esac; \
    case "$liveness_csv" in *,mediapipe,*) need_mediapipe=true ;; esac; \
    case "$hand_csv" in *,mediapipe,*) need_mediapipe=true ;; esac; \
    if [ "$need_mediapipe" = "true" ]; then \
        pip install mediapipe==1.1.0; \
    fi

# ultralytics (mivolo) pulls in opencv-python, and mediapipe pulls in a DIFFERENT
# package, opencv-contrib-python, both >=5.0 -- pip happily installs both alongside
# opencv-python-headless, and whichever's "cv2" package wins the import silently lacks
# Caffe support (removed in OpenCV 5.0), breaking caffe/dex age, caffe gender, and
# eccv16 colorization, which all load through cv2.dnn's Caffe importer. Uninstall
# every opencv variant before reinstalling the single 4.14.0.94 pin (the same one
# requirements.txt uses), so there's no ambiguity about which package's cv2 gets imported.
# lbph (recognition) needs cv2.face, which only ships in the "contrib" build -- install
# that build at the same 4.14.0.94 pin (contrib is a strict superset of the main build,
# Caffe support included) when lbph is requested, otherwise stick with the smaller
# opencv-python-headless.
RUN --mount=type=cache,target=/root/.cache/pip \
    recognition_csv=",$RECOGNITION_MODEL,"; opencv_pkg="opencv-python-headless"; \
    case "$recognition_csv" in *,lbph,*) opencv_pkg="opencv-contrib-python-headless" ;; esac; \
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>/dev/null; \
    pip install "${opencv_pkg}==4.14.0.94"

# Application code
COPY pyproject.toml ./
COPY src/ src/
# Editable so face_analyzer resolves to /app/src (BASE_DIR, and therefore models/, is derived
# from the package's own path) and so build-and-run.sh's src/ dev mount takes effect.
# --no-deps: requirements.txt and the per-model extras are already installed above.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

# Only the files tracked in git reach the build context (see .dockerignore): the manifest and
# the small permissively licensed configs. The required SSD detector and the selected weights
# are downloaded here from the sources in models/manifest.json and checked against its sha256s,
# so no local weights are needed to build. Bring-your-own models (dan, deep3d) only print where
# to get the file; the app then lists their features as unavailable, and you can mount the file
# into /app/models/ at runtime instead.
COPY models/ models/
COPY tools/fetch_models.py tools/
RUN python tools/fetch_models.py --keys \
    "AGE_MODEL=$AGE_MODEL" "GENDER_MODEL=$GENDER_MODEL" "EMOTION_MODEL=$EMOTION_MODEL" \
    "RACE_MODEL=$RACE_MODEL" "FACE_LANDMARKS_MODEL=$FACE_LANDMARKS_MODEL" \
    "LIVENESS_MODEL=$LIVENESS_MODEL" "RECOGNITION_MODEL=$RECOGNITION_MODEL" \
    "GLASSES_MODEL=$GLASSES_MODEL" "MASK_MODEL=$MASK_MODEL" "COLORIZATION_MODEL=$COLORIZATION_MODEL" \
    "HAND_MODEL=$HAND_MODEL" "RECONSTRUCTION_3D_MODEL=$RECONSTRUCTION_3D_MODEL" \
    "YOLO_FACE_MODEL=$YOLO_FACE_MODEL" "SCRFD_FACE_MODEL=$SCRFD_FACE_MODEL" \
    "RETINAFACE_MODEL=$RETINAFACE_MODEL" "AGE_PROGRESSION_MODEL=$AGE_PROGRESSION_MODEL"

# Expose default Streamlit port
EXPOSE 8501

# Run the Streamlit web app from the installed package's source directory
CMD ["streamlit", "run", "src/face_analyzer/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.enableXsrfProtection=true"]
