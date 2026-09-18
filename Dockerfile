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
#   LIVENESS_MODEL:    mediapipe                     (default: mediapipe)
#   RECOGNITION_MODEL: vggface, lbph                 (default: vggface)
#   FACIAL_HAIR_MODEL: bisenet                       (default: bisenet)
#   GLASSES_MODEL:     mobilenet                     (default: mobilenet)
#   MASK_MODEL:        mobilenetv2                   (default: mobilenetv2)
#   COLORIZATION_MODEL: eccv16                        (default: eccv16)
#   POSE_MODEL:         mpi                            (default: mpi)
#   HAND_MODEL:         mediapipe                       (default: mediapipe)
#   RECONSTRUCTION_3D_MODEL: deep3d                      (default: deep3d)
#   YOLO_FACE_MODEL:    yolo                             (default: yolo)
# YOLO_FACE_MODEL is additive, not a replacement -- the original SSD/ResNet-10 TensorFlow
# detector is always required and always on; this ARG only controls whether the alternative
# YOLOv8-Face ONNX file is ALSO built in, selectable at runtime via a sidebar dropdown (exactly
# one detector runs per frame). Needs onnxruntime (not this repo's usual cv2.dnn ONNX path --
# cv2.dnn cannot parse this specific export, verified against both OpenCV 4.10 and 5.0).
# lbph (Local Binary Patterns Histogram, opencv-contrib's cv2.face module) needs
# opencv-contrib-python-headless instead of opencv-python-headless -- see the final opencv
# reinstall step below. Unlike vggface, it has no pretrained weights: it trains from scratch
# on whatever's enrolled via the ENROLL button, same "trains fresh on demand" spirit as this
# app's eigenfaces feature.
# pose (CMU OpenPose MPI model) is ACADEMIC/NON-COMMERCIAL RESEARCH USE ONLY -- see README.
# Face Landmarks has no build ARG of its own -- it reuses the same face_landmarker.task file.
# Liveness uses LIVENESS_MODEL=mediapipe with that same file and dependency.
# RECONSTRUCTION_3D_MODEL wires the code path (torch/torchvision/scipy + the small bundled
# BFM landmark template) but ships NO working weights -- Deep3DFaceRecon_pytorch's checkpoint
# and the Basel Face Model data it needs are both gated (Google Drive / university license
# registration respectively); this ARG alone will never produce a working reconstruction.
# See README's Known Issues for what the user must supply themselves.
# insightface's genderage.onnx provides BOTH age and gender from one file
# (non-commercial research license -- see README). deepface's race model
# needs TensorFlow (~200-400MB) and a 513MB weight file, much heavier
# than fairface -- only pulled in if requested. mask also needs TensorFlow
# (Keras .h5 weights); facial_hair and glasses are plain ONNX.
# No SKIN_TONE_MODEL ARG -- the only known source for this feature
# (behra527/Skin-Tone-Classification-model) ships a corrupted weight file
# that doesn't load under any Keras version tried; see README's Known Issues.
# The Python-side plumbing exists (src/inference.py) for whenever a working
# weight file is found, but there's nothing to build into the image yet.
# e.g. --build-arg AGE_MODEL=caffe,ssrnet builds both age backends so the web
# app can switch between them at runtime. See build-and-run.sh for a guided
# prompt instead of typing these by hand.
ARG AGE_MODEL=caffe
ARG GENDER_MODEL=caffe
ARG EMOTION_MODEL=efficientnet
ARG DROWSINESS_MODEL=haarcascade
ARG RACE_MODEL=fairface
ARG EXPRESSION_MODEL=blendshapes
ARG LIVENESS_MODEL=mediapipe
ARG RECOGNITION_MODEL=vggface
ARG FACIAL_HAIR_MODEL=bisenet
ARG GLASSES_MODEL=mobilenet
ARG MASK_MODEL=mobilenetv2
ARG COLORIZATION_MODEL=eccv16
ARG POSE_MODEL=mpi
ARG HAND_MODEL=mediapipe
ARG RECONSTRUCTION_3D_MODEL=deep3d
ARG YOLO_FACE_MODEL=yolo

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

# torch/torchvision are needed for ssrnet, dan, and/or mivolo models
RUN --mount=type=cache,target=/root/.cache/pip \
    age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; emotion_csv=",$EMOTION_MODEL,"; \
    recon3d_csv=",$RECONSTRUCTION_3D_MODEL,"; need_torch=false; \
    case "$age_csv" in *,ssrnet,*) need_torch=true ;; esac; \
    case "$age_csv" in *,mivolo,*) need_torch=true ;; esac; \
    case "$gender_csv" in *,mivolo,*) need_torch=true ;; esac; \
    case "$emotion_csv" in *,dan,*) need_torch=true ;; esac; \
    case "$recon3d_csv" in *,deep3d,*) need_torch=true ;; esac; \
    if [ "$need_torch" = "true" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    fi

# scipy is only needed for RECONSTRUCTION_3D_MODEL=deep3d (loading .mat files)
RUN --mount=type=cache,target=/root/.cache/pip \
    recon3d_csv=",$RECONSTRUCTION_3D_MODEL,"; \
    case "$recon3d_csv" in *,deep3d,*) pip install scipy ;; esac

# YOLO's ONNX export cannot load in cv2.dnn; the glasses export loads but fails
# during inference there. Both use onnxruntime.
RUN --mount=type=cache,target=/root/.cache/pip \
    yolo_face_csv=",$YOLO_FACE_MODEL,"; glasses_csv=",$GLASSES_MODEL,"; \
    case "$yolo_face_csv:$glasses_csv" in *yolo*|*mobilenet*) pip install onnxruntime ;; esac

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
    if [ "$need_tf" = "true" ]; then pip install tensorflow-cpu tf-keras; fi

# MiVOLO dependencies (ultralytics, timm) are only needed for the mivolo age and/or gender models
RUN --mount=type=cache,target=/root/.cache/pip \
    age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; need_mivolo=false; \
    case "$age_csv" in *,mivolo,*) need_mivolo=true ;; esac; \
    case "$gender_csv" in *,mivolo,*) need_mivolo=true ;; esac; \
    if [ "$need_mivolo" = "true" ]; then \
        pip install ultralytics==8.1.0 timm==0.8.13.dev0 safetensors huggingface_hub; \
    fi

# mediapipe is needed for blendshapes expression, liveness, or hand landmarks
RUN --mount=type=cache,target=/root/.cache/pip \
    expression_csv=",$EXPRESSION_MODEL,"; liveness_csv=",$LIVENESS_MODEL,"; hand_csv=",$HAND_MODEL,"; need_mediapipe=false; \
    case "$expression_csv" in *,blendshapes,*) need_mediapipe=true ;; esac; \
    case "$liveness_csv" in *,mediapipe,*) need_mediapipe=true ;; esac; \
    case "$hand_csv" in *,mediapipe,*) need_mediapipe=true ;; esac; \
    if [ "$need_mediapipe" = "true" ]; then \
        pip install mediapipe; \
    fi

# ultralytics (mivolo) pulls in opencv-python, and mediapipe pulls in a DIFFERENT
# package, opencv-contrib-python, both >=5.0 -- pip happily installs both alongside
# opencv-python-headless, and whichever's "cv2" package wins the import silently lacks
# Caffe support (removed in OpenCV 5.0), breaking caffe/dex age, caffe gender, and
# haarcascade drowsiness (all use cv2.dnn.readNetFromCaffe/CascadeClassifier). Uninstall
# every opencv variant before reinstalling the one pinned version, so there's no
# ambiguity about which package's cv2 gets imported. lbph (recognition) needs cv2.face,
# which only ships in the "contrib" build -- swap the pinned package for that build (still
# <5.0.0, still has Caffe support -- contrib is a strict superset of the main build) when
# lbph is requested, otherwise stick with the smaller opencv-python-headless.
RUN --mount=type=cache,target=/root/.cache/pip \
    recognition_csv=",$RECOGNITION_MODEL,"; opencv_pkg="opencv-python-headless"; \
    case "$recognition_csv" in *,lbph,*) opencv_pkg="opencv-contrib-python-headless" ;; esac; \
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>/dev/null; \
    pip install "${opencv_pkg}>=4.8.0,<5.0.0"

# Application code and always-required model files (face detector)
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
    --mount=type=bind,source=models/deepface_vgg.h5,target=/tmp/models/deepface_vgg.h5 \
    --mount=type=bind,source=models/mini_xception_fer.h5,target=/tmp/models/mini_xception_fer.h5 \
    --mount=type=bind,source=models/dex_age.prototxt,target=/tmp/models/dex_age.prototxt \
    --mount=type=bind,source=models/dex_age.caffemodel,target=/tmp/models/dex_age.caffemodel \
    --mount=type=bind,source=models/mivolo_v2.safetensors,target=/tmp/models/mivolo_v2.safetensors \
    --mount=type=bind,source=models/mivolo_v2_config.json,target=/tmp/models/mivolo_v2_config.json \
    --mount=type=bind,source=models/face_landmarker.task,target=/tmp/models/face_landmarker.task \
    --mount=type=bind,source=models/bisenet_face_parsing.onnx,target=/tmp/models/bisenet_face_parsing.onnx \
    --mount=type=bind,source=models/glasses_detector.onnx,target=/tmp/models/glasses_detector.onnx \
    --mount=type=bind,source=models/mask_detector.h5,target=/tmp/models/mask_detector.h5 \
    --mount=type=bind,source=models/colorization_deploy_v2.prototxt,target=/tmp/models/colorization_deploy_v2.prototxt \
    --mount=type=bind,source=models/colorization_release_v2.caffemodel,target=/tmp/models/colorization_release_v2.caffemodel \
    --mount=type=bind,source=models/pts_in_hull.npy,target=/tmp/models/pts_in_hull.npy \
    --mount=type=bind,source=models/pose_deploy_linevec_faster_4_stages.prototxt,target=/tmp/models/pose_deploy_linevec_faster_4_stages.prototxt \
    --mount=type=bind,source=models/pose_iter_160000.caffemodel,target=/tmp/models/pose_iter_160000.caffemodel \
    --mount=type=bind,source=models/hand_landmarker.task,target=/tmp/models/hand_landmarker.task \
    --mount=type=bind,source=models/BFM/similarity_Lm3D_all.mat,target=/tmp/models/BFM/similarity_Lm3D_all.mat \
    --mount=type=bind,source=models/deep3d_recon_resnet50.pth,target=/tmp/models/deep3d_recon_resnet50.pth \
    --mount=type=bind,source=models/yolov8n_face.onnx,target=/tmp/models/yolov8n_face.onnx \
    set -e; \
    age_csv=",$AGE_MODEL,"; gender_csv=",$GENDER_MODEL,"; emotion_csv=",$EMOTION_MODEL,"; \
    drowsiness_csv=",$DROWSINESS_MODEL,"; race_csv=",$RACE_MODEL,"; expression_csv=",$EXPRESSION_MODEL,"; liveness_csv=",$LIVENESS_MODEL,"; recognition_csv=",$RECOGNITION_MODEL,"; \
    facial_hair_csv=",$FACIAL_HAIR_MODEL,"; glasses_csv=",$GLASSES_MODEL,"; mask_csv=",$MASK_MODEL,"; colorization_csv=",$COLORIZATION_MODEL,"; pose_csv=",$POSE_MODEL,"; hand_csv=",$HAND_MODEL,"; recon3d_csv=",$RECONSTRUCTION_3D_MODEL,"; yolo_face_csv=",$YOLO_FACE_MODEL,"; \
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
    case "$expression_csv" in *,blendshapes,*) cp /tmp/models/face_landmarker.task models/ ;; esac; \
    case "$liveness_csv" in *,mediapipe,*) cp /tmp/models/face_landmarker.task models/ ;; esac; \
    case "$recognition_csv" in *,vggface,*) cp /tmp/models/deepface_vgg.h5 models/ ;; esac; \
    case "$facial_hair_csv" in *,bisenet,*) cp /tmp/models/bisenet_face_parsing.onnx models/ ;; esac; \
    case "$glasses_csv" in *,mobilenet,*) cp /tmp/models/glasses_detector.onnx models/ ;; esac; \
    case "$mask_csv" in *,mobilenetv2,*) cp /tmp/models/mask_detector.h5 models/ ;; esac; \
    case "$colorization_csv" in *,eccv16,*) cp /tmp/models/colorization_deploy_v2.prototxt /tmp/models/colorization_release_v2.caffemodel /tmp/models/pts_in_hull.npy models/ ;; esac; \
    case "$pose_csv" in *,mpi,*) cp /tmp/models/pose_deploy_linevec_faster_4_stages.prototxt /tmp/models/pose_iter_160000.caffemodel models/ ;; esac; \
    case "$hand_csv" in *,mediapipe,*) cp /tmp/models/hand_landmarker.task models/ ;; esac; \
    case "$recon3d_csv" in *,deep3d,*) mkdir -p models/BFM && cp /tmp/models/BFM/similarity_Lm3D_all.mat models/BFM/ && cp /tmp/models/deep3d_recon_resnet50.pth models/ ;; esac; \
    case "$yolo_face_csv" in *,yolo,*) cp /tmp/models/yolov8n_face.onnx models/ ;; esac

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit web app from src directory
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
