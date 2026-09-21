#!/usr/bin/env bash
# Guided Docker build + run for the multimodal face analyzer.
# Prompts for which model(s) to build in per feature, then builds and runs the image.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
IMAGE_TAG="face-analyzer"
CONTAINER_NAME="face_analyzer_container"
PORT="8501"
BLUE=$'\033[1;34m'
GREEN=$'\033[1;32m'
ORANGE_BOLD=$'\033[1;33m'
RESET=$'\033[0m'

# prompt_feature FEATURE_NAME DEFAULT_SELECTION "key|label|color" ...
# Options must use the same order and colors as install-and-run.sh.
# Sets REPLY_MODEL to a comma-separated list of chosen option names, or "" for none.
prompt_feature() {
    local feature_name="$1"
    local default_selection="$2"
    shift 2
    local options=("$@")
    local zero_label="none"
    [ "$feature_name" = "FACE DETECTION" ] && zero_label="ssd"

    echo "" >&2
    printf '%b== %s ==%b\n' "$BLUE" "$feature_name" "$RESET" >&2
    local none_marker=" "
    if [ "$default_selection" = "0" ]; then
        none_marker="*"
    fi
    if [ "$zero_label" = "none" ]; then
        printf '%s 0) %s\n' "$none_marker" "$zero_label" >&2
    else
        printf '%b%s 0) %s%b\n' "$GREEN" "$none_marker" "$zero_label" "$RESET" >&2
    fi
    local i=1
    local marker key option color option_color
    for entry in "${options[@]}"; do
        key="${entry%%|*}"
        local rest="${entry#*|}"
        if [[ "$rest" == *"|"* ]]; then
            option="${rest%%|*}"
            color="${rest#*|}"
        else
            option="$key"
            color="green"
        fi
        if [ "$color" = "orange" ]; then
            option_color="$ORANGE_BOLD"
        else
            option_color="$GREEN"
        fi
        marker=" "
        if [ "$default_selection" != "0" ] && [ "$i" -eq 1 ]; then
            marker="*"
        fi
        printf '%b%s %s) %s%b\n' "$option_color" "$marker" "$i" "$option" "$RESET" >&2
        i=$((i + 1))
    done
    echo "  9) all" >&2

    read -rp "Select: " selection
    selection="${selection:-$default_selection}"

    local chosen=()
    local n token index
    local -a nums=() tokens=()
    IFS=' ' read -ra tokens <<< "${selection//,/ }"
    for token in "${tokens[@]}"; do
        if [[ "$token" =~ ^[0-9]+$ && "${#token}" -gt 1 ]]; then
            for ((index = 0; index < ${#token}; index++)); do
                nums+=("${token:index:1}")
            done
        else
            nums+=("$token")
        fi
    done
    for n in "${nums[@]}"; do
        n="$(echo "$n" | tr -d '[:space:]')"
        [ -z "$n" ] && continue
        if [ "$n" = "0" ]; then
            chosen=()
            break
        fi
        if [ "$n" = "9" ]; then
            for entry in "${options[@]}"; do
                chosen+=("${entry%%|*}")
            done
            break
        fi
        if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -lt 1 ] || [ "$n" -gt "${#options[@]}" ]; then
            echo "Invalid option: ${n}" >&2
            exit 1
        fi
        chosen+=("${options[$((n - 1))]%%|*}")
    done

    local result=""
    local sep=""
    for c in "${chosen[@]:-}"; do
        [ -z "$c" ] && continue
        result="${result}${sep}${c}"
        sep=","
    done
    REPLY_MODEL="$result"
}

set_face_detector_models() {
    YOLO_FACE_MODEL=""
    SCRFD_FACE_MODEL=""
    RETINAFACE_MODEL=""
    local detector
    IFS=',' read -ra detectors <<< "$1"
    for detector in "${detectors[@]:-}"; do
        case "$detector" in
            yolo) YOLO_FACE_MODEL="yolo" ;;
            scrfd) SCRFD_FACE_MODEL="scrfd" ;;
            retinaface) RETINAFACE_MODEL="retinaface" ;;
        esac
    done
}

prompt_feature "FACE DETECTION" 0 \
    "yolo|yolo|orange" \
    "scrfd|scrfd|orange" \
    "retinaface|retinaface|orange"
set_face_detector_models "$REPLY_MODEL"

prompt_feature "AGE" 1 \
    "caffe|caffe|green" \
    "insightface|insightface|green" \
    "fairface|fairface|green" \
    "dex|dex|green" \
    "ssrnet|ssrnet|orange" \
    "mivolo|mivolo|orange"
AGE_MODEL="$REPLY_MODEL"

prompt_feature "GENDER" 1 \
    "caffe|caffe|green" \
    "insightface|insightface|green" \
    "fairface|fairface|green" \
    "deepface|deepface|orange" \
    "mivolo|mivolo|orange"
GENDER_MODEL="$REPLY_MODEL"

prompt_feature "RACE" 1 \
    "fairface|fairface|green" \
    "deepface|deepface|orange"
RACE_MODEL="$REPLY_MODEL"

prompt_feature "EMOTION" 1 \
    "efficientnet|efficientnet|green" \
    "ferplus|ferplus|green" \
    "hsemotion|hsemotion|green" \
    "mini_xception|mini_xception|orange" \
    "dan|dan|orange"
EMOTION_MODEL="$REPLY_MODEL"

prompt_feature "RECOGNITION" 0 \
    "vggface|vggface|orange" \
    "lbph|lbph|orange"
RECOGNITION_MODEL="$REPLY_MODEL"

prompt_feature "ADDITIONAL CLASSIFICATIONS" 0 \
    "mediapipe|liveness - mediapipe|orange" \
    "mobilenet|glasses - mobilenet|orange" \
    "mobilenetv2|mask - mobilenetv2|orange" \
    "colorimetric|hair color - colorimetric|green"
set_additional_classification_models() {
    LIVENESS_MODEL=""
    GLASSES_MODEL=""
    MASK_MODEL=""
    HAIR_COLOR_MODEL=""
    local model
    IFS=',' read -ra models <<< "$1"
    for model in "${models[@]:-}"; do
        case "$model" in
            mediapipe) LIVENESS_MODEL="mediapipe" ;;
            mobilenet) GLASSES_MODEL="mobilenet" ;;
            mobilenetv2) MASK_MODEL="mobilenetv2" ;;
            colorimetric) HAIR_COLOR_MODEL="colorimetric" ;;
        esac
    done
}
set_additional_classification_models "$REPLY_MODEL"

prompt_feature "ADDITIONAL FEATURES" 0 \
    "eccv16|colorization - eccv16|green" \
    "facemesh|face landmarks - mediapipe|orange" \
    "deep3d|3d reconstruction - deep3d|orange" \
    "franunet|age progression - franunet|orange" \
    "mediapipe|hand landmarks - mediapipe|orange"
set_additional_feature_models() {
    COLORIZATION_MODEL=""
    RECONSTRUCTION_3D_MODEL=""
    AGE_PROGRESSION_MODEL=""
    HAND_MODEL=""
    FACE_LANDMARKS_MODEL=""
    local model
    IFS=',' read -ra models <<< "$1"
    for model in "${models[@]:-}"; do
        case "$model" in
            eccv16) COLORIZATION_MODEL="eccv16" ;;
            deep3d) RECONSTRUCTION_3D_MODEL="deep3d" ;;
            franunet) AGE_PROGRESSION_MODEL="franunet" ;;
            mediapipe) HAND_MODEL="mediapipe" ;;
            facemesh) FACE_LANDMARKS_MODEL="mediapipe" ;;
        esac
    done
}
set_additional_feature_models "$REPLY_MODEL"

# Docker otherwise receives the entire models/ directory as its build context
# (more than 3GB in this repository). Build a same-filesystem temporary context
# and hard-link only the selected model files into it, so staging adds no second
# copy of the large weights.
BUILD_CONTEXT="$(mktemp -d "$PROJECT_ROOT/.docker-context.XXXXXX")"
cleanup_build_context() {
    if [ -n "${BUILD_CONTEXT:-}" ] && [ -d "$BUILD_CONTEXT" ]; then
        rm -rf -- "$BUILD_CONTEXT"
    fi
}
trap cleanup_build_context EXIT

cp -a "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT/requirements.txt" "$BUILD_CONTEXT/"
cp -a "$PROJECT_ROOT/src" "$BUILD_CONTEXT/src"
mkdir -p "$BUILD_CONTEXT/models"

link_model() {
    local model_path="$1"
    local source="$PROJECT_ROOT/models/$model_path"
    local target="$BUILD_CONTEXT/models/$model_path"
    if [ ! -f "$source" ]; then
        echo "Missing model file: models/$model_path" >&2
        exit 1
    fi
    mkdir -p "$(dirname -- "$target")"
    [ -e "$target" ] || ln "$source" "$target"
}

stage_model() {
    local selected=",$1,"
    local model_key="$2"
    shift 2
    case "$selected" in
        *,$model_key,*)
            local model_path
            for model_path in "$@"; do
                link_model "$model_path"
            done
            ;;
    esac
}

# SSD is the required detector fallback. All other files are selected by the
# same build arguments that control dependency installation and Dockerfile
# copying.
link_model opencv_face_detector.pbtxt
link_model opencv_face_detector_uint8.pb
stage_model "$AGE_MODEL" caffe age_deploy.prototxt age_net.caffemodel
stage_model "$AGE_MODEL" insightface insightface_genderage.onnx
stage_model "$AGE_MODEL" fairface fairface_7class.onnx
stage_model "$AGE_MODEL" ssrnet ssrnet_morph2.pth
stage_model "$AGE_MODEL" dex dex_age.prototxt dex_age.caffemodel
stage_model "$AGE_MODEL" mivolo mivolo_v2.safetensors mivolo_v2_config.json
stage_model "$GENDER_MODEL" caffe gender_deploy.prototxt gender_net.caffemodel
stage_model "$GENDER_MODEL" insightface insightface_genderage.onnx
stage_model "$GENDER_MODEL" fairface fairface_7class.onnx
stage_model "$GENDER_MODEL" deepface deepface_gender.h5
stage_model "$GENDER_MODEL" mivolo mivolo_v2.safetensors mivolo_v2_config.json
stage_model "$EMOTION_MODEL" dan dan_affecnet7.pth
stage_model "$EMOTION_MODEL" efficientnet efficientnet_b0_fer.onnx
stage_model "$EMOTION_MODEL" ferplus emotion_ferplus.onnx
stage_model "$EMOTION_MODEL" hsemotion hsemotion_enet_b0_8_best_vgaf.onnx
stage_model "$EMOTION_MODEL" mini_xception mini_xception_fer.h5
link_model haarcascade_eye.xml
stage_model "$RACE_MODEL" fairface fairface_7class.onnx
stage_model "$RACE_MODEL" deepface deepface_race.h5
stage_model "$FACE_LANDMARKS_MODEL" facemesh face_landmarker.task
stage_model "$LIVENESS_MODEL" mediapipe face_landmarker.task
stage_model "$RECOGNITION_MODEL" vggface deepface_vgg.h5
stage_model "$GLASSES_MODEL" mobilenet glasses_detector.onnx
stage_model "$MASK_MODEL" mobilenetv2 mask_detector.h5
stage_model "$COLORIZATION_MODEL" eccv16 colorization_deploy_v2.prototxt colorization_release_v2.caffemodel pts_in_hull.npy
stage_model "$HAND_MODEL" mediapipe hand_landmarker.task
stage_model "$RECONSTRUCTION_3D_MODEL" deep3d BFM/similarity_Lm3D_all.mat deep3d_recon_resnet50.pth
stage_model "$YOLO_FACE_MODEL" yolo yolov8n_face.onnx
stage_model "$SCRFD_FACE_MODEL" scrfd scrfd_2.5g_bnkps.onnx
stage_model "$RETINAFACE_MODEL" retinaface retinaface_mobilenet0.25.onnx
stage_model "$AGE_PROGRESSION_MODEL" franunet face_reaging_unet.pth

# Reduce Docker build context from >3GB (all models/) to just the selected weights by
# hard-linking only requested files into a temporary build context. BuildKit then reads
# the minimized tree, avoiding large data transfer to the daemon and layer bloat.
# See Dockerfile's bind-mount comment for why every file must exist (even if not selected).
echo "" >&2
echo "Building ${IMAGE_TAG} with:" >&2
echo "  AGE_MODEL=${AGE_MODEL}" >&2
echo "  GENDER_MODEL=${GENDER_MODEL}" >&2
echo "  EMOTION_MODEL=${EMOTION_MODEL}" >&2
echo "  RACE_MODEL=${RACE_MODEL}" >&2
echo "  FACE_LANDMARKS_MODEL=${FACE_LANDMARKS_MODEL}" >&2
echo "  LIVENESS_MODEL=${LIVENESS_MODEL}" >&2
echo "  RECOGNITION_MODEL=${RECOGNITION_MODEL}" >&2
echo "  GLASSES_MODEL=${GLASSES_MODEL}" >&2
echo "  MASK_MODEL=${MASK_MODEL}" >&2
echo "  HAIR_COLOR_MODEL=${HAIR_COLOR_MODEL}" >&2
echo "  COLORIZATION_MODEL=${COLORIZATION_MODEL}" >&2
echo "  HAND_MODEL=${HAND_MODEL}" >&2
echo "  RECONSTRUCTION_3D_MODEL=${RECONSTRUCTION_3D_MODEL}" >&2
echo "  YOLO_FACE_MODEL=${YOLO_FACE_MODEL}" >&2
echo "  SCRFD_FACE_MODEL=${SCRFD_FACE_MODEL}" >&2
echo "  RETINAFACE_MODEL=${RETINAFACE_MODEL}" >&2
echo "  AGE_PROGRESSION_MODEL=${AGE_PROGRESSION_MODEL}" >&2
echo "" >&2

docker build \
    --build-arg AGE_MODEL="$AGE_MODEL" \
    --build-arg GENDER_MODEL="$GENDER_MODEL" \
    --build-arg EMOTION_MODEL="$EMOTION_MODEL" \
    --build-arg RACE_MODEL="$RACE_MODEL" \
    --build-arg FACE_LANDMARKS_MODEL="$FACE_LANDMARKS_MODEL" \
    --build-arg LIVENESS_MODEL="$LIVENESS_MODEL" \
    --build-arg RECOGNITION_MODEL="$RECOGNITION_MODEL" \
    --build-arg GLASSES_MODEL="$GLASSES_MODEL" \
    --build-arg MASK_MODEL="$MASK_MODEL" \
    --build-arg COLORIZATION_MODEL="$COLORIZATION_MODEL" \
    --build-arg HAND_MODEL="$HAND_MODEL" \
    --build-arg RECONSTRUCTION_3D_MODEL="$RECONSTRUCTION_3D_MODEL" \
    --build-arg YOLO_FACE_MODEL="$YOLO_FACE_MODEL" \
    --build-arg SCRFD_FACE_MODEL="$SCRFD_FACE_MODEL" \
    --build-arg RETINAFACE_MODEL="$RETINAFACE_MODEL" \
    --build-arg AGE_PROGRESSION_MODEL="$AGE_PROGRESSION_MODEL" \
    -t "$IMAGE_TAG" "$BUILD_CONTEXT"

cleanup_build_context
BUILD_CONTEXT=""

echo "" >&2
read -rp "Live-mount src/ for code edits without rebuilding? (testing only, code changes only -- not for Dockerfile/model/dependency changes) [y/N]: " dev_mount
dev_mount="${dev_mount:-n}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

if [[ "$dev_mount" =~ ^[Yy] ]]; then
    docker run -d -p "127.0.0.1:${PORT}:8501" --name "$CONTAINER_NAME" \
        -e "HAIR_COLOR_MODEL=$HAIR_COLOR_MODEL" \
        -v "$PROJECT_ROOT/src:/app/src" \
        "$IMAGE_TAG"
    echo "" >&2
    echo "Dev mode: edit src/*.py locally, Streamlit auto-reruns in the container." >&2
else
    docker run -d -p "127.0.0.1:${PORT}:8501" --name "$CONTAINER_NAME" \
        -e "HAIR_COLOR_MODEL=$HAIR_COLOR_MODEL" "$IMAGE_TAG"
fi

echo "" >&2
echo "Running at http://localhost:${PORT}" >&2

echo "" >&2
echo "Container running. Commands:" >&2
echo "  q = stop container, exit" >&2
echo "  d = stop container, delete image + build cache, exit" >&2
while true; do
    read -rp "> " cmd
    case "$cmd" in
        q|Q)
            docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
            echo "Container removed." >&2
            break
            ;;
        d|D)
            docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
            docker rmi -f "$IMAGE_TAG" >/dev/null 2>&1 || true
            docker builder prune -f >/dev/null 2>&1 || true
            echo "Container, image, and build cache removed." >&2
            break
            ;;
        *)
            echo "Unknown command. Use q or d." >&2
            ;;
    esac
done
