#!/usr/bin/env bash
# Guided Docker build + run for the multimodal face analyzer.
# Prompts for which model(s) to build in per feature, then builds and runs the image.
set -euo pipefail

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
    IFS=',' read -ra nums <<< "$selection"
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
    "haarcascade|drowsiness - haarcascade|green" \
    "bisenet|facial hair - bisenet|green" \
    "blendshapes|expressions - blendshapes|orange" \
    "mediapipe|liveness - mediapipe|orange" \
    "mobilenet|glasses - mobilenet|orange" \
    "mobilenetv2|mask - mobilenetv2|orange"
set_additional_classification_models() {
    EXPRESSION_MODEL=""
    DROWSINESS_MODEL=""
    LIVENESS_MODEL=""
    FACIAL_HAIR_MODEL=""
    GLASSES_MODEL=""
    MASK_MODEL=""
    local model
    IFS=',' read -ra models <<< "$1"
    for model in "${models[@]:-}"; do
        case "$model" in
            blendshapes) EXPRESSION_MODEL="blendshapes" ;;
            haarcascade) DROWSINESS_MODEL="haarcascade" ;;
            mediapipe) LIVENESS_MODEL="mediapipe" ;;
            bisenet) FACIAL_HAIR_MODEL="bisenet" ;;
            mobilenet) GLASSES_MODEL="mobilenet" ;;
            mobilenetv2) MASK_MODEL="mobilenetv2" ;;
        esac
    done
}
set_additional_classification_models "$REPLY_MODEL"

prompt_feature "ADDITIONAL FEATURES" 0 \
    "eccv16|colorization - eccv16|green" \
    "mpi|body pose - mpi|green" \
    "deep3d|3d reconstruction - deep3d|orange" \
    "franunet|age progression - franunet|orange" \
    "mediapipe|hand landmarks - mediapipe|orange"
set_additional_feature_models() {
    COLORIZATION_MODEL=""
    RECONSTRUCTION_3D_MODEL=""
    AGE_PROGRESSION_MODEL=""
    HAND_MODEL=""
    POSE_MODEL=""
    local model
    IFS=',' read -ra models <<< "$1"
    for model in "${models[@]:-}"; do
        case "$model" in
            eccv16) COLORIZATION_MODEL="eccv16" ;;
            deep3d) RECONSTRUCTION_3D_MODEL="deep3d" ;;
            franunet) AGE_PROGRESSION_MODEL="franunet" ;;
            mediapipe) HAND_MODEL="mediapipe" ;;
            mpi) POSE_MODEL="mpi" ;;
        esac
    done
}
set_additional_feature_models "$REPLY_MODEL"

echo "" >&2
echo "Building ${IMAGE_TAG} with:" >&2
echo "  AGE_MODEL=${AGE_MODEL}" >&2
echo "  GENDER_MODEL=${GENDER_MODEL}" >&2
echo "  EMOTION_MODEL=${EMOTION_MODEL}" >&2
echo "  DROWSINESS_MODEL=${DROWSINESS_MODEL}" >&2
echo "  RACE_MODEL=${RACE_MODEL}" >&2
echo "  EXPRESSION_MODEL=${EXPRESSION_MODEL}" >&2
echo "  LIVENESS_MODEL=${LIVENESS_MODEL}" >&2
echo "  RECOGNITION_MODEL=${RECOGNITION_MODEL}" >&2
echo "  FACIAL_HAIR_MODEL=${FACIAL_HAIR_MODEL}" >&2
echo "  GLASSES_MODEL=${GLASSES_MODEL}" >&2
echo "  MASK_MODEL=${MASK_MODEL}" >&2
echo "  COLORIZATION_MODEL=${COLORIZATION_MODEL}" >&2
echo "  POSE_MODEL=${POSE_MODEL}" >&2
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
    --build-arg DROWSINESS_MODEL="$DROWSINESS_MODEL" \
    --build-arg RACE_MODEL="$RACE_MODEL" \
    --build-arg EXPRESSION_MODEL="$EXPRESSION_MODEL" \
    --build-arg LIVENESS_MODEL="$LIVENESS_MODEL" \
    --build-arg RECOGNITION_MODEL="$RECOGNITION_MODEL" \
    --build-arg FACIAL_HAIR_MODEL="$FACIAL_HAIR_MODEL" \
    --build-arg GLASSES_MODEL="$GLASSES_MODEL" \
    --build-arg MASK_MODEL="$MASK_MODEL" \
    --build-arg COLORIZATION_MODEL="$COLORIZATION_MODEL" \
    --build-arg POSE_MODEL="$POSE_MODEL" \
    --build-arg HAND_MODEL="$HAND_MODEL" \
    --build-arg RECONSTRUCTION_3D_MODEL="$RECONSTRUCTION_3D_MODEL" \
    --build-arg YOLO_FACE_MODEL="$YOLO_FACE_MODEL" \
    --build-arg SCRFD_FACE_MODEL="$SCRFD_FACE_MODEL" \
    --build-arg RETINAFACE_MODEL="$RETINAFACE_MODEL" \
    --build-arg AGE_PROGRESSION_MODEL="$AGE_PROGRESSION_MODEL" \
    -t "$IMAGE_TAG" .

echo "" >&2
read -rp "Live-mount src/ for code edits without rebuilding? (testing only, code changes only -- not for Dockerfile/model/dependency changes) [y/N]: " dev_mount
dev_mount="${dev_mount:-n}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

if [[ "$dev_mount" =~ ^[Yy] ]]; then
    docker run -d -p "127.0.0.1:${PORT}:8501" --name "$CONTAINER_NAME" \
        -v "$(pwd)/src:/app/src" \
        "$IMAGE_TAG"
    echo "" >&2
    echo "Dev mode: edit src/*.py locally, Streamlit auto-reruns in the container." >&2
else
    docker run -d -p "127.0.0.1:${PORT}:8501" --name "$CONTAINER_NAME" "$IMAGE_TAG"
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
