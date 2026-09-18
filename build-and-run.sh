#!/usr/bin/env bash
# Guided Docker build + run for the multimodal face analyzer.
# Prompts for which model(s) to build in per feature, then builds and runs the image.
set -euo pipefail

IMAGE_TAG="face-analyzer"
CONTAINER_NAME="face_analyzer_container"
PORT="8501"

# prompt_feature FEATURE_NAME option1 option2 ...
# Options must be given in quickest-to-build order (option1 = default).
# Sets REPLY_MODEL to a comma-separated list of chosen option names, or "" for none.
prompt_feature() {
    local feature_name="$1"
    shift
    local options=("$@")

    echo "" >&2
    echo "== ${feature_name} model ==" >&2
    echo "  0) none" >&2
    local i=1
    for opt in "${options[@]}"; do
        echo "  ${i}) ${opt}" >&2
        i=$((i + 1))
    done
    echo "  9) all" >&2

    read -rp "Select (comma-separated for multiple) [default: 1]: " selection
    selection="${selection:-1}"

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
            chosen=("${options[@]}")
            break
        fi
        if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -lt 1 ] || [ "$n" -gt "${#options[@]}" ]; then
            echo "Invalid option: ${n}" >&2
            exit 1
        fi
        chosen+=("${options[$((n - 1))]}")
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

prompt_feature "AGE" caffe insightface ssrnet fairface dex mivolo
AGE_MODEL="$REPLY_MODEL"

prompt_feature "GENDER" caffe insightface deepface fairface mivolo
GENDER_MODEL="$REPLY_MODEL"

prompt_feature "EMOTION" efficientnet ferplus mini_xception dan
EMOTION_MODEL="$REPLY_MODEL"

prompt_feature "DROWSINESS" haarcascade
DROWSINESS_MODEL="$REPLY_MODEL"

prompt_feature "RACE" fairface deepface
RACE_MODEL="$REPLY_MODEL"

prompt_feature "EXPRESSION" blendshapes
EXPRESSION_MODEL="$REPLY_MODEL"

prompt_feature "RECOGNITION" vggface lbph
RECOGNITION_MODEL="$REPLY_MODEL"

prompt_feature "FACIAL HAIR" bisenet
FACIAL_HAIR_MODEL="$REPLY_MODEL"

prompt_feature "GLASSES" mobilenet
GLASSES_MODEL="$REPLY_MODEL"

prompt_feature "MASK" mobilenetv2
MASK_MODEL="$REPLY_MODEL"

prompt_feature "COLORIZATION" eccv16
COLORIZATION_MODEL="$REPLY_MODEL"

prompt_feature "POSE" mpi
POSE_MODEL="$REPLY_MODEL"

prompt_feature "HAND LANDMARKS" mediapipe
HAND_MODEL="$REPLY_MODEL"

prompt_feature "3D RECONSTRUCTION (ships no working weights -- see README)" deep3d
RECONSTRUCTION_3D_MODEL="$REPLY_MODEL"

prompt_feature "YOLO FACE DETECTOR (additive -- SSD detector stays required/always on)" yolo
YOLO_FACE_MODEL="$REPLY_MODEL"

prompt_feature "SCRFD FACE DETECTOR (additive -- SSD detector stays required/always on)" scrfd
SCRFD_FACE_MODEL="$REPLY_MODEL"

echo "" >&2
echo "Building ${IMAGE_TAG} with:" >&2
echo "  AGE_MODEL=${AGE_MODEL}" >&2
echo "  GENDER_MODEL=${GENDER_MODEL}" >&2
echo "  EMOTION_MODEL=${EMOTION_MODEL}" >&2
echo "  DROWSINESS_MODEL=${DROWSINESS_MODEL}" >&2
echo "  RACE_MODEL=${RACE_MODEL}" >&2
echo "  EXPRESSION_MODEL=${EXPRESSION_MODEL}" >&2
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
echo "" >&2

docker build \
    --build-arg AGE_MODEL="$AGE_MODEL" \
    --build-arg GENDER_MODEL="$GENDER_MODEL" \
    --build-arg EMOTION_MODEL="$EMOTION_MODEL" \
    --build-arg DROWSINESS_MODEL="$DROWSINESS_MODEL" \
    --build-arg RACE_MODEL="$RACE_MODEL" \
    --build-arg EXPRESSION_MODEL="$EXPRESSION_MODEL" \
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
    -t "$IMAGE_TAG" .

echo "" >&2
read -rp "Live-mount src/ for code edits without rebuilding? (testing only, code changes only -- not for Dockerfile/model/dependency changes) [y/N]: " dev_mount
dev_mount="${dev_mount:-n}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

if [[ "$dev_mount" =~ ^[Yy] ]]; then
    docker run -d -p "${PORT}:8501" --name "$CONTAINER_NAME" \
        -v "$(pwd)/src:/app/src" \
        "$IMAGE_TAG"
    echo "" >&2
    echo "Dev mode: edit src/*.py locally, Streamlit auto-reruns in the container." >&2
else
    docker run -d -p "${PORT}:8501" --name "$CONTAINER_NAME" "$IMAGE_TAG"
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
