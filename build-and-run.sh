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

prompt_feature "AGE" caffe insightface ssrnet
AGE_MODEL="$REPLY_MODEL"

prompt_feature "GENDER" caffe insightface
GENDER_MODEL="$REPLY_MODEL"

prompt_feature "EMOTION" efficientnet dan
EMOTION_MODEL="$REPLY_MODEL"

prompt_feature "DROWSINESS" haarcascade
DROWSINESS_MODEL="$REPLY_MODEL"

prompt_feature "RACE" fairface
RACE_MODEL="$REPLY_MODEL"

echo "" >&2
echo "Building ${IMAGE_TAG} with:" >&2
echo "  AGE_MODEL=${AGE_MODEL}" >&2
echo "  GENDER_MODEL=${GENDER_MODEL}" >&2
echo "  EMOTION_MODEL=${EMOTION_MODEL}" >&2
echo "  DROWSINESS_MODEL=${DROWSINESS_MODEL}" >&2
echo "  RACE_MODEL=${RACE_MODEL}" >&2
echo "" >&2

docker build \
    --build-arg AGE_MODEL="$AGE_MODEL" \
    --build-arg GENDER_MODEL="$GENDER_MODEL" \
    --build-arg EMOTION_MODEL="$EMOTION_MODEL" \
    --build-arg DROWSINESS_MODEL="$DROWSINESS_MODEL" \
    --build-arg RACE_MODEL="$RACE_MODEL" \
    -t "$IMAGE_TAG" .

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d -p "${PORT}:8501" --name "$CONTAINER_NAME" "$IMAGE_TAG"

echo "" >&2
echo "Running at http://localhost:${PORT}" >&2
