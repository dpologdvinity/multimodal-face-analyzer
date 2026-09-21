#!/usr/bin/env bash
# Guided native setup + run for the multimodal face analyzer.
# Prompts for model selections, installs the matching dependencies, and runs Streamlit.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="8501"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VERBOSE=false
HIDDEN=false
ALLOWED_PACKAGES=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--hidden)
            HIDDEN=true
            shift
            ;;
        -p|--package)
            if [ "$#" -lt 2 ]; then
                echo "${1} requires a comma-separated package list." >&2
                exit 2
            fi
            if [ -n "$ALLOWED_PACKAGES" ]; then
                ALLOWED_PACKAGES="${ALLOWED_PACKAGES},$2"
            else
                ALLOWED_PACKAGES="$2"
            fi
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Usage: $0 [-v|--verbose] [-h|--hidden] [-p|--package package[,package...]]" >&2
            exit 2
            ;;
    esac
done

BLUE=$'\033[1;34m'
GREEN=$'\033[1;32m'
PACKAGE_GREEN=$'\033[32m'
ORANGE=$'\033[33m'
ORANGE_BOLD=$'\033[1;33m'
RESET=$'\033[0m'
BASE_PACKAGES="opencv-python-headless,numpy"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python 3.11 (or python3) is required." >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10 or newer is required." >&2
    exit 1
fi

# prompt_feature FEATURE_NAME DEFAULT_SELECTION "key|label|package[,package...]" ...
# Option order and defaults must exactly match build-and-run.sh so both scripts behave identically.
# When --hidden is set, filters to show only options that would install new packages.
# Sets REPLY_MODEL to a comma-separated list of selected model keys.
prompt_feature() {
    local feature_name="$1"
    local default_selection="$2"
    shift 2
    local entries=("$@")
    local entry key option package_spec rest marker option_color package package_name
    local green_packages orange_packages package_line_count package_glyph
    local -a packages visible_entries
    local zero_label="none"
    [ "$feature_name" = "FACE DETECTION" ] && zero_label="ssd"

    # --hidden mode filters to show only options that install new packages (marked with +),
    # hiding already-installed backends. This lets users quickly re-run the script to toggle
    # additional features without retracing their dependency footprint.
    if [ "$HIDDEN" = true ]; then
        visible_entries=()
        for entry in "${entries[@]}"; do
            rest="${entry#*|}"
            if [[ "$rest" == *"|"* ]]; then
                package_spec="${rest#*|}"
            else
                package_spec="$rest"
            fi
            local has_new_package=false
            IFS=',' read -ra packages <<< "$package_spec"
            for package in "${packages[@]}"; do
                package_name="${package#+}"
                if [[ "$package" == +* ]] && ! package_is_installed "$package_name"; then
                    has_new_package=true
                    break
                fi
            done
            if [ "$has_new_package" = false ]; then
                visible_entries+=("$entry")
            fi
        done
        entries=("${visible_entries[@]}")
        if [ "${#entries[@]}" -eq 0 ]; then
            default_selection="0"
        fi
    fi

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
    for entry in "${entries[@]}"; do
        key="${entry%%|*}"
        rest="${entry#*|}"
        if [[ "$rest" == *"|"* ]]; then
            option="${rest%%|*}"
            package_spec="${rest#*|}"
        else
            option="$key"
            package_spec="$rest"
        fi
        marker=" "
        if [ "$default_selection" != "0" ] && [ "$i" -eq 1 ]; then
            marker="*"
        fi
        option_color="$GREEN"
        green_packages=""
        orange_packages=""
        IFS=',' read -ra packages <<< "$package_spec"
        for package in "${packages[@]}"; do
            package_name="${package#+}"
            if [[ "$package" == +* ]] && ! package_is_installed "$package_name"; then
                if [ -n "$orange_packages" ]; then
                    orange_packages="$orange_packages, $package_name"
                else
                    orange_packages="$package_name"
                fi
                option_color="$ORANGE_BOLD"
            else
                if [ -n "$green_packages" ]; then
                    green_packages="$green_packages, $package_name"
                else
                    green_packages="$package_name"
                fi
            fi
        done
        printf '%b%s %s) %s%b\n' "$option_color" "$marker" "$i" "$option" "$RESET" >&2
        if [ "$VERBOSE" = true ]; then
            package_line_count=0
            [ -n "$green_packages" ] && package_line_count=$((package_line_count + 1))
            [ -n "$orange_packages" ] && package_line_count=$((package_line_count + 1))
            if [ -n "$green_packages" ]; then
                package_glyph="└──"
                [ "$package_line_count" -eq 2 ] && package_glyph="├──"
                printf '      %s %b%s%b\n' "$package_glyph" "$PACKAGE_GREEN" "$green_packages" "$RESET" >&2
            fi
            if [ -n "$orange_packages" ]; then
                printf '      └── %b%s%b\n' "$ORANGE" "$orange_packages" "$RESET" >&2
            fi
        fi
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
            for entry in "${entries[@]}"; do
                chosen+=("${entry%%|*}")
            done
            break
        fi
        if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -lt 1 ] || [ "$n" -gt "${#entries[@]}" ]; then
            echo "Invalid option: ${n}" >&2
            exit 1
        fi
        chosen+=("${entries[$((n - 1))]%%|*}")
    done

    local result=""
    local sep=""
    local model
    for model in "${chosen[@]:-}"; do
        [ -z "$model" ] && continue
        result="${result}${sep}${model}"
        sep=","
    done
    REPLY_MODEL="$result"

    local selected_entry selected_rest selected_packages selected_package selected_name
    for model in "${chosen[@]:-}"; do
        for selected_entry in "${entries[@]}"; do
            if [ "${selected_entry%%|*}" != "$model" ]; then
                continue
            fi
            selected_rest="${selected_entry#*|}"
            if [[ "$selected_rest" == *"|"* ]]; then
                selected_packages="${selected_rest#*|}"
            else
                selected_packages="$selected_rest"
            fi
            IFS=',' read -ra packages <<< "$selected_packages"
            for selected_package in "${packages[@]}"; do
                selected_name="${selected_package#+}"
                if [[ "$selected_package" == +* ]]; then
                    add_installed_package "$selected_name"
                fi
            done
        done
    done
}

INSTALLED_PACKAGES=",$BASE_PACKAGES,"
package_is_installed() {
    [[ "$INSTALLED_PACKAGES" == *",$1,"* ]]
}
add_installed_package() {
    if ! package_is_installed "$1"; then
        INSTALLED_PACKAGES="${INSTALLED_PACKAGES}$1,"
    fi
}
local_allowed_package=""
IFS=',' read -ra allowed_packages <<< "$ALLOWED_PACKAGES"
for local_allowed_package in "${allowed_packages[@]:-}"; do
    local_allowed_package="$(echo "$local_allowed_package" | tr -d '[:space:]')"
    [ -z "$local_allowed_package" ] && continue
    if [ "$local_allowed_package" = "onxruntime" ]; then
        local_allowed_package="onnxruntime"
    fi
    add_installed_package "$local_allowed_package"
done
set_face_detector_models() {
    YOLO_FACE_MODEL=""
    SCRFD_FACE_MODEL=""
    RETINAFACE_MODEL=""
    local -a detectors=()
    local detector
    if [ -n "$1" ]; then
        IFS=',' read -ra detectors <<< "$1"
    fi
    for detector in "${detectors[@]:-}"; do
        case "$detector" in
            yolo) YOLO_FACE_MODEL="yolo" ;;
            scrfd) SCRFD_FACE_MODEL="scrfd" ;;
            retinaface) RETINAFACE_MODEL="retinaface" ;;
        esac
    done
}

prompt_feature "FACE DETECTION" 0 \
    "yolo|yolo|$BASE_PACKAGES,+onnxruntime" \
    "scrfd|scrfd|$BASE_PACKAGES,+onnxruntime" \
    "retinaface|retinaface|$BASE_PACKAGES,+onnxruntime"
set_face_detector_models "$REPLY_MODEL"

prompt_feature "AGE" 1 \
    "caffe|caffe|$BASE_PACKAGES" \
    "fairface|fairface|$BASE_PACKAGES" \
    "dex|dex|$BASE_PACKAGES" \
    "ssrnet|ssrnet|$BASE_PACKAGES,+torch,+torchvision" \
    "mivolo|mivolo|$BASE_PACKAGES,+torch,+torchvision,+ultralytics,+timm,+safetensors,+huggingface_hub"
AGE_MODEL="$REPLY_MODEL"

prompt_feature "GENDER" 1 \
    "caffe|caffe|$BASE_PACKAGES" \
    "fairface|fairface|$BASE_PACKAGES" \
    "deepface|deepface|$BASE_PACKAGES,+tensorflow-cpu,+tf-keras" \
    "mivolo|mivolo|$BASE_PACKAGES,+torch,+torchvision,+ultralytics,+timm,+safetensors,+huggingface_hub"
GENDER_MODEL="$REPLY_MODEL"

prompt_feature "RACE" 1 \
    "fairface|fairface|$BASE_PACKAGES" \
    "deepface|deepface|$BASE_PACKAGES,+tensorflow-cpu,+tf-keras"
RACE_MODEL="$REPLY_MODEL"

prompt_feature "EMOTION" 1 \
    "efficientnet|efficientnet|$BASE_PACKAGES" \
    "ferplus|ferplus|$BASE_PACKAGES" \
    "hsemotion|hsemotion|$BASE_PACKAGES" \
    "mini_xception|mini_xception|$BASE_PACKAGES,+tensorflow-cpu,+tf-keras" \
    "dan|dan|$BASE_PACKAGES,+torch,+torchvision"
EMOTION_MODEL="$REPLY_MODEL"

prompt_feature "RECOGNITION" 0 \
    "vggface|vggface|$BASE_PACKAGES,+tensorflow-cpu,+tf-keras" \
    "lbph|lbph|$BASE_PACKAGES,+opencv-contrib-python-headless"
RECOGNITION_MODEL="$REPLY_MODEL"

prompt_feature "ADDITIONAL CLASSIFICATIONS" 0 \
    "mediapipe|liveness - mediapipe|$BASE_PACKAGES,+mediapipe" \
    "mobilenet|glasses - mobilenet|$BASE_PACKAGES,+onnxruntime" \
    "mobilenetv2|mask - mobilenetv2|$BASE_PACKAGES,+tensorflow-cpu,+tf-keras" \
    "colorimetric|hair color - colorimetric|"
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
    "eccv16|colorization - eccv16|$BASE_PACKAGES" \
    "facemesh|face landmarks - mediapipe|$BASE_PACKAGES,+mediapipe" \
    "deep3d|3d reconstruction - deep3d|$BASE_PACKAGES,+torch,+torchvision,+scipy" \
    "franunet|age progression - franunet|$BASE_PACKAGES,+torch,+torchvision" \
    "mediapipe|hand landmarks - mediapipe|$BASE_PACKAGES,+mediapipe"
set_additional_feature_models() {
    COLORIZATION_MODEL=""
    RECONSTRUCTION_3D_MODEL=""
    AGE_PROGRESSION_MODEL=""
    FACE_LANDMARKS_MODEL=""
    HAND_MODEL=""
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

csv_has() {
    local csv="$1"
    local value="$2"
    [[ ",${csv}," == *,"${value}",* ]]
}

NEED_TORCH=false
NEED_TENSORFLOW=false
NEED_MEDIAPIPE=false
NEED_ONNXRUNTIME=false
NEED_MIVOLO=false
NEED_SCIPY=false

if csv_has "$AGE_MODEL" ssrnet || csv_has "$AGE_MODEL" mivolo \
    || csv_has "$GENDER_MODEL" mivolo || csv_has "$EMOTION_MODEL" dan \
    || csv_has "$RECONSTRUCTION_3D_MODEL" deep3d || csv_has "$AGE_PROGRESSION_MODEL" franunet; then
    NEED_TORCH=true
fi
if csv_has "$RACE_MODEL" deepface || csv_has "$GENDER_MODEL" deepface \
    || csv_has "$EMOTION_MODEL" mini_xception || csv_has "$RECOGNITION_MODEL" vggface \
    || csv_has "$MASK_MODEL" mobilenetv2; then
    NEED_TENSORFLOW=true
fi
if csv_has "$FACE_LANDMARKS_MODEL" mediapipe || csv_has "$LIVENESS_MODEL" mediapipe \
    || csv_has "$HAND_MODEL" mediapipe; then
    NEED_MEDIAPIPE=true
fi
if csv_has "$GLASSES_MODEL" mobilenet || csv_has "$YOLO_FACE_MODEL" yolo \
    || csv_has "$SCRFD_FACE_MODEL" scrfd || csv_has "$RETINAFACE_MODEL" retinaface; then
    NEED_ONNXRUNTIME=true
fi
if csv_has "$AGE_MODEL" mivolo || csv_has "$GENDER_MODEL" mivolo; then
    NEED_MIVOLO=true
fi
if csv_has "$RECONSTRUCTION_3D_MODEL" deep3d; then
    NEED_SCIPY=true
fi

echo "" >&2
echo "Selected models:" >&2
echo "  AGE_MODEL=${AGE_MODEL}" >&2
echo "  GENDER_MODEL=${GENDER_MODEL}" >&2
echo "  EMOTION_MODEL=${EMOTION_MODEL}" >&2
echo "  RACE_MODEL=${RACE_MODEL}" >&2
echo "  FACE_LANDMARKS_MODEL=${FACE_LANDMARKS_MODEL}" >&2
echo "  LIVENESS_MODEL=${LIVENESS_MODEL}" >&2
echo "  RECOGNITION_MODEL=${RECOGNITION_MODEL}" >&2
echo "  GLASSES_MODEL=${GLASSES_MODEL}" >&2
echo "  MASK_MODEL=${MASK_MODEL}" >&2
echo "  COLORIZATION_MODEL=${COLORIZATION_MODEL}" >&2
echo "  HAND_MODEL=${HAND_MODEL}" >&2
echo "  RECONSTRUCTION_3D_MODEL=${RECONSTRUCTION_3D_MODEL}" >&2
echo "  YOLO_FACE_MODEL=${YOLO_FACE_MODEL}" >&2
echo "  SCRFD_FACE_MODEL=${SCRFD_FACE_MODEL}" >&2
echo "  RETINAFACE_MODEL=${RETINAFACE_MODEL}" >&2
echo "  AGE_PROGRESSION_MODEL=${AGE_PROGRESSION_MODEL}" >&2

install_system_packages() {
    local packages=(libgl1 libglib2.0-0 libegl1 libgles2)
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Skipping system packages: apt-get is unavailable." >&2
        echo "Install these runtime libraries using your OS package manager: ${packages[*]}" >&2
        return
    fi

    local runner=()
    if [ "$(id -u)" -ne 0 ]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is required to install: ${packages[*]}" >&2
            exit 1
        fi
        runner=(sudo)
    fi

    echo "Installing system packages: ${packages[*]}" >&2
    "${runner[@]}" apt-get update
    "${runner[@]}" apt-get install -y --no-install-recommends "${packages[@]}"
}

install_system_packages

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Creating virtual environment at ${VENV_DIR}" >&2
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
PIP=("$VENV_PYTHON" -m pip)

echo "Installing base requirements from requirements.txt" >&2
"${PIP[@]}" install --upgrade pip
"${PIP[@]}" install -r requirements.txt

if [ "$NEED_TORCH" = true ]; then
    echo "Installing: torch, torchvision (CPU wheels)" >&2
    "${PIP[@]}" install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
fi
if [ "$NEED_TENSORFLOW" = true ]; then
    echo "Installing: tensorflow-cpu, tf-keras" >&2
    "${PIP[@]}" install tensorflow-cpu tf-keras
fi
if [ "$NEED_MEDIAPIPE" = true ]; then
    echo "Installing: mediapipe" >&2
    "${PIP[@]}" install mediapipe
fi
if [ "$NEED_ONNXRUNTIME" = true ]; then
    echo "Installing: onnxruntime" >&2
    "${PIP[@]}" install onnxruntime
fi
if [ "$NEED_MIVOLO" = true ]; then
    echo "Installing: ultralytics, timm, safetensors, huggingface_hub" >&2
    "${PIP[@]}" install ultralytics==8.1.0 timm==0.8.13.dev0 safetensors huggingface_hub
fi
if [ "$NEED_SCIPY" = true ]; then
    echo "Installing: scipy" >&2
    "${PIP[@]}" install scipy
fi

# mivolo pulls in opencv-python and mediapipe pulls in opencv-contrib-python, both >=5.0.
# First one to win causes Caffe support loss. Uninstall all variants and reinstall the one
# needed (opencv-contrib-python-headless if lbph is selected, otherwise the smaller
# opencv-python-headless), matching Docker build behavior. See Dockerfile's opencv comment.
OPENCV_PACKAGE="opencv-python-headless"
if csv_has "$RECOGNITION_MODEL" lbph; then
    OPENCV_PACKAGE="opencv-contrib-python-headless"
fi
echo "Installing ${OPENCV_PACKAGE} (selected OpenCV runtime)" >&2
"${PIP[@]}" uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
"${PIP[@]}" install "${OPENCV_PACKAGE}>=4.8.0,<5.0.0"

echo "" >&2
echo "Running at http://localhost:${PORT}" >&2
echo "Press Ctrl-C to stop Streamlit." >&2
export AGE_MODEL GENDER_MODEL RACE_MODEL EMOTION_MODEL RECOGNITION_MODEL \
    FACE_LANDMARKS_MODEL LIVENESS_MODEL \
    GLASSES_MODEL MASK_MODEL HAIR_COLOR_MODEL COLORIZATION_MODEL HAND_MODEL \
    RECONSTRUCTION_3D_MODEL AGE_PROGRESSION_MODEL YOLO_FACE_MODEL \
    SCRFD_FACE_MODEL RETINAFACE_MODEL
exec "${VENV_PYTHON}" -m streamlit run src/app.py \
    --server.address=127.0.0.1 \
    --server.port="${PORT}" \
    --server.enableXsrfProtection=true
