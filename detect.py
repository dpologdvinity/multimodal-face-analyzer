import argparse
import os
from pathlib import Path
import cv2
import numpy as np

try:
    import torch
    from src.nets.dan_model import DAN
    from src.nets.ssrnet_model import SSRNet
    TORCH_SUPPORTED = True
except ImportError:
    TORCH_SUPPORTED = False

# Directory & Model Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"
GENDER_PROTO = MODEL_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"
EYE_CASCADE_FILE = MODEL_DIR / "haarcascade_eye.xml"
EMOTION_MODEL = MODEL_DIR / "dan_affecnet7.pth"
SSRNET_MODEL = MODEL_DIR / "ssrnet_morph2.pth"

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
EMOTION_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
EMOTION_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SSRNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SSRNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AGE_BACKENDS = ("caffe", "ssrnet")
MIN_EYES_OPEN = 2


def load_networks(age_backend="caffe"):
    """Verify paths and load DNN models into OpenCV. Face detection is required; age, gender,
    drowsiness, and emotion are each optional and skipped (returned as None) if their model
    file(s) aren't present, so the app degrades gracefully to whichever features were built in.
    age_backend selects which age model to load: "caffe" (bucketed) or "ssrnet" (continuous)."""
    if age_backend not in AGE_BACKENDS:
        raise ValueError(f"age_backend must be one of {AGE_BACKENDS}, got {age_backend!r}")

    if not FACE_PROTO.exists() or not FACE_MODEL.exists():
        raise FileNotFoundError(
            f"Missing face detector file(s) in {MODEL_DIR}: {FACE_PROTO.name}, {FACE_MODEL.name} (required)."
        )
    face_net = cv2.dnn.readNetFromTensorflow(str(FACE_MODEL), str(FACE_PROTO))

    age_net = None
    if age_backend == "caffe":
        if AGE_PROTO.exists() and AGE_MODEL.exists():
            age_net = cv2.dnn.readNetFromCaffe(str(AGE_PROTO), str(AGE_MODEL))
        else:
            print("Age detection disabled (model files not found).")
    elif age_backend == "ssrnet":
        if TORCH_SUPPORTED and SSRNET_MODEL.exists():
            age_net = SSRNet()
            checkpoint = torch.load(str(SSRNET_MODEL), map_location="cpu")
            age_net.load_state_dict(checkpoint["state_dict"])
            age_net.eval()
        else:
            print("Age detection disabled (torch or models/ssrnet_morph2.pth not found).")

    gender_net = None
    if GENDER_PROTO.exists() and GENDER_MODEL.exists():
        gender_net = cv2.dnn.readNetFromCaffe(str(GENDER_PROTO), str(GENDER_MODEL))
    else:
        print("Gender detection disabled (model files not found).")

    eye_cascade = None
    if EYE_CASCADE_FILE.exists():
        eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_FILE))
    else:
        print("Drowsiness detection disabled (model file not found).")

    emotion_net = None
    if TORCH_SUPPORTED and EMOTION_MODEL.exists():
        emotion_net = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint = torch.load(str(EMOTION_MODEL), map_location="cpu")
        emotion_net.load_state_dict(checkpoint["model_state_dict"])
        emotion_net.eval()
    else:
        print("Emotion classification disabled (torch or models/dan_affecnet7.pth not found).")

    return face_net, age_net, gender_net, eye_cascade, emotion_net


def predict_emotion(emotion_net, face_bgr):
    """Classify facial expression into one of EMOTION_LABELS."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - EMOTION_MEAN) / EMOTION_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits, _, _ = emotion_net(tensor)
    return EMOTION_LABELS[logits[0].argmax().item()]


def predict_age_ssrnet(age_net, face_bgr):
    """Predict a continuous age with SSR-Net and format it as a label string."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - SSRNET_MEAN) / SSRNET_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        age = age_net(tensor).item()
    return f"{age:.0f}"


def detect_drowsiness(eye_cascade, face_bgr):
    """Return True if fewer than MIN_EYES_OPEN eyes are visible (eyes likely closed)."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    return len(eyes) < MIN_EYES_OPEN


def draw_outlined_text(frame, text, org, color):
    """Draw text with a black outline so it stays readable over any background. Clamps origin so text stays inside the frame."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    frame_h, frame_w = frame.shape[:2]

    x, y = org
    x = max(0, min(x, frame_w - text_w))
    y = max(text_h, min(y, frame_h - baseline))
    org = (x, y)

    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def detect_faces(net, frame, conf_threshold=0.7):
    """Detect faces and return bounding box coordinates."""
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes


def process_image(image_path, face_net, age_net, gender_net, eye_cascade, emotion_net, age_backend="caffe", crop_only=False, show=True, save=False, output_dir="output", conf_threshold=0.7):
    """Run face detection, age prediction, and gender prediction on an image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    face_boxes = detect_faces(face_net, frame, conf_threshold)

    if not face_boxes:
        print(f"[{image_path.name}] No faces detected.")
        return

    annotated_frame = frame.copy()
    out_path_dir = Path(output_dir)

    for idx, (x1, y1, x2, y2) in enumerate(face_boxes, 1):
        # Bound padding checks
        y1_crop = max(0, y1 - 20)
        y2_crop = min(y2 + 20, frame.shape[0] - 1)
        x1_crop = max(0, x1 - 20)
        x2_crop = min(x2 + 20, frame.shape[1] - 1)

        face = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        gender, age = None, None
        if gender_net is not None or (age_net is not None and age_backend == "caffe"):
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            if gender_net is not None:
                gender_net.setInput(blob)
                gender = GENDER_LIST[gender_net.forward()[0].argmax()]
            if age_net is not None and age_backend == "caffe":
                age_net.setInput(blob)
                age = AGE_LIST[age_net.forward()[0].argmax()]
        if age_net is not None and age_backend == "ssrnet":
            age = predict_age_ssrnet(age_net, face)

        emotion = predict_emotion(emotion_net, face) if emotion_net is not None else None
        drowsy = detect_drowsiness(eye_cascade, face) if eye_cascade is not None else None

        label_parts = [p for p in (gender, age, emotion) if p is not None]
        label = ", ".join(label_parts) if label_parts else "Face"

        # Draw box and text on main image frame, black outline for readability
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame.shape[0] / 150)), 8)
        draw_outlined_text(annotated_frame, label, (x1, y1 - 10), (0, 255, 255))

        if drowsy is not None:
            status_label = "DROWSY" if drowsy else "ALERT"
            status_color = (0, 0, 255) if drowsy else (0, 255, 0)
            draw_outlined_text(annotated_frame, status_label, (x1, y1 + (y2 - y1) + 30), status_color)
            if drowsy:
                print(f"[{image_path.name}] Face #{idx}: DROWSINESS DETECTED")

        # Save cropped face
        if save and crop_only:
            os.makedirs(out_path_dir, exist_ok=True)
            crop_out_path = out_path_dir / f"{image_path.stem}_face_{idx}{image_path.suffix}"
            cv2.imwrite(str(crop_out_path), face)
            print(f"Saved crop: {crop_out_path}")

        # Display cropped face window
        if show and crop_only:
            cv2.imshow(f"Cropped Face #{idx} - {label}", face)
            print("Press any key to view next face or image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Save full annotated image
    if save and not crop_only:
        os.makedirs(out_path_dir, exist_ok=True)
        full_out_path = out_path_dir / image_path.name
        cv2.imwrite(str(full_out_path), annotated_frame)
        print(f"Saved annotated image: {full_out_path}")

    # Display full annotated image window
    if show and not crop_only:
        cv2.imshow("Age & Gender Detection", annotated_frame)
        print("Press any key to show next image or exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="CLI tool for Age and Gender Detection from images or folders.")
    parser.add_argument("path", type=str, help="Path to image file or directory containing images")
    parser.add_argument("--crop", action="store_true", help="Display/Save cropped face(s) instead of full image")
    parser.add_argument("--save", action="store_true", help="Save output image(s) to disk")
    parser.add_argument("--no-show", action="store_true", help="Disable GUI pop-up window")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory where results are saved (default: output)")
    parser.add_argument("--conf", type=float, default=0.7, help="Face detection confidence threshold (default: 0.7)")
    parser.add_argument("--age-model", choices=AGE_BACKENDS, default="caffe", help="Age model backend: 'caffe' (bucketed ranges) or 'ssrnet' (continuous age, default: caffe)")

    args = parser.parse_args()

    try:
        face_net, age_net, gender_net, eye_cascade, emotion_net = load_networks(age_backend=args.age_model)
    except Exception as e:
        print(f"Error: {e}")
        return

    input_path = Path(args.path)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if input_path.is_file():
        process_image(
            input_path,
            face_net,
            age_net,
            gender_net,
            eye_cascade,
            emotion_net,
            age_backend=args.age_model,
            crop_only=args.crop,
            show=not args.no_show,
            save=args.save,
            output_dir=args.out_dir,
            conf_threshold=args.conf,
        )
    elif input_path.is_dir():
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
        if not image_files:
            print(f"No valid image files found in {input_path}")
            return
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            process_image(
                img_path,
                face_net,
                age_net,
                gender_net,
                eye_cascade,
                emotion_net,
                age_backend=args.age_model,
                crop_only=args.crop,
                show=not args.no_show,
                save=args.save,
                output_dir=args.out_dir,
                conf_threshold=args.conf,
            )
    else:
        print(f"Error: Path '{input_path}' is not a valid file or directory.")


if __name__ == "__main__":
    main()