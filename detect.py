import argparse
import os
from pathlib import Path
import cv2
import numpy as np

# Directory & Model Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"
GENDER_PROTO = MODEL_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


def load_networks():
    """Verify paths and load DNN models into OpenCV."""
    required_files = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {file_path}\n"
                f"Ensure all weights and configs are placed inside: {MODEL_DIR}"
            )

    face_net = cv2.dnn.readNetFromTensorflow(str(FACE_MODEL), str(FACE_PROTO))
    age_net = cv2.dnn.readNetFromCaffe(str(AGE_PROTO), str(AGE_MODEL))
    gender_net = cv2.dnn.readNetFromCaffe(str(GENDER_PROTO), str(GENDER_MODEL))

    return face_net, age_net, gender_net


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


def process_image(image_path, face_net, age_net, gender_net, crop_only=False, show=True, save=False, output_dir="output", conf_threshold=0.7):
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

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward()[0].argmax()]

        # Predict Age
        age_net.setInput(blob)
        age = AGE_LIST[age_net.forward()[0].argmax()]

        label = f"{gender}, {age}"

        # Draw box and text on main image frame, black outline for readability
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame.shape[0] / 150)), 8)
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

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

    args = parser.parse_args()

    try:
        face_net, age_net, gender_net = load_networks()
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