import argparse
import os
from pathlib import Path
import cv2
import numpy as np

# Global constants required by the models
MODEL_MEAN_VALUES = (78.4263377603, 87.768914374, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load the pre-trained models
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

# Read the face detection and age prediction models into OpenCV
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)


def detect_faces(net, frame, conf_threshold=0.7):
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame, face_boxes


def predict_age(face, net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    age_preds = net.forward()
    return age_list[age_preds[0].argmax()]


def process_image(image_path, show=True, save=False, output_dir="output"):
    # Load the image from the specified path
    frame = cv2.imread(str(image_path))
    
    # Check if the image is loaded correctly
    if frame is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Detect faces in the image
    frame, face_boxes = detect_faces(face_net, frame)

    # Process each detected face
    for x1, y1, x2, y2 in face_boxes:
        # Prevent indexing out of image bounds
        y1_crop = max(0, y1 - 20)
        y2_crop = min(y2 + 20, frame.shape[0] - 1)
        x1_crop = max(0, x1 - 20)
        x2_crop = min(x2 + 20, frame.shape[1] - 1)

        face = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if face.size == 0:
            continue

        age = predict_age(face, age_net)
        # Add the predicted age as text on the image.
        cv2.putText(
            frame,
            f"Age: {age}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir) / image_path.name
        cv2.imwrite(str(out_path), frame)
        print(f"Saved result: {out_path}")

    if show:
        cv2.imshow("Age Detection", frame)
        print("Press any key to show the next image or exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Age Detection script for single images or directories.")
    parser.add_argument("path", type=str, help="Path to an image file or directory containing images")
    parser.add_argument("--save", action="store_true", help="Save output images to disk")
    parser.add_argument("--no-show", action="store_true", help="Do not open display window")
    args = parser.parse_args()

    input_path = Path(args.path)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if input_path.is_file():
        process_image(input_path, show=not args.no_show, save=args.save)
    elif input_path.is_dir():
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
        if not image_files:
            print(f"No valid image files found in {input_path}")
            return
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            process_image(img_path, show=not args.no_show, save=args.save)
    else:
        print(f"Error: {input_path} is not a valid file or directory.")


if __name__ == "__main__":
    main()