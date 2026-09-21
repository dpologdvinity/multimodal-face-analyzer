"""Draw numbered detection boxes on every assets/ image, for hand-labelling ground truth.

Writes tmp/annotated/<name>.png and prints each face's normalized centre, which is the key
assets/ground_truth.json uses to bind a label to a face independently of detector ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import inference  # noqa: E402


def main() -> None:
    models = inference.load_models()
    out_dir = ROOT / "tmp" / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "assets").iterdir()):
        if path.suffix.lower() not in inference.IMAGE_FILE_EXTENSIONS:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        boxes = inference.detect_faces(models.face_net, frame, conf_threshold=0.5)
        boxes.sort(key=lambda b: (b[0], b[1]))
        print(f"\n{path.name}  ({width}x{height})  faces={len(boxes)}")
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            print(f"  [{index}] center=({cx:.3f}, {cy:.3f}) box=({x1},{y1},{x2},{y2})")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, str(index), (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), frame)


if __name__ == "__main__":
    main()
