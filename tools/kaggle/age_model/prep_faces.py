"""Record the app's face crop (box, roll angle, landmarks) for every FairFace TRAIN image.

Runs in the CPU prep kernel (see kernel_prep.py). Downloads the 1.25-padding TRAIN parquets
from the same Hugging Face mirror and revision as tools/fetch_fairface.py; the validation split
is never downloaded. Writes faces_train.npz (labels plus one crop record per image),
reference_crops.npz (aligned faces built by the app itself, for the training kernel's parity
check) and prep_summary.json.

    python prep_faces.py --out /kaggle/working [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fetch_fairface import REPO_ID, REVISION  # noqa: E402

CONFIG = "1.25"
REFERENCE_CROPS = 64
_STATE: dict = {}


def train_files() -> list[str]:
    """List the TRAIN parquet files of the padding config at the pinned revision, sorted."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=REVISION)
    chosen = sorted(f for f in files if f.startswith(f"{CONFIG}/train-") and f.endswith(".parquet"))
    if not chosen or any("validation" in f for f in chosen):
        raise SystemExit(f"unexpected file list: {chosen}")
    return chosen


def _init_worker(parquets: list[str]) -> None:
    """Load the prep models and the image columns once per worker process."""
    import cv2
    import pyarrow.parquet as pq

    cv2.setNumThreads(1)
    import faceprep

    _STATE["models"] = faceprep.load_prep_models()
    _STATE["images"] = [pq.read_table(p, columns=["image"]).column("image").combine_chunks().field("bytes")
                        for p in parquets]


def _process(task: tuple[int, int]) -> tuple[int, int, dict | None, np.ndarray | None]:
    """Prepare one image; return (shard, row, crop record or None, app-built crop if a reference)."""
    import cv2
    import faceprep

    shard, row = task
    frame = cv2.imdecode(np.frombuffer(_STATE["images"][shard][row].as_py(), np.uint8), cv2.IMREAD_COLOR)
    result = faceprep.face_record(_STATE["models"], frame)
    if result is None:
        return shard, row, None, None
    record, inputs = result
    reference = None
    if row < REFERENCE_CROPS and shard == 0:
        reference = faceprep.app_aligned_face(inputs)
        rebuilt = faceprep.aligned_face(frame, record)
        if not np.array_equal(reference, rebuilt):
            raise RuntimeError(f"crop record does not rebuild the app crop for shard {shard} row {row}")
    return shard, row, record, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/tmp/age/fairface"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="images per shard (smoke test)")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    started = time.time()
    names = train_files()
    parquets = [hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION,
                                local_dir=args.data) for name in names]
    labels = {key: [] for key in ("shard", "row", "age", "gender", "race")}
    for shard, path in enumerate(parquets):
        table = pq.read_table(path, columns=["age", "gender", "race"])
        n = table.num_rows if not args.limit else min(args.limit, table.num_rows)
        labels["shard"] += [shard] * n
        labels["row"] += list(range(n))
        for key in ("age", "gender", "race"):
            labels[key] += table.column(key).to_pylist()[:n]
    total = len(labels["row"])
    print(f"{total} TRAIN images in {len(parquets)} shards", flush=True)

    boxes = np.full((total, 4), -1, np.int32)
    angles = np.full(total, np.nan, np.float64)
    landmarks = np.full((total, 5, 2), np.nan, np.float32)
    detected = np.zeros(total, bool)
    references: dict[int, np.ndarray] = {}
    index = {(s, r): i for i, (s, r) in enumerate(zip(labels["shard"], labels["row"], strict=True))}
    tasks = list(index)
    with mp.get_context("fork").Pool(args.workers, _init_worker, (parquets,)) as pool:
        for count, (shard, row, record, reference) in enumerate(
                pool.imap_unordered(_process, tasks, chunksize=64), 1):
            i = index[(shard, row)]
            if record is not None:
                detected[i] = True
                boxes[i] = record["box"]
                if record["angle"] is not None:
                    angles[i] = record["angle"]
                if record["landmarks"] is not None:
                    landmarks[i] = record["landmarks"]
            if reference is not None:
                references[i] = reference
            if count % 2000 == 0 or count == total:
                rate = count / (time.time() - started)
                print(f"{count}/{total} images, {rate:.1f}/s, ETA {(total - count) / rate / 60:.0f} min",
                      flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "faces_train.npz",
        shard=np.asarray(labels["shard"], np.int16), row=np.asarray(labels["row"], np.int32),
        age=np.asarray(labels["age"], np.int8), gender=np.asarray(labels["gender"], np.int8),
        race=np.asarray(labels["race"], np.int8), detected=detected, box=boxes, angle=angles,
        landmarks=landmarks, files=np.asarray(names),
    )
    ref_index = sorted(references)
    np.savez_compressed(args.out / "reference_crops.npz", index=np.asarray(ref_index, np.int64),
                        crops=np.stack([references[i] for i in ref_index]))
    from importlib import metadata

    import cv2

    summary = {
        "dataset": {"repo_id": REPO_ID, "revision": REVISION, "config": CONFIG, "split": "train",
                    "files": names, "license": "CC BY 4.0"},
        "images": total, "detected": int(detected.sum()),
        "landmark_aligned": int(np.isfinite(landmarks[:, 0, 0]).sum()),
        "roll_leveled": int((np.abs(np.nan_to_num(angles)) > 3).sum()),
        "reference_crops_checked": len(ref_index),
        "minutes": round((time.time() - started) / 60, 1),
        "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                     **{name: metadata.version(name) for name in ("numpy", "mediapipe", "pyarrow")}},
    }
    (args.out / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
