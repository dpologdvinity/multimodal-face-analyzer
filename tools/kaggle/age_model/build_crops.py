"""Rebuild every FairFace TRAIN face once into a uint8 array of the app's aligned 224x224 crops.

Runs in the CPU crops kernel (see crops/kernel_crops.py) on the prep kernel's crop records
(prep_faces.py), with faceprep.aligned_face, so the GPU kernel only reads finished crops from a
memory-mapped file instead of decoding and warping JPEGs on its few CPU cores. Undetected
images stay all-zero and are never used. The app-built reference crops must come out
identical, or the run stops.

    python build_crops.py --prep DIR --out DIR [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fetch_fairface import REPO_ID, REVISION  # noqa: E402

SIZE = 224


def crop_record(meta, i: int) -> dict:
    """Return sample i's crop record in the form faceprep.aligned_face takes."""
    angle = meta["angle"][i]
    return {"box": meta["box"][i], "angle": None if np.isnan(angle) else float(angle),
            "landmarks": meta["landmarks"][i]}


def _build_shard(task: tuple[int, str, str, str]) -> tuple[int, int, float]:
    """Write the crops of one parquet shard into the shared output array."""
    import cv2
    import faceprep
    import pyarrow.parquet as pq

    cv2.setNumThreads(1)
    shard, parquet, prep_path, out_path = task
    started = time.time()
    meta = dict(np.load(prep_path))
    crops = np.load(out_path, mmap_mode="r+")
    images = pq.read_table(parquet, columns=["image"]).column("image").combine_chunks().field("bytes")
    mine = np.flatnonzero((meta["shard"] == shard) & meta["detected"])
    for i in mine:
        frame = cv2.imdecode(np.frombuffer(images[int(meta["row"][i])].as_py(), np.uint8), cv2.IMREAD_COLOR)
        crops[i] = faceprep.aligned_face(frame, crop_record(meta, int(i)))
    crops.flush()
    return shard, len(mine), time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--prep", type=Path, required=True, help="prep kernel output directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/tmp/age/fairface"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    started = time.time()
    prep_path = args.prep / "faces_train.npz"
    prep = np.load(prep_path)
    files = [str(f) for f in prep["files"]]
    if any("validation" in f or not f.startswith("1.25/train-") for f in files):
        raise SystemExit(f"refusing non-TRAIN inputs: {files}")
    parquets = [hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION, local_dir=args.data)
                for name in files]
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "crops_train.npy"
    total = len(prep["row"])
    np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(total, SIZE, SIZE, 3)).flush()
    tasks = [(shard, str(path), str(prep_path), str(out_path)) for shard, path in enumerate(parquets)]
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        shards = pool.map(_build_shard, tasks)
    for shard, count, seconds in shards:
        print(f"shard {shard}: {count} crops in {seconds / 60:.1f} min", flush=True)

    crops = np.load(out_path, mmap_mode="r")
    refs = np.load(args.prep / "reference_crops.npz")
    for i, crop in zip(refs["index"], refs["crops"], strict=True):
        if not np.array_equal(crops[int(i)], crop):
            raise SystemExit(f"crop {i} differs from the app-built reference crop")
    import cv2

    summary = {
        "file": out_path.name, "shape": list(crops.shape), "dtype": "uint8", "channel_order": "BGR",
        "built": int(sum(count for _, count, _ in shards)), "reference_crops_identical": len(refs["index"]),
        "minutes": round((time.time() - started) / 60, 1), "opencv": cv2.__version__,
        "numpy": np.__version__, "python": sys.version.split()[0],
        "disk": shutil.disk_usage(args.out)._asdict(),
        "memory": subprocess.run(["free", "-m"], capture_output=True, text=True).stdout,
        "cpus": mp.cpu_count(),
    }
    (args.out / "crops_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
