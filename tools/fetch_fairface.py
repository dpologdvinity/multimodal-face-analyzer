"""Download the FairFace validation split used by tools/eval_heldout.py.

FairFace (Karkkainen & Joo, 2021) is CC BY 4.0. This fetches one validation parquet from the
Hugging Face Hub mirror at a pinned revision, without a login, into data/fairface/<config>/
(untracked). Only the validation split is downloaded; the train split is never touched.

The eval uses the 1.25-padding images by default. The 0.25-padding crops are so tight that
the face fills the frame, and the SSD detector then returns mostly out-of-frame boxes (see
docs/eval/heldout_fairface.md), so they measure that detector failure rather than the
attribute models.

    python tools/fetch_fairface.py [--config 1.25] [--dest data/fairface]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent

REPO_ID = "HuggingFaceM4/FairFace"
# Pinned so the evaluated images and labels cannot change underneath a published number.
REVISION = "54d573cdb8b5af490ba8da9da2799628f6e5c496"
# Padding config -> its single validation parquet at REVISION.
FILENAMES = {
    "0.25": "0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet",
    "1.25": "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet",
}


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--config", choices=sorted(FILENAMES), default="1.25",
                        help="face padding of the crops (default 1.25)")
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "fairface")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    local = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
        filename=FILENAMES[args.config], local_dir=args.dest,
    ))
    manifest = {
        "repo_id": REPO_ID, "revision": REVISION, "config": args.config, "split": "validation",
        "file": local.name, "sha256": sha256_of(local), "license": "CC BY 4.0",
    }
    (local.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{local} ({local.stat().st_size / 1e6:.1f} MB) sha256={manifest['sha256']}")


if __name__ == "__main__":
    main()
