#!/usr/bin/env python3
"""Download the model weights listed in models/manifest.json and verify their sha256."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "models" / "manifest.json"
SOURCES = ("hf", "upstream", "manual")
USER_AGENT = "face-analyzer-fetch-models/1"
CHUNK_SIZE = 1 << 20
TIMEOUT_SECONDS = 60


class FetchError(Exception):
    """Report a download or verification failure for one manifest entry."""


def default_model_dir() -> Path:
    """Return the model directory, honoring FACE_ANALYZER_MODEL_DIR like the app does."""
    return Path(os.environ.get("FACE_ANALYZER_MODEL_DIR") or (REPO_ROOT / "models"))


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    """Load the manifest and return its validated file entries."""
    data = json.loads(path.read_text())
    entries = data.get("files")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'files' must be a list")
    for entry in entries:
        validate_entry(entry)
    return entries


def validate_entry(entry: dict) -> None:
    """Raise ValueError when a manifest entry is missing a field its source needs."""
    path = entry.get("path")
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
        raise ValueError(f"invalid path: {path!r}")
    for field, kind in (("size", int), ("sha256", str), ("license", str), ("features", list), ("models", dict)):
        if not isinstance(entry.get(field), kind):
            raise ValueError(f"{path}: '{field}' must be a {kind.__name__}")
    if len(entry["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in entry["sha256"]):
        raise ValueError(f"{path}: 'sha256' must be 64 lowercase hex digits")
    source = entry.get("source")
    if source not in SOURCES:
        raise ValueError(f"{path}: 'source' must be one of {', '.join(SOURCES)}")
    if source == "manual":
        if not entry.get("instructions"):
            raise ValueError(f"{path}: manual entries need 'instructions'")
    elif not str(entry.get("url", "")).startswith("https://"):
        raise ValueError(f"{path}: {source} entries need an https 'url'")


def parse_selection(tokens: list[str]) -> tuple[set[tuple[str, str]], set[str]]:
    """Split ``ARG=key[,key]`` and bare ``key`` tokens into (arg, key) pairs and bare keys."""
    pairs: set[tuple[str, str]] = set()
    bare: set[str] = set()
    for token in tokens:
        if "=" in token:
            arg, _, values = token.partition("=")
            pairs.update((arg.strip(), key.strip()) for key in values.split(",") if key.strip())
        else:
            bare.update(key.strip() for key in token.split(",") if key.strip())
    return pairs, bare


def select_entries(entries: list[dict], tokens: list[str] | None, select_all: bool) -> list[dict]:
    """Return the required entries plus those matching the selected model keys."""
    if select_all:
        return list(entries)
    pairs, bare = parse_selection(tokens or [])
    known_pairs = {item for entry in entries for item in entry["models"].items()}
    known_keys = {key for _, key in known_pairs}
    unknown = sorted(f"{arg}={key}" for arg, key in pairs - known_pairs) + sorted(bare - known_keys)
    for name in unknown:
        # Keys without weight files (lbph, colorimetric) are valid selections with nothing to fetch.
        print(f"note: no weight files for {name}", file=sys.stderr)
    return [
        entry
        for entry in entries
        if entry.get("required")
        or any((arg, key) in pairs or key in bare for arg, key in entry["models"].items())
    ]


def sha256_of(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, part: Path) -> None:
    """Download url into part, resuming from an existing partial file when the server allows it."""
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        if offset:
            # A stale or oversized partial file can make the Range request fail; start over once.
            print(f"  resume of {part.name} failed (HTTP {exc.code}); restarting", file=sys.stderr)
            part.unlink()
            download(url, part)
            return
        raise FetchError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        raise FetchError(f"cannot reach {url}: {exc}") from exc
    with response:
        resumed = offset and getattr(response, "status", 200) == 206
        if offset and not resumed:
            print(f"  server ignored the resume request; restarting {part.name}", file=sys.stderr)
        try:
            with part.open("ab" if resumed else "wb") as fh:
                shutil.copyfileobj(response, fh, CHUNK_SIZE)
        except (http.client.HTTPException, OSError) as exc:
            # The .part file is kept so the next run resumes from where this one stopped.
            raise FetchError(f"download of {url} incomplete ({exc!r}); rerun to resume") from exc


def fetch_entry(entry: dict, model_dir: Path, force: bool = False) -> str:
    """Make sure one manifest entry is present and verified; return a short status word."""
    dest = model_dir / entry["path"]
    expected = entry["sha256"]
    if dest.is_file():
        actual = sha256_of(dest)
        if actual == expected:
            return "ok"
        if entry["source"] == "manual":
            # Self-supplied files are legitimately produced in different ways, so only warn.
            print(f"warning: {entry['path']} differs from the file this repo was tested with "
                  f"(sha256 {actual}, expected {expected})", file=sys.stderr)
            return "ok"
        if not force:
            raise FetchError(f"{dest} exists but its sha256 is {actual}, expected {expected}; "
                             "rerun with --force to replace it")
    if entry["source"] == "manual":
        instructions = entry["instructions"].format(dest=dest)
        print(f"bring your own: {entry['path']} ({entry['license']})\n  {instructions}", file=sys.stderr)
        return "manual"

    dest.parent.mkdir(parents=True, exist_ok=True)
    member = entry.get("archive_member")
    part = dest.with_name(dest.name + (".zip.part" if member else ".part"))
    print(f"downloading {entry['path']} ({entry['size'] / 1e6:.1f} MB) from {entry['url']}", file=sys.stderr)
    download(entry["url"], part)
    if member:
        extracted = dest.with_name(dest.name + ".tmp")
        try:
            with zipfile.ZipFile(part) as archive:
                member_size = archive.getinfo(member).file_size
                if member_size != entry["size"]:
                    raise FetchError(f"{entry['url']}: {member} is {member_size} bytes, expected {entry['size']}")
                with archive.open(member) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK_SIZE)
        except (zipfile.BadZipFile, KeyError, OSError, FetchError) as exc:
            part.unlink(missing_ok=True)
            extracted.unlink(missing_ok=True)
            if isinstance(exc, FetchError):
                raise
            raise FetchError(f"{entry['url']}: cannot extract {member}: {exc}") from exc
        part.unlink()
        part = extracted
    elif part.stat().st_size < entry["size"]:
        raise FetchError(f"{entry['path']} incomplete ({part.stat().st_size} of {entry['size']} bytes); "
                         "rerun to resume")
    actual = sha256_of(part)
    if actual != expected:
        part.unlink()
        raise FetchError(f"sha256 mismatch for {entry['path']}: got {actual}, expected {expected}")
    os.replace(part, dest)
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--keys", nargs="*", metavar="SELECTION",
        help="ARG=key[,key] pairs as used by the Docker build args (e.g. AGE_MODEL=fairface,caffe) "
             "or bare model keys (e.g. ferplus). Required files are always included.",
    )
    group.add_argument("--all", action="store_true", help="fetch every file in the manifest")
    group.add_argument("--list", action="store_true", help="list the manifest and exit")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="target directory (default: $FACE_ANALYZER_MODEL_DIR or models/)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help="replace existing files whose sha256 differs")
    args = parser.parse_args(argv)

    entries = load_manifest(args.manifest)
    if args.list:
        for entry in entries:
            keys = ", ".join(f"{arg}={key}" for arg, key in entry["models"].items()) or "required"
            print(f"{entry['source']:8} {entry['size'] / 1e6:9.1f} MB  {entry['path']}  [{keys}]")
        return 0

    model_dir = args.model_dir or default_model_dir()
    failures = manual = 0
    for entry in select_entries(entries, args.keys, args.all):
        try:
            status = fetch_entry(entry, model_dir, force=args.force)
        except FetchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
            continue
        manual += status == "manual"
        if status != "manual":
            print(f"{status:10} {entry['path']}", file=sys.stderr)
    if manual:
        print(f"{manual} bring-your-own file(s) missing; those features stay unavailable until supplied.",
              file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
