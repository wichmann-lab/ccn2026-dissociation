#!/usr/bin/env python3
"""Download the archived training runs behind Figure 1 into ``data/runs/``.

The runs (per-run ``checkpoint.pt``, ``config.yaml`` and ``train.log``) are
deposited on Zenodo. This script fetches every file of the record, verifies the
MD5 checksum reported by the Zenodo API and extracts archives in place,
producing the layout that ``scripts/precompute_results.py`` expects::

    data/runs/
    ├── panel-a/<model_pair>/<measure>/target_<x>/
    └── panel-c/{alexnet-cifar10,baselines/run_<seed>,trained/run_<k>}/

Typical usage example:

    uv run --locked python scripts/download_runs.py
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Zenodo record of the archived runs (doi:10.5281/zenodo.24162774).
ZENODO_RECORD = "24162774"
ZENODO_API = "https://zenodo.org/api/records/{record}"

_CHUNK = 1 << 20


def _record_files(record: str) -> list[dict]:
    with urllib.request.urlopen(ZENODO_API.format(record=record)) as response:  # noqa: S310
        payload = json.load(response)
    return payload["files"]


def _download(url: str, target: Path) -> str:
    """Stream ``url`` to ``target`` and return the file's MD5 hex digest."""
    digest = hashlib.md5()  # noqa: S324 - Zenodo publishes MD5 checksums
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as out:  # noqa: S310
        while chunk := response.read(_CHUNK):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: Path, destination: Path) -> bool:
    """Extract a tar or zip archive; return False for other files."""
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            # filter="data" rejects absolute paths, links and traversal.
            handle.extractall(destination, filter="data")
        return True
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            root = destination.resolve()
            for member in handle.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    msg = f"{archive}: unsafe member path {member.filename!r}"
                    raise ValueError(msg)
            handle.extractall(destination)  # noqa: S202 - members checked
        return True
    return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--record",
        default=ZENODO_RECORD,
        help="Zenodo record id (default: the deposited archive).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("data/runs"),
        help="Where to place the extracted runs.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the downloaded archives next to the extracted runs.",
    )
    return parser.parse_args()


def main() -> int:
    """Download, verify and extract every file of the Zenodo record."""
    args = parse_args()

    files = _record_files(args.record)
    print(f"Zenodo record {args.record}: {len(files)} file(s)")
    for entry in files:
        name = entry["key"]
        expected = entry["checksum"].removeprefix("md5:")
        target = args.runs_root / name
        print(f"downloading {name} ({entry['size'] / 1e6:.0f} MB)")
        actual = _download(entry["links"]["self"], target)
        if actual != expected:
            print(f"checksum mismatch for {name}", file=sys.stderr)
            return 1
        if _extract(target, args.runs_root) and not args.keep_archives:
            target.unlink()
    shutil.rmtree(args.runs_root / "__MACOSX", ignore_errors=True)
    print(f"runs are in {args.runs_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
