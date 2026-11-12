#!/usr/bin/env python3
"""Package the archived training runs for the Zenodo deposit.

Writes an uncompressed tar archive of ``data/runs/`` (its top-level entries
``panel-a/``, ``panel-c/`` and ``README.md``) and a ``SHA256SUMS`` file
listing every member, in exactly the layout ``scripts/download_runs.py``
extracts. The tar is left uncompressed because the checkpoints inside are
zip archives already. Members are added in sorted order with owner fields
cleared, so packaging the same files again yields the same bytes.

Typical usage example:

    uv run --locked python scripts/package_runs.py --out dist
"""

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

ARCHIVE_NAME = "ccn2026-dissociation-runs.tar"
CHECKSUMS_NAME = "SHA256SUMS"
_CHUNK = 1 << 22


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def archive_members(runs_root: Path) -> list[Path]:
    """Return every file of the run archive, sorted, without OS clutter."""
    return sorted(
        path
        for path in runs_root.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dist"),
        help="Directory receiving the tar archive and the checksum file.",
    )
    return parser.parse_args()


def main() -> int:
    """Write the tar archive and the checksum file."""
    args = parse_args()
    members = archive_members(args.runs_root)
    if not members:
        print(f"no files under {args.runs_root}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / ARCHIVE_NAME
    checksums = args.out / CHECKSUMS_NAME

    lines = []
    with tarfile.open(archive, "w") as tar:
        for path in members:
            name = path.relative_to(args.runs_root).as_posix()
            info = tar.gettarinfo(path, arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                tar.addfile(info, handle)
            lines.append(f"{sha256(path)}  {name}\n")
            print(f"added {name} ({path.stat().st_size / 1e6:.1f} MB)")
    checksums.write_text("".join(lines), encoding="utf-8")

    print(
        f"wrote {archive} ({archive.stat().st_size / 1e9:.2f} GB, "
        f"{len(members)} files)\n"
        f"      {checksums}\n"
        f"sha256 of the archive: {sha256(archive)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
