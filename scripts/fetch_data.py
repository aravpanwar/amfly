"""Fetch MaleCNS v1.0 and verify against pinned hashes.

    python scripts/fetch_data.py --out data

About 1.1GB total. Files are verified by SHA-256, not by size alone, so a
truncated or silently updated download fails here rather than three
milestones later when the numbers have quietly changed.

MaleCNS v1.0 is CC BY 4.0. Credit Janelia FlyEM and Google Research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "manifest.json"
CHUNK = 1 << 22  # 4MB


def sha256_of(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def fetch(url: str, dest: Path, expect_bytes: int) -> None:
    print(f"  fetching {dest.name} ({expect_bytes / 1e6:.0f} MB)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    got = 0
    with urlopen(url) as r, open(tmp, "wb") as out:
        while chunk := r.read(CHUNK):
            out.write(chunk)
            got += len(chunk)
            pct = 100.0 * got / expect_bytes if expect_bytes else 0.0
            print(f"\r    {got / 1e6:8.0f} MB  {pct:5.1f}%", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--force", action="store_true", help="refetch even if present")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    base = manifest["base_url"]

    print(f"{manifest['dataset']}  ({manifest['license']})")
    print(f"source: {manifest['source']}\n")

    ok = True
    for local, spec in manifest["files"].items():
        dest = args.out / local
        if dest.exists() and not args.force:
            print(f"  {local}: present, verifying")
        else:
            fetch(base + spec["remote"], dest, spec["bytes"])

        digest, size = sha256_of(dest)
        if size != spec["bytes"]:
            print(f"    FAIL size {size} != {spec['bytes']}")
            ok = False
        elif digest != spec["sha256"]:
            print(f"    FAIL sha256 mismatch\n      got      {digest}"
                  f"\n      expected {spec['sha256']}")
            ok = False
        else:
            print(f"    ok  {digest[:16]}...")

    if not ok:
        print("\nverification failed; refusing to proceed")
        return 1

    counts = manifest["derived_counts"]
    print(f"\nverified. expect {counts['neurons']:,} neurons, "
          f"{counts['connections']:,} connections, "
          f"{counts['synapses']:,} synapses")
    print("next: python -m pytest tests/test_loader.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
