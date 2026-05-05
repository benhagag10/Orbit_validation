#!/usr/bin/env python3
"""Fetch RedCode-Gen upstream data at a pinned commit.

Orbit does not redistribute the RedCode-Gen malware-stub dataset
because the upstream ``AI-secure/RedCode`` repository has no LICENSE
file. This script downloads the 160 Python function stubs (8
categories × 20 variants) needed to run the RedCode-Gen scenario.

Downloaded content inherits whatever the upstream ultimately declares
its license to be. The paper (Guo et al., NeurIPS 2024 D&B) declares
the *dataset* under CC BY 4.0. See
``orbit/scenarios/coding/redcode_gen/data/README.md`` for full attribution.

Pinned revision:
    ``f3b9a8a63a1b6deef0f4af7c643b0da185517ea6``
    (https://github.com/AI-secure/RedCode)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REVISION = "f3b9a8a63a1b6deef0f4af7c643b0da185517ea6"
TREE_API = (
    f"https://api.github.com/repos/AI-secure/RedCode/git/trees/"
    f"{REVISION}?recursive=1"
)
BASE_URL = f"https://raw.githubusercontent.com/AI-secure/RedCode/{REVISION}"

CATEGORIES = (
    "adware", "ddos", "others", "ransomware",
    "rootkit", "spyware", "trojan", "virus",
)
UPSTREAM_PREFIX = "dataset/RedCode-Gen/"
DEST_REL = "orbit/scenarios/coding/redcode_gen/data"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str) -> bytes:
    try:
        with urlopen(url) as resp:
            return resp.read()
    except (HTTPError, URLError) as exc:
        raise SystemExit(
            f"Failed to download {url}: {exc}. Check network and the "
            f"pinned revision."
        ) from exc


def _list_upstream_py_files() -> list[str]:
    """Fetch the pinned tree and return every .py path under RedCode-Gen/."""
    import json

    raw = _download(TREE_API)
    tree = json.loads(raw).get("tree", [])
    paths = [
        entry["path"]
        for entry in tree
        if entry.get("type") == "blob"
        and entry["path"].startswith(UPSTREAM_PREFIX)
        and entry["path"].endswith(".py")
    ]
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a file already exists.",
    )
    args = parser.parse_args()

    print(
        "RedCode-Gen upstream has no LICENSE file. The paper declares the\n"
        "dataset under CC BY 4.0; the code repository itself is\n"
        "undeclared. By running this script you fetch malware-stub code\n"
        "for AI safety evaluation only.\n"
    )

    data_root = args.repo_root / DEST_REL
    data_root.mkdir(parents=True, exist_ok=True)

    paths = _list_upstream_py_files()
    if not paths:
        raise SystemExit(
            f"No .py files found under {UPSTREAM_PREFIX} at revision "
            f"{REVISION}. Upstream layout may have changed."
        )

    any_changed = False
    for upstream_path in paths:
        rel = upstream_path[len(UPSTREAM_PREFIX):]  # e.g. adware/adware_1_email.py
        dst = data_root / rel
        if dst.exists() and not args.force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{upstream_path}"
        dst.write_bytes(_download(url))
        any_changed = True
        print(f"↓ {rel}")

    # Sanity check: every declared category should have at least one file.
    missing = [c for c in CATEGORIES if not (data_root / c).exists()]
    if missing:
        raise SystemExit(
            f"Missing expected categories after fetch: {missing}. "
            f"Upstream layout may have changed."
        )

    print(
        f"\nRedCode-Gen data ready under {data_root}. "
        f"Fetched {len(paths)} files. "
        f"{'Updated' if any_changed else 'No changes needed'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
