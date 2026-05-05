#!/usr/bin/env python3
"""Fetch BrowserART upstream data at a pinned commit.

Orbit does not redistribute BrowserART's upstream datasets because the
upstream `scaleapi/browser-art` LICENSE file is CC BY-NC-ND 4.0, which
is incompatible with Orbit's Apache-2.0 license. This script downloads
the three files needed to run the BrowserART scenarios from upstream
directly.

Downloaded content is subject to the upstream license; do not
redistribute from Orbit's tree. See
``orbit/scenarios/browser/browserart/data/README.md``.

Pinned revision:
    ``0d72180042f2a076c68e1114e7494cb3fc7dd30b``
    (https://github.com/scaleapi/browser-art)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REVISION = "0d72180042f2a076c68e1114e7494cb3fc7dd30b"
BASE_URL = f"https://raw.githubusercontent.com/scaleapi/browser-art/{REVISION}"

# (upstream_path, destination relative to repo root)
FILES = (
    (
        "src/datasets/behaviors/hbb.json",
        "orbit/scenarios/browser/browserart/data/hbb.json",
    ),
    (
        "src/datasets/behaviors/hbb_benign.json",
        "orbit/scenarios/browser/browserart/data/hbb_benign.json",
    ),
    (
        "src/agents/OpenDevin/BrowserGym/hbb/src/browsergym/hbb/behaviors.json",
        "orbit/scenarios/browser/browserart/hbb/behaviors.json",
    ),
)


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
        "BrowserART upstream data is licensed under CC BY-NC-ND 4.0 per\n"
        "scaleapi/browser-art's LICENSE file. By running this script you\n"
        "download that data subject to its upstream terms. Do not\n"
        "redistribute it from Orbit's tree.\n"
    )

    any_changed = False
    for upstream_path, dest_rel in FILES:
        dst = args.repo_root / dest_rel
        if dst.exists() and not args.force:
            print(f"✓ {dest_rel} (already present, sha256={_sha256(dst)[:12]})")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{upstream_path}"
        print(f"↓ downloading {dest_rel} from {url}")
        dst.write_bytes(_download(url))
        any_changed = True

    print(
        f"\nBrowserART data ready under {args.repo_root}. "
        f"{'Updated' if any_changed else 'No changes needed'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
