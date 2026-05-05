# BrowserART data

## What Orbit ships in this directory

- **`hbb_extension.json`** — *Orbit-original* content. A set of
  multi-step BrowserART behaviours authored in-house (all entries
  tagged `"source": "Original-MultiStep"`). Licensed under Apache-2.0
  with the rest of Orbit (see `/LICENSE`).

## What Orbit does **not** ship here

Orbit does **not** redistribute the upstream BrowserART datasets:

- `hbb.json` — the primary BrowserART behaviours listing (sourced from
  HarmBench, AirBench 2024, and Scale originals)
- `hbb_benign.json` — the benign counterpart dataset
- `hbb/behaviors.json` — the BrowserGym task configuration file

### Why

The upstream repository `scaleapi/browser-art` ships with a
**CC BY-NC-ND 4.0** `LICENSE` file (verified 2026-04-17). The upstream
README text claims MIT, but where `LICENSE` and `README` disagree, the
`LICENSE` file controls. CC BY-NC-ND 4.0 prohibits commercial use and
forbids derivative works, which is incompatible with Orbit's goal of
being a redistributable research toolkit under Apache-2.0. Rather than
propagate the non-commercial / no-derivatives restriction to everyone
who installs Orbit, we require users to fetch the upstream data
themselves from its authoritative source.

## How to obtain the data

### Option A: run the fetch script (recommended)

```bash
uv run python scripts/fetch_browserart_data.py
```

This downloads the three files from upstream `scaleapi/browser-art`
at a pinned commit (see the script's `REVISION` constant) and writes
them into this directory. The downloads are subject to the upstream
CC BY-NC-ND 4.0 license — do not redistribute them.

### Option B: clone upstream and point Orbit at it

```bash
git clone https://github.com/scaleapi/browser-art.git
orbit browserart -m openai/gpt-4o -T data_path=/path/to/browser-art/src/datasets/behaviors
```

## Upstream attribution

- Repository: https://github.com/scaleapi/browser-art
- `LICENSE` file content: CC BY-NC-ND 4.0 (verified 2026-04-17)
- README claims MIT; upstream issue tracker should clarify which is
  authoritative.

BrowserART's HBB behaviours are themselves compiled from multiple
upstreams:

- **HarmBench** (MIT) — https://github.com/centerforaisafety/HarmBench
- **AirBench 2024** (Apache-2.0) — https://github.com/stanford-crfm/air-bench-2024
- **Original** — Scale AI original contributions

When you fetch the upstream data, the compiled listing inherits all
three licenses for its respective rows; see each row's `"source"`
field.

## NeurIPS paper note

The paper should reflect that Orbit does not redistribute BrowserART's
upstream datasets, and that users fetch the data directly from the
authoritative upstream under its own license terms.
