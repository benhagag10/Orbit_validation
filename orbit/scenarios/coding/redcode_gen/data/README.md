# RedCode-Gen data

## What Orbit ships in this directory

Nothing. This directory exists only as a package marker so that
`importlib.resources` can locate the scenario root.

## What Orbit does **not** ship here

Orbit does **not** redistribute the RedCode-Gen malware function-stub
dataset — 160 Python files across eight categories
(`adware`, `ddos`, `others`, `ransomware`, `rootkit`, `spyware`,
`trojan`, `virus`) sourced from `AI-secure/RedCode`.

### Why

The upstream repository `AI-secure/RedCode` has **no `LICENSE` file**
(`gh api repos/AI-secure/RedCode --jq '.license.spdx_id'` returns
`null`, verified 2026-04-17). The companion paper (Guo et al.,
*RedCode: Risky Code Execution and Generation Benchmark for Code
Agents*, NeurIPS 2024 D&B) declares the *dataset* as CC BY 4.0, but
the status of the code snippets in the repository itself is
undeclared.

We decline to redistribute code of undeclared license that also
happens to be malware-implementation stubs. Users who opt into running
the scenario fetch the data directly from the authoritative upstream.

## How to obtain the data

### Option A: run the fetch script (recommended)

```bash
uv run python scripts/fetch_redcode_data.py
```

This downloads all 160 function-stub files from upstream
`AI-secure/RedCode` at a pinned commit (see the script's `REVISION`
constant) and writes them into this directory.

### Option B: clone upstream and point Orbit at it

```bash
git clone https://github.com/AI-secure/RedCode.git
orbit run examples/redcode_gen.yaml \
  -T data_path=/path/to/RedCode/dataset/RedCode-Gen
```

## Upstream attribution

- Repository: https://github.com/AI-secure/RedCode (no `LICENSE` file
  as of 2026-04-17)
- Paper: Guo et al., *RedCode: Risky Code Execution and Generation
  Benchmark for Code Agents*, NeurIPS 2024 Datasets & Benchmarks
  (https://arxiv.org/abs/2411.07781)
- Paper-declared dataset license: CC BY 4.0
- Repository code license: undeclared

## Safety note

The RedCode-Gen function stubs describe implementations of
malicious software (adware, ransomware, trojans, etc.) — they are
*not* working malware, only function signatures with docstrings used
as code-generation prompts. Orbit's role is to evaluate whether an
LLM agent refuses or completes these prompts. Do not repurpose the
fetched data for any use outside AI safety and security research.
