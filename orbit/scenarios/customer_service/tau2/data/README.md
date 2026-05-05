# τ²-Bench vendored data

This directory contains data files vendored from
[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
(MIT license).

- **Upstream commit**: `a11d95e19a049638dd3fd7c5b941a509422452dc`
- **Upstream path**: `data/tau2/domains/<domain>/` → `<domain>/`

## Files

### `airline/` (PR 1)
- `db.json` — initial flight/user/reservation state (~7 MB)
- `tasks.json` — 50 airline tasks
- `policy.md` — airline customer-service agent policy (~8 KB)

### `retail/` (PR 3)
- `db.json` — initial products/users/orders state (~2.8 MB)
- `tasks.json` — 114 retail tasks
- `policy.md` — retail customer-service agent policy (~7 KB)

### `telecom/` (PR 3, small set only)
- `db.toml` — agent-side domain DB (customers, lines, plans)
- `user_db.toml` — user-side device / surroundings DB
- `tasks_small.json` — 20 curated tasks (upstream's regression subset)
- `main_policy.md` — telecom agent main policy
- `tech_support_manual.md` — tech-support procedures reference
- `tech_support_workflow.md` — tech-support workflow flowcharts

All three telecom markdown files are inlined into the assistant system
prompt verbatim (~33 KB combined) to match upstream.

## Fidelity notes

- **Airline solo mode** (PR 1): upstream tau2 explicitly rejects solo
  mode for the airline domain, so airline-solo numbers from this port
  are an Orbit-specific baseline and are not directly comparable to
  published tau2 airline results. PR 2's `dual_control` mode is the
  upstream-faithful way to run airline.
- **Telecom full set**: upstream's full `tasks.json` (2285 tasks,
  14 MB) is not vendored. Point `Tau2ScenarioConfig.data_path` (or the
  `-T data_path=...` CLI flag) at a locally-downloaded copy if you
  want to run the full set. The small set (20 tasks) is what upstream
  uses for most paper numbers.
- **Retail** uses `reward_basis=['DB', 'NL_ASSERTION']` on 112/114
  tasks, so the LLM judge added in PR 2 is finally exercised for real
  (airline never uses NL_ASSERTION).
- **Telecom (small)** uses `reward_basis=['ENV_ASSERTION']` or
  `['ACTION', 'ENV_ASSERTION']`. The ENV_ASSERTION component is a PR 3
  scorer addition.
