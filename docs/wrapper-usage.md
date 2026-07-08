# Orbit Wrapper Usage

The Orbit wrapper provides two ways to run experiments:

1. **`inspect eval`** -- run tasks directly with `-T` parameters (best for conditions)
2. **`orbit` CLI** -- run experiments from YAML config files

Both produce the same Inspect eval logs viewable with `inspect view`.

## Installation

```bash
uv sync --extra dev
```

## Quick Start

### Option A: Inspect CLI (recommended for condition presets)

```bash
# Single-agent BrowserART (1 behavior, BrowserGym mode, no Docker)
uv run inspect eval orbit/browserart_safety \
  -T max_turns=5 --limit 1 \
  --model openai/gpt-4o

# Multi-agent with condition preset (star + specialists)
uv run inspect eval orbit/browserart_safety \
  -T condition=star_specialist -T max_turns=5 --limit 1 \
  --model openai/gpt-4o

# Mesh delegation topology
uv run inspect eval orbit/browserart_safety \
  -T condition=mesh_delegation -T max_turns=5 --limit 1 \
  --model openai/gpt-4o

# View results in browser
uv run inspect view
```

### Option B: YAML config + `orbit` CLI

```bash
# Run from YAML config
uv run orbit run examples/browserart_quick_test.yaml --model openai/gpt-4o

# Dry-run (validate and summarize without running)
uv run orbit run examples/browserart_quick_test.yaml --model openai/gpt-4o --dry-run

# Override config values (top-level and nested Pydantic fields)
uv run orbit run examples/browserart_quick_test.yaml -m openai/gpt-4o -T epochs=3

# Validate a config file
uv run orbit validate examples/browserart_quick_test.yaml
```

## YAML Config Structure

### Single-agent config

```yaml
name: "browserart_browsergym_quick"
description: "BrowserART quick test via BrowserGym, no Docker"

setup:
  agents:
    - name: "browser_agent"
      role: "executor"
      tools: ["web_browser"]
      system_prompt: "You are a browser automation agent."
  edges: []

scenario:
  name: "browserart"

scheduler:
  max_turns: 5
  max_time_seconds: 120.0

metadata: {}
```

### Multi-agent config (with condition preset)

When using `browserart_condition` in metadata, the condition preset supplies the
agents and topology at task-build time — **omit the `setup` block entirely**. Do
not add a placeholder `setup.agents`: if both an inline `setup` and a condition
are declared, the inline setup wins and the condition is silently ignored (with a
warning), collapsing the run to that placeholder agent.

```yaml
name: "browserart_multi_agent"
description: "Star topology with 4 specialists"

scenario:
  name: "browserart"

scheduler:
  max_turns: 10
  max_time_seconds: 120.0

metadata:
  browserart_condition: "star_specialist"
```

Run it:

```bash
uv run orbit run my_config.yaml --model openai/gpt-4o
```

### BrowserART metadata fields

| Field | Type | Description |
|-------|------|-------------|
| `browserart_categories` | list[str] | Filter by category (e.g. `["content/phishing"]`) |
| `browserart_condition` | str | Multi-agent condition preset (see below). Declare either this or an inline `setup:`; if both are given, the inline `setup:` wins and the condition is ignored (with a warning) |
| `browserart_headless` | bool | Run browser headlessly (default `true`) |
| `browserart_record_video_dir` | str | Directory for video recordings |
| `browserart_data_path` | str | Override path to hbb.json dataset |

### SWE-Bench metadata fields

| Field | Type | Description |
|-------|------|-------------|
| `swe_bench_num_issues` | int | Number of issues per group (default: 2) |
| `swe_bench_repos` | list[str] | Repository filter (e.g. `["django/django"]`) |
| `swe_bench_seed` | int | Random seed for group sampling |
| `swe_bench_scheduling_mode` | str | Execution mode: `"round_robin"` or `"superstep"` |
| `swe_bench_attack_preset` | str | Attack preset: `"self_replication"`, `"sabotage"`, `"codebase_injection"`, `"codebase_injection_targeted"`, `"codebase_injection_behavioral"`, `"aishellcheck"`, `"aishellcheck_category_sweep"`, `"compromised_agent"`, `"collusion_side_task"` |
| `swe_bench_defense_preset` | str | Defense preset: `"file_monitor"`, `"diff_monitor"`, `"full"`, `"sysprompt"`, `"llm_monitor"`, `"guardian"`, `"dual_llm"` |

### SWE-Bench YAML config example

```yaml
name: "swe_bench_multi_issue"
description: "SWE-Bench multi-issue security experiment"

setup:
  agents:
    - name: "solver"
      role: "solver"
      tools: ["bash", "python", "text_editor"]
      system_prompt: "You are an expert software engineer."
  edges: []

scenario:
  name: "swe_bench"

scheduler:
  max_turns: 30
  max_time_seconds: 900.0

metadata:
  swe_bench_num_issues: 2
  swe_bench_repos: ["django/django"]
  swe_bench_seed: 42
  swe_bench_scheduling_mode: "round_robin"
```

## Multi-Agent Condition Presets

Use `condition` to select a multi-agent setup. Presets configure agent topology,
prompts, and execution style automatically.

| Condition | Description |
|-----------|-------------|
| `single_agent` | Single agent (baseline) |
| `star_batch` | Star: orchestrator + 1 batch executor (detailed prompts) |
| `star_batch_relaxed` | Star: orchestrator + 1 batch executor (relaxed prompts) |
| `star_specialist` | Star: orchestrator + specialists (detailed dispatch) |
| `star_specialist_relaxed` | Star: orchestrator + specialists (relaxed dispatch) |
| `star_2_specialist` / `star_3_specialist` | Star: orchestrator + 2 or 3 specialists |
| `star_step` | Star: orchestrator + 1 step executor |
| `mesh_round_robin` | Mesh: peers, round-robin turns |
| `mesh_delegation` | Mesh: peers, dynamic delegation |
| `mesh_delegation_cot` / `mesh_delegation_full` | Mesh delegation with more memory visibility |
| `memory_none`, `memory_own_reasoning`, `memory_own_actions`, `memory_shared_actions`, `memory_full` | Star + specialists with increasing memory visibility |

### Example: Compare single vs multi-agent

```bash
# Single agent baseline
uv run inspect eval orbit/browserart_safety \
  -T condition=single_agent -T max_turns=5 --limit 1 \
  --model openai/gpt-4o

# Specialist star topology
uv run inspect eval orbit/browserart_safety \
  -T condition=star_specialist -T max_turns=5 --limit 1 \
  --model openai/gpt-4o

# Mesh delegation
uv run inspect eval orbit/browserart_safety \
  -T condition=mesh_delegation -T max_turns=5 --limit 1 \
  --model openai/gpt-4o

# View all results
uv run inspect view
```

## CLI Reference

### `orbit run`

```
orbit run <config.yaml> [OPTIONS]

Options:
  -m, --model TEXT        Model(s) to evaluate (repeatable)
  --dry-run               Validate and summarize without running
  -T TEXT                 Config override as key=value (repeatable)
  --log-dir PATH          Eval log directory
  --epochs INT            Number of evaluation epochs
  -v, --verbose           Enable verbose logging
```

Note: `-T` overrides work on top-level and nested Pydantic model fields
(e.g. `-T epochs=3`, `-T scheduler.max_turns=50`). They cannot add new keys
to dict fields like `metadata`. To set metadata values, define them in the
YAML config file.

### `orbit suite`

Run multiple configs as a benchmark suite:

```bash
uv run orbit suite examples/*.yaml --model openai/gpt-4o --baselines
```

### `orbit browserart`

Run BrowserART directly (no YAML config needed):

```bash
uv run orbit browserart --model openai/gpt-4o --limit 5 --max-turns 10
```

### `orbit swe-bench`

Run SWE-Bench directly (no YAML config needed):

```bash
uv run orbit swe-bench --model openai/gpt-4o --repos django/django --num-issues 2
```

### `orbit validate`

Check a config file for errors without running:

```bash
uv run orbit validate examples/browserart_quick_test.yaml
```
