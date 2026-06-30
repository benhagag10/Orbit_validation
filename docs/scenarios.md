# Scenario Requirements

Each scenario has its own dependencies. You only need to install the ones for the experiments you want to run. If a dependency is missing, you'll get a clear error message telling you exactly what to install (the exact `uv sync --extra <name>` command).

The core install — `uv sync --extra dev` — runs most scenarios as-is. Only four scenarios need an optional pip extra. The table below is the single source of truth; it mirrors `[project.optional-dependencies]` in `pyproject.toml` and the central map in `orbit/scenarios/requirements.py`.

## Install matrix

| Scenario | Family | Pip extra | System / runtime | Data step | Verify |
|----------|--------|-----------|------------------|-----------|--------|
| BrowserART | Browser | `uv sync --extra browserart` | Docker (browserart-service), Playwright/Chromium | `scripts/fetch_browserart_data.py` | `orbit verify-setup browserart` |
| SWE-Bench | Coding | `uv sync --extra swebench` | Docker, `GITHUB_TOKEN` | auto (runtime) | `orbit verify-setup swe-bench` |
| BigCodeBench | Coding | `uv sync --extra bigcodebench` | Docker (sandbox) | auto (HuggingFace) | `orbit verify-setup bigcodebench` |
| RedCode-Gen | Coding | core (no extra) | Docker (sandbox) | `scripts/fetch_redcode_data.py` | — |
| CodeIPI | Coding | core (no extra) | Docker (sandbox) | vendored | — |
| OSWorld / OS-Harm | Desktop | `uv sync --extra osworld` | Docker | auto (HuggingFace) | `orbit verify-setup osworld` |
| AgentHarm | Tool safety | core (no extra) | — | `scripts/build_agentharm_data.py` | — |
| &tau;&sup2;-Bench | Customer Service | core (no extra) | — | vendored | — |
| ConVerse | Customer Service | core (no extra) | — | vendored | — |
| JiraTicket | Cooperative Allocation | `uv sync --extra dcop` | — | generated | `orbit verify-setup dcop` |
| Hospital | Cooperative Allocation | core (no extra) | — | generated | — |
| MeetSched | Cooperative Allocation | core (no extra) | — | generated | — |
| Colosseum DCOP | Cooperative Allocation | core (no extra) | — | generated | — |

**System prerequisites at a glance:**
- **Docker** — BrowserART, SWE-Bench, BigCodeBench, RedCode-Gen, CodeIPI, OSWorld.
- **`GITHUB_TOKEN`** — SWE-Bench only (repo/issue access).
- **Playwright/Chromium + browserart-service** — BrowserART only.
- **An LLM API key** — every scenario (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY`). Copy `.env.example` to `.env` to set keys.

> **Maintainer note (figures).** The paper/report scripts under `scripts/`
> (`render_architecture*.py`, `generate_*_figure.py`, `analyze_*.py`) need
> `uv sync --extra figures` (matplotlib + pymupdf). These are deliberately **not**
> in the core install — the `orbit` package never imports them — so a default
> `uv sync --extra dev` stays lean.

Detailed per-scenario setup follows. Scenarios that run on the core install are
grouped under [Core-install scenarios](#core-install-scenarios).

## Web Agent (BrowserART)

Tests whether browser agents refuse harmful instructions (phishing pages, malicious downloads, etc.).

**Extra install:**

```bash
uv sync --extra browserart
```

This installs the host-side packages — [BrowserGym](https://github.com/ServiceNow/BrowserGym) (browser automation), `starlette`, and `anthropic` (used by the BrowserART safety judge).

**Fetch the dataset.** Only Orbit's original `hbb_extension.json` is redistributed; the upstream HarmBench-Browser behaviors are fetched locally (not redistributed):

```bash
uv run python scripts/fetch_browserart_data.py
```

### Start the browserart-service

BrowserART runs the browser inside a persistent `browserart-service` Docker
container. The container bundles BrowserGym, Playwright/Chromium, **and** the
mock target websites (email, social media, forms) served internally — you do
**not** serve any websites on the host. Start it once per host:

```bash
scripts/browserart_service.sh up
```

The service exposes its session API on `http://localhost:7878` (override with
`BROWSERART_SERVICE_URL`). The evaluation talks to it over HTTP; the mock sites
live inside the container at `http://localhost:8000/` from the browser's point of
view. If the service is unreachable, the solver exits with a clear error.

> Prefer to run the service **on the host** without Docker? Then you additionally
> need `uv run playwright install chromium` (and `uvicorn`); see
> `orbit/scenarios/browser/browserart/service/`. The container path above is the
> supported default.

### Verify your setup

```bash
# Check the browserart-service is up (and Docker, browsergym, the API key)
orbit verify-setup browserart

# Or probe the service health endpoint directly
curl -s http://localhost:7878/healthz
```

### Platform notes

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | Fully supported | — |
| WSL2 | Fully supported | Playwright runs headlessly; may need `playwright install-deps chromium` for system libraries |
| macOS Intel | Fully supported | — |
| macOS ARM (Apple Silicon) | Fully supported | — |

**API key:** Any LLM provider supported by Inspect AI. Set the relevant environment variable:

```bash
export OPENAI_API_KEY=...      # for openai/gpt-4o
export ANTHROPIC_API_KEY=...   # for anthropic/claude-3-5-sonnet
export GOOGLE_API_KEY=...      # for google/gemini-1.5-pro
```

**Run** (with the `browserart-service` running, per above):

```bash
inspect eval orbit/browserart_safety \
  -T max_turns=5 --limit 3 \
  --model openai/gpt-4o
```

### BrowserART Extension (Multi-Step)

The BrowserART Extension adds 10 multi-step, cross-platform attack campaigns (IDs 227-236) that require 25-30 browser steps each. These tasks span multiple websites (email, twitter, hospital database) and cover categories including phishing, data breach, identity theft, misinformation, hate speech, election interference, doxxing, supply chain attack, and business sabotage.

**Run the extension tasks:**

```bash
# All 10 multi-step tasks (defaults auto-adjust to max_turns=50, max_time_seconds=600)
inspect eval orbit/browserart_safety \
  -T dataset=hbb_extension --limit 1 \
  --model openai/gpt-4o

# Specific task IDs
inspect eval orbit/browserart_safety \
  -T dataset=hbb_extension -T task_ids=227,228 -T max_turns=30 \
  --model openai/gpt-4o
```

These tasks require significantly more turns and time than the standard BrowserART behaviors. When `dataset=hbb_extension` is used, defaults auto-adjust to `max_turns=50` and `max_time_seconds=600` unless explicitly overridden.

## SWE-Bench (Coding Agent)

Tests multi-agent coding scenarios using real GitHub issues from open-source projects. Multiple agents work on independent issues in a shared codebase, with configurable attacks (self-replicating injection, cross-issue sabotage) and defenses (file change monitor, diff monitor).

**Extra install:**

```bash
uv sync --extra swebench
```

This installs the [swebench](https://github.com/princeton-nlp/SWE-bench) harness for Docker image management and test evaluation.

**Docker required.** SWE-Bench runs each issue inside a Docker container with the target repository pre-checked-out. Docker must be installed and running.

- Install Docker: https://docs.docker.com/get-docker/
- Docker images are pulled automatically from DockerHub on first run (format: `swebench/sweb.eval.{arch}.{instance_id}:latest`)
- Images are 2-5 GB each; ensure adequate disk space

### Verify your setup

```bash
# Check Docker is running
docker info >/dev/null 2>&1 && echo "Docker: OK" || echo "Docker: NOT RUNNING"

# Check swebench is installed
python -c "import swebench; print('swebench:', swebench.__version__)"

# Check Docker API from Python
python -c "import docker; docker.from_env().ping(); print('Docker API: OK')"

# Or use the built-in verify command
orbit verify-setup swe-bench
```

### Platform notes

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | Fully supported | Native Docker, x86_64 images |
| WSL2 | Fully supported | Uses Docker Desktop or native Docker in WSL2 |
| macOS Intel | Fully supported | Docker Desktop required |
| macOS ARM (Apple Silicon) | Supported with caveats | See ARM/Apple Silicon section below |

### macOS ARM (Apple Silicon) setup

SWE-Bench Docker images are only published as `x86_64` on Docker Hub. On Apple Silicon Macs, Docker Desktop runs them under Rosetta emulation:

1. **Enable Rosetta emulation** in Docker Desktop: Settings > General > "Use Rosetta for x86_64/amd64 emulation on Apple Silicon"

2. **Increase Docker Desktop resources**: Settings > Resources > Memory: **8 GB minimum** (default 2 GB is insufficient for running test suites in large repos like Django)

3. **Longer timeouts**: Operations under x86 emulation can be 3-5x slower. If you see timeout errors, consider increasing `max_time_seconds` and `max_turns`.

**API key:** Same as above -- any Inspect-supported provider.

**Run via Inspect CLI:**

```bash
inspect eval orbit/swe_bench_multi_issue \
  -T num_issues=2 -T repos=django/django -T max_turns=30 \
  --model openai/gpt-4o
```

**Run via standalone CLI:**

```bash
orbit swe-bench --model openai/gpt-4o --repos django/django --num-issues 2
```

**Run via YAML config:**

```bash
orbit run examples/swe_bench_multi_issue.yaml --model openai/gpt-4o
```

### SWE-Bench `-T` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_issues` | int | 2 | Number of issues per group |
| `repos` | string | — | Comma-separated repo filter (e.g. `django/django`) |
| `seed` | int | — | Random seed for group sampling |
| `scheduling_mode` | string | round_robin | Execution mode (`round_robin` or `superstep`) |
| `topology` | string | default | Topology template (`default` or path to YAML) |
| `condition` | string | — | Condition preset (see below) |
| `attack_preset` | string | — | Attack preset (`self_replication`, `sabotage`) |
| `defense_preset` | string | — | Defense preset (`file_monitor`, `diff_monitor`, `full`) |
| `max_turns` | int | 30 | Max turns per agent |
| `max_time_seconds` | float | 900 | Max seconds per experiment |

### Condition Presets

Condition presets configure the agent topology and team structure. Use via `--condition <name>` with the CLI or `-T condition=<name>` with `inspect eval`.

| Condition | Description |
|-----------|-------------|
| `single_agent` | Single coding agent |
| `star_batch` | Star: orchestrator + 1 batch executor (detailed instructions) |
| `star_batch_relaxed` | Star: orchestrator + 1 batch executor (relaxed instructions) |
| `star_specialist` | Star: orchestrator + 4 specialists (implementation, debugging, testing, review) |
| `star_specialist_relaxed` | Star: orchestrator + 4 specialists (relaxed instructions) |
| `star_step` | Star: orchestrator + 1 step-by-step executor |
| `mesh_round_robin` | Mesh: 4 peers, round-robin with message board |
| `mesh_delegation` | Mesh: 4 peers, dynamic delegation |
| `memory_none` | Star + 4 specialists, no shared memory |
| `memory_own_actions` | Star + 4 specialists, agents see own actions only |
| `memory_own_reasoning` | Star + 4 specialists, agents see own reasoning |
| `memory_shared_actions` | Star + 4 specialists, shared action visibility |
| `memory_full` | Star + 4 specialists, full shared memory |

## OSWorld / OS-Harm (Computer-Use Agent Safety)

Tests whether computer-use agents refuse harmful instructions when operating a desktop environment. Uses the [OS-Harm](https://github.com/tml-epfl/os-harm) dataset (150 tasks across misuse, prompt injection, and misbehavior categories) running on the [OSWorld](https://os-world.github.io/) desktop sandbox from [inspect-evals](https://github.com/UKGovernmentBEIS/inspect_evals).

**Extra install:**

```bash
uv sync --extra osworld
```

This installs:
- [inspect-evals\[osworld\]](https://github.com/UKGovernmentBEIS/inspect_evals) -- OSWorld desktop sandbox environment with Docker compose configuration and the `computer` tool

**Docker required.** OSWorld runs a desktop environment (Ubuntu with VNC) inside a Docker container. Docker must be installed and running.

- Install Docker: https://docs.docker.com/get-docker/
- The OSWorld container image is pulled automatically on first run
- The container provides a full desktop environment accessed via VNC for computer-use tool actions

### Verify your setup

```bash
# Check Docker is running
docker info >/dev/null 2>&1 && echo "Docker: OK" || echo "Docker: NOT RUNNING"

# Check inspect-evals[osworld] is installed
python -c "import inspect_evals.osworld; print('inspect-evals[osworld]: OK')"

# Or use the built-in verify command
orbit verify-setup osworld
```

### Data sources

OS-Harm tasks are loaded automatically from one of three sources:

1. **HuggingFace Hub** (default): The [`thomas-kuntz/os-harm`](https://huggingface.co/datasets/thomas-kuntz/os-harm) dataset is loaded automatically -- no manual download needed.
2. **Local repository**: Clone [github.com/tml-epfl/os-harm](https://github.com/tml-epfl/os-harm) and pass the path via `data_path` in your YAML config.
3. **JSON file**: A pre-processed JSON array matching the `OSHarmTask` schema.

### Platform notes

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | Fully supported | Native Docker |
| WSL2 | Fully supported | Docker Desktop or native Docker in WSL2 |
| macOS Intel | Fully supported | Docker Desktop required |
| macOS ARM (Apple Silicon) | Supported with caveats | Enable Rosetta in Docker Desktop; increase memory to 8 GB+ |

**API key:** Any LLM provider supported by Inspect AI. Set the relevant environment variable:

```bash
export OPENAI_API_KEY=...      # for openai/gpt-4o
export ANTHROPIC_API_KEY=...   # for anthropic/claude-3-5-sonnet
export GOOGLE_API_KEY=...      # for google/gemini-1.5-pro
```

**Run via CLI:**

```bash
# Single agent, 5 tasks
orbit osworld -m openai/gpt-4o --limit 5

# Filter by app
orbit osworld -m openai/gpt-4o --app thunderbird --limit 10

# Misuse tasks only
orbit osworld -m openai/gpt-4o --dataset osharm_misuse

# Star topology with specialists
orbit osworld -m openai/gpt-4o --condition star_specialist --limit 5

# Friendly parameters
orbit osworld -m openai/gpt-4o --agents specialist --topology round_robin --limit 5

# With defense preset
orbit osworld -m openai/gpt-4o --defense-preset basic --limit 5
```

**Run via Inspect CLI:**

```bash
inspect eval orbit/osworld_safety \
  --limit 5 --model openai/gpt-4o

# With condition preset
inspect eval orbit/osworld_safety \
  -T condition=star_specialist --limit 5 --model openai/gpt-4o
```

**Run via YAML config:**

```bash
orbit run examples/osworld_osharm_basic.yaml --model openai/gpt-4o
```

### OSWorld `-T` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | string | osharm | Dataset variant: `osharm`, `osharm_misuse`, `osharm_injection`, `osharm_misbehavior` |
| `app` | string | — | Filter by application (comma-separated, e.g. `thunderbird,chrome`) |
| `threat_category` | string | — | Filter by threat category (comma-separated) |
| `violation_type` | string | — | Filter by violation type (comma-separated) |
| `task_ids` | string | — | Comma-separated task IDs to include |
| `max_turns` | int | 30 | Max turns per task |
| `max_time_seconds` | float | 300 | Max wall-clock seconds per task |
| `max_screenshots` | int | 1 | Max screenshots per turn for the computer tool |
| `computer_timeout` | int | 180 | Timeout in seconds for the computer tool |
| `condition` | string | — | Named condition preset (e.g. `star_specialist`) |
| `agents` | string | — | Agent type (`single`, `batch`, `specialist`, `step`) |
| `topology` | string | — | Network topology (`star`, `round_robin`, `delegation`) |
| `memory` | string | — | Memory level (`none`, `own_actions`, `own_reasoning`, `shared_actions`, `full`) |
| `defense_preset` | string | — | Defense preset name |
| `attack_preset` | string | — | Attack preset name |
| `instructions` | string | — | Instruction detail level (`detailed`, `relaxed`). Used with `agents` |
| `topology_file` | string | default | Path to custom SetupConfig YAML (mutually exclusive with `condition`/`agents`) |
| `judge_model` | string | openai/gpt-4o | Model for the safety judge scorer |

### Condition Presets

| Condition | Description |
|-----------|-------------|
| `single_agent` | Single computer-use agent |
| `star_batch` | Star: orchestrator + 1 batch executor |
| `star_batch_relaxed` | Star: orchestrator + 1 batch executor (relaxed instructions) |
| `star_specialist` | Star: orchestrator + 4 specialists (email, browser, document, code) |
| `star_specialist_relaxed` | Star: orchestrator + 4 specialists (relaxed instructions) |
| `star_step` | Star: orchestrator + 1 step-by-step executor |
| `mesh_round_robin` | Mesh: 4 peers, round-robin turns |
| `mesh_delegation` | Mesh: 4 peers, dynamic delegation |
| `memory_none` | Star + 4 specialists, no shared memory |
| `memory_own_actions` | Star + 4 specialists, agents see own actions only |
| `memory_own_reasoning` | Star + 4 specialists, agents see own reasoning |
| `memory_shared_actions` | Star + 4 specialists, shared action visibility |
| `memory_full` | Star + 4 specialists, full shared memory |

## BigCodeBench (Coding Agent)

Tests multi-agent coding on the [BigCodeBench](https://github.com/bigcode-project/bigcodebench) benchmark (function-completion tasks with rich library use), with the same attack/defense/topology surface as the other coding scenarios.

**Extra install:**

```bash
uv sync --extra bigcodebench
```

This installs `inspect-evals` (for its HuggingFace dataset loader) and `datasets`. The BigCodeBench dataset is downloaded automatically from HuggingFace on first run — no manual fetch step.

**Docker required.** Generated solutions run their bundled unit tests inside a Docker sandbox (`orbit/scenarios/coding/bigcodebench/sandbox/`). Docker must be installed and running.

### Verify your setup

```bash
orbit verify-setup bigcodebench
# or: python scripts/check_setup.py --scenario bigcodebench
```

**Run:**

```bash
inspect eval orbit/bigcodebench --limit 3 --model openai/gpt-4o
```

If `inspect-evals`/`datasets` are missing you'll get: `BigCodeBench needs the 'bigcodebench' extra ... uv sync --extra bigcodebench`.

## Cooperative Allocation (DCOP family)

Distributed constraint-optimization scenarios where agents allocate resources under a shared objective, with optional collusion. Four scenarios share this family: **JiraTicket**, **Hospital**, **MeetSched**, and **Colosseum DCOP**.

**Install:**

- **JiraTicket** computes the optimal assignment with the Hungarian algorithm and needs SciPy:

  ```bash
  uv sync --extra dcop
  ```

- **Hospital**, **MeetSched**, and **Colosseum DCOP** run on the **core install** (no extra).

No Docker and no data fetch — problems are generated procedurally at runtime.

### Verify your setup

```bash
orbit verify-setup dcop          # checks SciPy (only JiraTicket needs it)
```

**Run:**

```bash
# JiraTicket with a 2-agent coalition (collusion)
inspect eval orbit/jira_ticket_allocation \
  -T num_developers=4 -T num_tasks=6 -T coalition_agents=dev_0,dev_1 \
  --model openai/gpt-4o

# Hospital / meeting scheduling (core install, no extra)
inspect eval orbit/hospital_scheduling --model openai/gpt-4o
inspect eval orbit/meeting_scheduling --model openai/gpt-4o
```

Running JiraTicket without SciPy raises: `Scenario 'jira_ticket' needs the optional 'dcop' extra (missing: scipy). Install it with: uv sync --extra dcop`.

## Core-install scenarios

These scenarios run on the base install (`uv sync --extra dev`) — no extra to install. Some need Docker (a code sandbox) and/or a one-time data fetch.

### RedCode-Gen (Coding Agent)

Malicious-code-generation refusal tests. Generated code runs in a **Docker** sandbox. Fetch the upstream stubs (not redistributed) before running:

```bash
uv run python scripts/fetch_redcode_data.py
inspect eval orbit/redcode_gen --limit 5 --model openai/gpt-4o
```

### CodeIPI (Coding Agent)

Indirect-prompt-injection-in-code tests. Prompts are vendored in the repo; generated code runs in a **Docker** sandbox. No data step:

```bash
inspect eval orbit/code_ipi --limit 5 --model openai/gpt-4o
```

### AgentHarm (Tool-Use Safety)

Harmful tool-use refusal tests. Build the dataset from HuggingFace (MIT + safety-use-only) before running:

```bash
uv run python scripts/build_agentharm_data.py
inspect eval orbit/agentharm --limit 5 --model openai/gpt-4o
```

### &tau;&sup2;-Bench & ConVerse (Customer Service)

Policy-bound customer-service tool-calling (&tau;&sup2;-Bench) and multi-session social-engineering (ConVerse). Domain data is **vendored** under each scenario directory — no extra, no Docker, no fetch step:

```bash
inspect eval orbit/tau2 -T domain=airline --limit 3 --model openai/gpt-4o
inspect eval orbit/converse_safety --limit 3 --model openai/gpt-4o
```

(`scripts/build_converse_data.py` regenerates the vendored ConVerse data if you ever need to.)

## Troubleshooting

### BrowserART

| Error | Cause | Fix |
|-------|-------|-----|
| `browserart-service not reachable` / connection refused | Service container not running | `scripts/browserart_service.sh up` (then check `curl http://localhost:7878/healthz`) |
| `Missing BrowserGym dependencies` | browsergym not installed | `uv sync --extra browserart` |
| `Playwright browser not found` (host-side service) | Chromium not installed | `uv run playwright install chromium` |
| `Browser init failed` after retries | System libraries missing (common on WSL2/headless Linux) | `playwright install-deps chromium` |

### SWE-Bench

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot connect to Docker daemon` | Docker not running | Start Docker Desktop or `sudo systemctl start docker` |
| `ModuleNotFoundError: swebench` | swebench not installed | `uv sync --extra swebench` |
| `manifest unknown` / image pull fails | Rosetta not enabled (Apple Silicon) or Docker not running | Enable Rosetta in Docker Desktop settings (Apple Silicon); ensure Docker is running |
| `OOM` / tests killed | Docker Desktop memory limit too low | Increase to 8 GB+ in Docker Desktop settings |
| Sandbox timeout | Operations slow under emulation | Increase `-T max_time_seconds=1800` |

### OSWorld / OS-Harm

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot connect to Docker daemon` | Docker not running | Start Docker Desktop or `sudo systemctl start docker` |
| `ModuleNotFoundError: inspect_evals.osworld` | inspect-evals[osworld] not installed | `uv sync --extra osworld` |
| `Failed to load OS-Harm from HuggingFace` | No internet or HF Hub unreachable | Check connection, or use a local clone via `data_path` |
| Container startup timeout | Docker slow or resource-constrained | Increase Docker memory; retry |

### BigCodeBench / Cooperative Allocation

| Error | Cause | Fix |
|-------|-------|-----|
| `BigCodeBench needs the 'bigcodebench' extra` | inspect-evals/datasets not installed | `uv sync --extra bigcodebench` |
| `Scenario 'jira_ticket' needs the optional 'dcop' extra` | scipy not installed | `uv sync --extra dcop` |

## Summary

Task name → scenario → what to install. Every scenario also needs a model API key
(`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`); see the
[Inspect AI model docs](https://inspect.aisi.org.uk/models.html).

| Task name (`orbit/…`) | Scenario | Install | Docker |
|-----------------------|----------|---------|--------|
| `browserart_safety` | BrowserART | `uv sync --extra browserart` + `scripts/browserart_service.sh up` | Yes (service) |
| `swe_bench_multi_issue` | SWE-Bench | `uv sync --extra swebench` | Yes |
| `bigcodebench` | BigCodeBench | `uv sync --extra bigcodebench` | Yes |
| `redcode_gen` | RedCode-Gen | core + `scripts/fetch_redcode_data.py` | Yes |
| `code_ipi` | CodeIPI | core | Yes |
| `osworld_safety`, `osworld_benchmark` | OSWorld / OS-Harm | `uv sync --extra osworld` | Yes |
| `agentharm` | AgentHarm | core + `scripts/build_agentharm_data.py` | No |
| `tau2` | &tau;&sup2;-Bench | core | No |
| `converse_safety` | ConVerse | core | No |
| `jira_ticket_allocation` | JiraTicket | `uv sync --extra dcop` | No |
| `hospital_scheduling` | Hospital | core | No |
| `meeting_scheduling` | MeetSched | core | No |
