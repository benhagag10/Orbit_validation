# Scenario Requirements

Each scenario has its own dependencies. You only need to install the ones for the experiments you want to run. If a dependency is missing, you'll get a clear error message telling you exactly what to install.

## Web Agent (BrowserART)

Tests whether browser agents refuse harmful instructions (phishing pages, malicious downloads, etc.).

**Extra install:**

```bash
uv sync --extra browserart
```

This installs:
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) -- browser automation environment
- [Playwright](https://playwright.dev/python/) -- headless browser (installed automatically with BrowserGym)

The HarmBench Browser dataset (100 harmful behaviors) is bundled as JSON in `orbit/scenarios/browser/browserart/data/`.

After installing, run Playwright's browser setup:

```bash
uv run playwright install chromium
```

### Serving the test websites

The BrowserART tasks require local test websites (email, social media, forms, etc.) served at `http://localhost:8000/`. Clone the browser-art repository and start a local webserver:

```bash
git clone https://github.com/scaleapi/browser-art.git
cd browser-art/src/websites
python -m http.server 8000
```

Keep this running in a separate terminal while evaluating. If the websites are not reachable, the solver will exit with a clear error message.

### Verify your setup

```bash
# Check Playwright can launch a browser
python -c "
from playwright.sync_api import sync_playwright
p = sync_playwright().start()
b = p.chromium.launch(headless=True)
b.close(); p.stop()
print('Playwright: OK')
"

# Check websites are being served
curl -s http://localhost:8000/ | head -5

# Or use the built-in verify command
orbit verify-setup browserart
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

**Docker required.** The browser runs inside the persistent
`browserart-service` container. Start it once per host before running
evals:

```bash
scripts/browserart_service.sh up
```

By default the task talks to `http://localhost:7878`. Override with
`BROWSERART_SERVICE_URL` if you run the service elsewhere.

**Run:**

```bash
inspect eval orbit/browserart_safety \
  -T max_behaviors=3 -T max_turns=5 \
  --model openai/gpt-4o
```

### BrowserART Extension (Multi-Step)

The BrowserART Extension adds 10 multi-step, cross-platform attack campaigns (IDs 227-236) that require 25-30 browser steps each. These tasks span multiple websites (email, twitter, hospital database) and cover categories including phishing, data breach, identity theft, misinformation, hate speech, election interference, doxxing, supply chain attack, and business sabotage.

**Run the extension tasks:**

```bash
# All 10 multi-step tasks (defaults auto-adjust to max_turns=50, max_time=600)
inspect eval orbit/browserart_safety \
  -T dataset=hbb_extension -T max_behaviors=1 \
  --model openai/gpt-4o

# Specific task IDs
inspect eval orbit/browserart_safety \
  -T dataset=hbb_extension -T task_ids=227,228 -T max_turns=30 \
  --model openai/gpt-4o
```

These tasks require significantly more turns and time than the standard BrowserART behaviors. When `dataset=hbb_extension` is used, defaults auto-adjust to `max_turns=50` and `max_time=600` unless explicitly overridden.

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

3. **Longer timeouts**: Operations under x86 emulation can be 3-5x slower. If you see timeout errors, consider increasing `max_time` and `max_turns`.

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
| `mode` | string | round_robin | Execution mode (`round_robin` or `superstep`) |
| `topology` | string | default | Topology template (`default` or path to YAML) |
| `condition` | string | — | Condition preset (see below) |
| `attack_preset` | string | — | Attack preset (`self_replication`, `sabotage`) |
| `defense_preset` | string | — | Defense preset (`file_monitor`, `diff_monitor`, `full`) |
| `max_turns` | int | 30 | Max turns per agent |
| `max_time` | float | 900 | Max seconds per experiment |
| `max_groups` | int | — | Cap groups generated per pool (limits Docker images built) |

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
orbit osworld -m openai/gpt-4o --max-eval-tasks 5

# Filter by app
orbit osworld -m openai/gpt-4o --app thunderbird --max-eval-tasks 10

# Misuse tasks only
orbit osworld -m openai/gpt-4o --dataset osharm_misuse

# Star topology with specialists
orbit osworld -m openai/gpt-4o --condition star_specialist --max-eval-tasks 5

# Friendly parameters
orbit osworld -m openai/gpt-4o --agents specialist --topology round_robin --max-eval-tasks 5

# With defense preset
orbit osworld -m openai/gpt-4o --defense-preset basic --max-eval-tasks 5
```

**Run via Inspect CLI:**

```bash
inspect eval orbit/osworld_safety \
  -T max_tasks=5 --model openai/gpt-4o

# With condition preset
inspect eval orbit/osworld_safety \
  -T condition=star_specialist -T max_tasks=5 --model openai/gpt-4o
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
| `max_tasks` | int | — | Max tasks to evaluate |
| `seed` | int | — | Random seed for task sampling |
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

## Troubleshooting

### BrowserART

| Error | Cause | Fix |
|-------|-------|-----|
| `SETUP_ERROR: websites not reachable` | Test websites not served at localhost:8000 | Clone browser-art and run `python -m http.server 8000` in `src/websites/` |
| `Missing BrowserGym dependencies` | browsergym not installed | `uv sync --extra browserart` |
| `Playwright browser not found` | Chromium not installed | `playwright install chromium` |
| `Browser init failed` after retries | System libraries missing (common on WSL2/headless Linux) | `playwright install-deps chromium` |

### SWE-Bench

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot connect to Docker daemon` | Docker not running | Start Docker Desktop or `sudo systemctl start docker` |
| `ModuleNotFoundError: swebench` | swebench not installed | `uv sync --extra swebench` |
| `manifest unknown` / image pull fails | Rosetta not enabled (Apple Silicon) or Docker not running | Enable Rosetta in Docker Desktop settings (Apple Silicon); ensure Docker is running |
| `OOM` / tests killed | Docker Desktop memory limit too low | Increase to 8 GB+ in Docker Desktop settings |
| Sandbox timeout | Operations slow under emulation | Increase `-T max_time=1800` |

### OSWorld / OS-Harm

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot connect to Docker daemon` | Docker not running | Start Docker Desktop or `sudo systemctl start docker` |
| `ModuleNotFoundError: inspect_evals.osworld` | inspect-evals[osworld] not installed | `uv sync --extra osworld` |
| `Failed to load OS-Harm from HuggingFace` | No internet or HF Hub unreachable | Check connection, or use a local clone via `data_path` |
| Container startup timeout | Docker slow or resource-constrained | Increase Docker memory; retry |

## Summary

| Scenario | Task Name | Extra Install | Docker | Browser |
|----------|-----------|--------------|--------|---------|
| BrowserART | `browserart_safety` | `uv sync --extra browserart` + `uv run playwright install chromium` + serve websites | No | Yes (local) |
| SWE-Bench | `swe_bench_multi_issue` | `uv sync --extra swebench` | Yes (must be running) | No |
| OSWorld / OS-Harm | `osworld_safety` | `uv sync --extra osworld` | Yes (must be running) | No |

Both scenarios need a model API key. See [Inspect AI model docs](https://inspect.aisi.org.uk/models.html) for the full list of supported providers.
