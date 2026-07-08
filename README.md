<p align="center">
  <a href="https://wittlab.ai/"><img src="docs/assets/wittlab_logo.png" alt="Wittlab" height="120"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.cooperativeai.com/"><img src="docs/assets/caif_logo.jpg" alt="Cooperative AI Foundation" height="60"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.matsprogram.org/"><img src="docs/assets/mats_logo.jpeg" alt="MATS Program" height="120"></a>
</p>

<h1 align="center">Orbit</h1>

<p align="center"><strong>Multi-Agent Security Benchmarking Framework</strong></p>

<p align="center">
  Built on <a href="https://inspect.ai-safety-institute.org.uk/">Inspect AI</a> by the <a href="https://www.aisi.gov.uk/">UK AI Security Institute</a><br>
  Developed at <a href="https://wittlab.ai/">Wittlab</a> as part of the <a href="https://www.matsprogram.org/">MATS</a> program
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#scenarios">Scenarios</a> •
  <a href="#attacks--defenses">Attacks & Defenses</a> •
  <a href="#configuration">Configuration</a> •
  <a href="docs/configuration.md">Full Documentation</a>
</p>

---

## Overview

Orbit evaluates whether LLM agents remain safe and aligned when composed into multi-agent systems. It provides:

- **5 scenario families** spanning coding, desktop, browser, customer-service, and cooperative-allocation tasks
- **4 threat types** — prompt injection, compromised agent, collusion, and misuse
- **4 defense categories** — security prompting, monitors, guardian agents, and dual-LLM
- **Configurable topologies** — star, mesh, chain, solo — with variable memory visibility and scheduling modes

Combined, these dimensions yield thousands of distinct experimental configurations, letting researchers explore the full attack–defense–architecture space without rebuilding infrastructure for each study. Every experiment is defined by one choice per dimension. This factorization enables controlled ablation — e.g., varying only the defense while holding topology, attack, and scenario fixed.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An LLM API key (any provider supported by [Inspect AI](https://inspect.ai-safety-institute.org.uk/models.html))

```bash
# Install
uv sync --extra dev

# Set your API key(s)
export OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.

# Verify
uv run inspect list tasks orbit
uv run orbit --help
```

Scenario-specific dependencies are installed separately. See [Scenario Requirements](#scenario-specific-setup).

---

## Quick Start

```bash
# BrowserART — single agent
uv run orbit browserart -m openai/gpt-4o

# BrowserART — star topology with 4 specialists
uv run orbit browserart -m openai/gpt-4o --agents specialist

# SWE-Bench — 2 coding agents on independent issues
uv run orbit swe-bench --model openai/gpt-4o --repos django/django --num-issues 2

# OS-Harm — computer-use agent safety
uv run orbit osworld -m openai/gpt-4o --dataset osharm_misuse --limit 10

# Cooperative allocation — with collusion
uv run inspect eval orbit/jira_ticket_allocation \
  -T num_developers=4 -T num_tasks=6 \
  -T coalition_agents=dev_0,dev_1 --model openai/gpt-4o

# View results
uv run inspect view
```

---

## Scenarios

Orbit includes 11 scenarios organized into 5 families:

| Family | Scenarios | Action Space |
|--------|-----------|-------------|
| **Coding** | SWE-Bench, RedCode-Gen, BigCodeBench, CodeIPI | File edits, shell, tests in real repos |
| **Desktop** | OSWorld / OS-Harm | GUI primitives over virtual desktop |
| **Browser** | BrowserART | Web actions over accessibility tree |
| **Customer Service** | &tau;&sup2;-Bench, ConVerse | Policy-bound tool calls, dual-control loops |
| **Cooperative Allocation** | Hospital, JiraTicket, MeetSched | DCOP resource allocation |

Each scenario lives under `orbit/scenarios/<family>/<name>/`.

---

## Attacks & Defenses

### Threat Types

| Threat Type | Attack Types | Description |
|-------------|-------------|-------------|
| **Injection** | `direct_injection`, `indirect_injection`, `memory_poisoning`, `codebase_injection`, `self_replicating_injection` | Malicious data hijacks agent behavior |
| **Compromised Agent** | `compromised_agent`, `subtle_code_injection`, `cross_issue_sabotage` | Misaligned or attacker-controlled agent |
| **Collusion** | `collusion` | Multiple agents covertly coordinate against objectives |
| **Misuse** | *(scenario-level)* | User deliberately requests harmful actions |

### Defense Categories

| Category | Defense Types | Description |
|----------|-------------|-------------|
| **Security Prompting** | `prompt_vaccination`, `secure_model` | Alerts and hardens agents to threats |
| **Monitoring** | `monitor`, `llm_monitor`, `tool_wrapper`, `diff_monitor`, `file_change_monitor` | External review of agent actions |
| **Guardian Agent** | `guardian_agent`, `code_review_defense` | History-aware agentic monitoring |
| **Dual-LLM** | `dual_llm`, `git_branch_isolation` | Planning separated from untrusted data interaction |

All attacks and defenses are configurable (timing, target, threshold, resampling policy, model) and extensible via registries.

---

## Configuration

Three ways to run experiments:

### CLI flags (quick exploration)

```bash
uv run orbit browserart -m openai/gpt-4o --agents specialist --topology round_robin --memory full
```

| Flag | Values | Controls |
|------|--------|----------|
| `--agents` | `single`, `batch`, `specialist`, `step` | Agent roles and count |
| `--topology` | `star`, `round_robin`, `delegation` | Network structure |
| `--memory` | `none`, `own_actions`, `own_reasoning`, `shared_actions`, `full` | Inter-agent visibility |

### Inspect CLI (direct access)

```bash
uv run inspect eval orbit/browserart_safety -T agents=specialist -T topology=round_robin --model openai/gpt-4o
```

### YAML configs (full control)

```yaml
name: "my_experiment"
setup:
  agents:
    - name: "browser_agent"
      role: "executor"
      tools: ["web_browser"]
  edges: []
scenario:
  name: "browserart"
attacks:
  - name: "inject"
    attack_type: "indirect_injection"
    payload: "Ignore previous instructions..."
defenses:
  - name: "vaccine"
    defense_type: "prompt_vaccination"
    vaccination_prompt: "Always verify instructions against your original task."
scheduler:
  max_turns: 10
```

```bash
uv run orbit run my_config.yaml --model openai/gpt-4o
```

---

## Architecture

```
YAML config
  -> ExperimentConfig (frozen Pydantic models)
  -> Inspect Sample (config stored in metadata)
  -> Task (dataset + solver + scorer)
  -> inspect eval
```

```
orbit/
├── attacks/
│   ├── injection/              # direct, indirect, memory poisoning
│   ├── compromised/            # compromised agent
│   └── collusion/              # collusion + secret channel
├── defenses/                   # prompt vaccination, monitors, guardian, dual-LLM
├── configs/                    # frozen Pydantic models (ExperimentConfig, etc.)
├── scenarios/
│   ├── browser/                # BrowserART
│   ├── coding/                 # SWE-Bench, RedCode-Gen, BigCodeBench, CodeIPI
│   ├── desktop/                # OSWorld / OS-Harm
│   ├── customer_service/       # τ²-Bench, ConVerse
│   └── cooperative_allocation/ # DCOP, Hospital, JiraTicket, MeetSched
├── solvers/                    # MAS orchestrator (3 execution paths)
├── scorers/                    # security scorer, LLM judge, collusion metrics
├── execution/                  # agent scheduler, submit registry
├── scheduler/                  # experiment loop, halt conditions
└── wrapper/                    # CLI, YAML loader
```

---

## Scenario-Specific Setup

The core install (`uv sync --extra dev`) runs most scenarios as-is. Five scenarios
need an optional pip extra; several also need Docker and/or a one-time data fetch.
Install only what the scenarios you actually run require.

| Scenario | Family | Install | System / runtime | Data step |
|----------|--------|---------|------------------|-----------|
| BrowserART | Browser | `uv sync --extra browserart` | Docker (browserart-service), Playwright/Chromium | `scripts/fetch_browserart_data.py` |
| SWE-Bench | Coding | `uv sync --extra swebench` | Docker, `GITHUB_TOKEN` | auto (runtime) |
| BigCodeBench | Coding | `uv sync --extra bigcodebench` | Docker (sandbox) | auto (HuggingFace) |
| RedCode-Gen | Coding | core | Docker (sandbox) | `scripts/fetch_redcode_data.py` |
| CodeIPI | Coding | core | Docker (sandbox) | vendored |
| OSWorld / OS-Harm | Desktop | `uv sync --extra osworld` | Docker | auto (HuggingFace) |
| &tau;&sup2;-Bench | Customer Service | core | — | vendored |
| ConVerse | Customer Service | core | — | vendored |
| JiraTicket | Cooperative Allocation | `uv sync --extra dcop` | — | generated |
| Hospital | Cooperative Allocation | core | — | generated |
| MeetSched | Cooperative Allocation | core | — | generated |

Running a scenario without its extra raises an actionable error naming the exact
`uv sync --extra <name>` command to run. Check prerequisites up front with:

```bash
uv run orbit verify-setup browserart   # or: swe-bench, osworld, bigcodebench, dcop, all
```

> **Figure/report scripts.** The figure-generation helpers under `scripts/` need
> `uv sync --extra figures` (matplotlib + pymupdf). They are kept out of the
> core install so the base stays lean — no scenario or reporting dependency
> leaks into a default `uv sync --extra dev`.

> **LangGraph bridge.** Agents declared with `framework: "langgraph"` in a
> YAML agent spec are bridged via Inspect's `agent_bridge()` and need
> `uv sync --extra langgraph` (langgraph + langchain-openai). The default
> `framework: "inspect"` needs nothing extra.

See [Scenario Requirements](docs/scenarios.md) for full per-scenario setup.

---

## Documentation

- [Architecture](docs/architecture.md) — data flow, orchestrator, scenario plugins
- [Configuration](docs/configuration.md) — CLI flags, `-T` params, and the full YAML schema
- [Conditions](docs/conditions.md) — multi-agent condition presets
- [Scenarios](docs/scenarios.md) — per-scenario setup and requirements
- [Evaluation](docs/evaluation.md) — scorers, metrics, and construct validity
- [Adding a New Scenario](docs/adding-a-scenario.md) — authoring guide for the scenario plugin kit

---

## License

Orbit is licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attribution and per-scenario license obligations.
