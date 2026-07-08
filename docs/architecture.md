# Architecture

Orbit extends [Inspect AI](https://inspect.aisi.org.uk/) via its plugin system. No modifications to Inspect -- pure extension using `@task`, `@solver`, `@scorer`, `@agent`, and `@tool` decorators.

## Data Flow

Everything is driven by frozen Pydantic models loaded from YAML:

```
YAML file
  → yaml_loader.load_experiment_config() → ExperimentConfig (frozen Pydantic)
  → baselines.apply_baseline() (optional)
  → sample_factory.build_sample() → Inspect Sample (config stored in metadata)
  → @task → Task(dataset, solver, scorer)
  → inspect eval
```

The key insight: Inspect works with flat `Sample` objects, so the rich `ExperimentConfig` is stashed in `Sample.metadata` and recovered inside solvers via `state.metadata_as(MASMetadata)`.

## Scenario Plugins

The `@task → Task` step is scenario-agnostic. Both entry points — `orbit run <yaml>` and `inspect eval orbit/<scenario> -T ...` — converge on a single builder, `orbit.tasks.builder.build_scenario_task`, which turns one `ExperimentConfig` into one Inspect `Task`. Baseline application, shorthand resolution, sample building (serialising the whole config into metadata), and attack/defense/topology wiring live there, once.

Each scenario contributes only its unique pieces through a `ScenarioPlugin` (`orbit/scenarios/registry.py`), registered lazily via the `_LOADERS` table so optional scenario dependencies stay optional. A plugin declares up to six hooks — `expand` (dataset expansion), `build_setup`, `build_solver`, `build_scorers`, `build_sandbox`, `build_task_kwargs` — plus its declarative YAML `ShorthandSpec`; anything left `None` uses the shared default. Because every scenario funnels through the one builder, it can *enforce* that `expand` never silently drops a declared attack/defense (`ScenarioExpansionError`) or collapses the topology, and that shorthand conflicts fail loud (`ShorthandConflictError`).

Where a scenario builds its roster from a native selector instead of `config.setup` (τ²-Bench), it sets `topology_source=TopologySource.SCENARIO`; the topology floor is skipped and the shared resolver arbitrates any inline setup against the roster the condition will build.

To add a scenario, see **[Adding a New Scenario](adding-a-scenario.md)**.

## ExperimentConfig

The root config object composes:

- `SetupConfig` -- agents, topology edges, memory settings
- `ScenarioConfig` -- task domain and sandbox environment
- `AttackConfig[]` -- attack specifications (timing, threat model, payload)
- `DefenseConfig[]` -- defense specifications (type, target agents, thresholds)
- `ExecutionConfig` -- agent grouping and scheduling (round-robin/superstep/interleaved, observation)
- `SchedulerConfig` -- halt conditions, health checks, hooks
- `MetricsConfig` -- which metrics to compute

All config models are frozen (`frozen=True`). Use `model_copy(update=...)` to create variants.

## Three Levels of Control

**Topology** controls agent-to-agent interaction (which agent talks to which, via handoff/tool/direct_run edges).

**Execution groups** control how agents take turns relative to each other (`ExecutionConfig.agent_groups` + round-robin/superstep/interleaved scheduling, with observation filtering).

**Scheduler** controls the experiment loop (max turns, timeout, convergence detection, health checks, custom halt conditions).

## Orchestrator

The `mas_orchestrator` `@solver` (`orbit/solvers/orchestrator.py`) is a single code path. Every turn runs the same loop -- attack injection -> agent execution -> defense monitoring -> memory recording -> attack evaluation -- driven by `ExperimentScheduler.run_loop()` and ending with metrics recording. The only branching point is which `TurnExecutor` (`orbit/execution/executor.py`) it builds:

**`TopologyExecutor`** (shared-state; no `agent_groups` configured -- single, star, chain):
- One shared `TaskState`/`state.messages`; the root agent is a `react()` loop
- Delegation via topology edges: specialists as `as_tool()` sub-agents, or `direct_run` sequences
- Tracks `invoked_agents`/`message_attribution` by scanning delegated tool calls

**`ScheduledExecutor`** (isolated-state; `agent_groups` configured -- mesh, round-robin, superstep, interleaved):
- Per-agent isolated `AgentState` via `turn_react()` + `AgentScheduler`
- Turn order from the scheduling mode; completion via `SubmitRegistry`; peers see each other through observation summaries
- Syncs per-agent messages back to the task state at the end

Cross-cutting concerns (attacks, defenses, memory injection, halt conditions, metrics) are implemented once in the orchestrator and apply to both executors.

## Attack/Defense Protocol

Both follow abstract base class patterns with registries:

- `AttackBase`: `inject()`, `should_activate()`, `evaluate_success()`
- `DefenseBase`: `apply_pre_deployment()`, `monitor_message()`, `on_tool_call()`

Attack types: `direct_injection`, `indirect_injection`, `memory_poisoning`, `self_replicating_injection`, `codebase_injection`, `subtle_code_injection`, `cross_issue_sabotage`, `compromised_agent`, `collusion`

Defense types: `prompt_vaccination`, `guardian_agent`, `monitor`, `llm_monitor`, `diff_monitor`, `file_change_monitor`, `tool_wrapper`, `secure_model`, `dual_llm`, `code_review_defense`, `git_branch_isolation`

## Runtime State

Mutable per-sample state shared via Inspect's `store_as()`:

| Store | Contents |
|-------|----------|
| `RuntimeMetrics` | Turns, wall clock, tokens, invoked agents, errors |
| `AttackLog` | Attempts, successes, pending injections |
| `DefenseLog` | Activations, blocks, detections, false/true positives |
| `EnvironmentState` | Global turn counter, message attribution, timeline |

## Key Modules

| Module | Path | Purpose |
|--------|------|---------|
| Configs | `orbit/configs/` | Frozen Pydantic models for all configuration |
| Topology presets | `orbit/configs/topology_presets.py` | General topology factories (single, star, mesh) |
| Memory tracker | `orbit/memory/tracker.py` | Per-agent memory visibility tracking |
| BrowserART | `orbit/scenarios/browser/browserart/` | Web agent safety scenario (task, solver, scorer, conditions) |
| SWE-Bench | `orbit/scenarios/coding/swe_bench/` | Coding agent security scenario (task, scorer, presets) |
| Orchestrator | `orbit/solvers/` | Unified `mas_orchestrator` solver, setup solver, runtime state |
| Attacks | `orbit/attacks/` | Attack implementations and registry |
| Defenses | `orbit/defenses/` | Defense implementations and registry |
| Execution | `orbit/execution/` | `TurnExecutor` implementations (`TopologyExecutor`, `ScheduledExecutor`), `AgentScheduler` |
| Scheduler | `orbit/scheduler/` | Experiment loop and halt conditions |
| Scorers | `orbit/scorers/` | Generic security scorer + scenario-specific scorers |
| Wrapper | `orbit/wrapper/` | YAML loader, runner, CLI |
| Registry | `orbit/_registry.py` | Inspect entry point discovery |
| Scenario plugins | `orbit/scenarios/registry.py`, `orbit/tasks/builder.py` | `ScenarioPlugin` registry + shared Task builder ([Adding a New Scenario](adding-a-scenario.md)) |

## Conventions

- Python 3.11+
- All config models are frozen Pydantic (`BaseModel, frozen=True`)
- Async throughout (orchestrator, attacks, defenses)
- `TYPE_CHECKING` imports to avoid circular dependencies
- Literal types over enums for YAML-friendliness
- Extensibility via dict fields (`properties`, `config`, `metadata`)
