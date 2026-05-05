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

## ExperimentConfig

The root config object composes:

- `SetupConfig` -- agents, topology edges, memory settings
- `ScenarioConfig` -- task domain and sandbox environment
- `AttackConfig[]` -- attack specifications (timing, threat model, payload)
- `DefenseConfig[]` -- defense specifications (type, target agents, thresholds)
- `ExecutionConfig` -- multi-lane execution (parallel/sequential/round-robin)
- `SchedulerConfig` -- halt conditions, health checks, hooks
- `MetricsConfig` -- which metrics to compute

All config models are frozen (`frozen=True`). Use `model_copy(update=...)` to create variants.

## Three Levels of Control

**Topology** controls agent-to-agent interaction (which agent talks to which, via handoff/tool/direct_run edges).

**Execution lanes** control how groups of agents execute relative to each other (parallel, sequential, round-robin, with observation filtering and conflict detection).

**Scheduler** controls the experiment loop (max turns, timeout, convergence detection, health checks, custom halt conditions).

## Orchestrator

The `mas_orchestrator` `@solver` supports two execution paths:

**Path A -- Single-topology** (no lanes configured):
1. Extract config, build agents, apply pre-deployment attacks/defenses
2. Build topology, determine root agent
3. Run loop: attack injection -> agent execution -> defense monitoring -> attack evaluation

**Path B -- Multi-lane** (lanes configured):
1. Extract config, build per-lane runners
2. Run loop: dynamic lane assignment -> dispatch lanes -> resolve actions -> update metrics

Both paths use `ExperimentScheduler.run_loop()` and end with metrics recording.

## Attack/Defense Protocol

Both follow abstract base class patterns with registries:

- `AttackBase`: `inject()`, `should_activate()`, `evaluate_success()`
- `DefenseBase`: `apply_pre_deployment()`, `monitor_message()`, `on_tool_call()`

Attack types: `direct_injection`, `indirect_injection`, `compromised_agent`

Defense types: `prompt_vaccination`, `guardian_agent`, `monitor`, `tool_wrapper`, `secure_model`

## Runtime State

Mutable per-sample state shared via Inspect's `store_as()`:

| Store | Contents |
|-------|----------|
| `RuntimeMetrics` | Turns, wall clock, tokens, invoked agents, errors |
| `AttackLog` | Attempts, successes, pending injections |
| `DefenseLog` | Activations, blocks, detections, false/true positives |
| `EnvironmentState` | Global turn counter, message attribution, timeline |
| `LaneMetrics` | Per-lane turns, messages, agents invoked, conflicts |

## Key Modules

| Module | Path | Purpose |
|--------|------|---------|
| Configs | `orbit/configs/` | Frozen Pydantic models for all configuration |
| Topology presets | `orbit/configs/topology_presets.py` | General topology factories (single, star, mesh) |
| Memory tracker | `orbit/memory/tracker.py` | Per-agent memory visibility tracking |
| BrowserART | `orbit/scenarios/browser/browserart/` | Web agent safety scenario (task, solver, scorer, conditions) |
| SWE-Bench | `orbit/scenarios/coding/swe_bench/` | Coding agent security scenario (task, scorer, presets) |
| Orchestrator | `orbit/orchestrator/` | Central execution engine |
| Attacks | `orbit/attacks/` | Attack implementations and registry |
| Defenses | `orbit/defenses/` | Defense implementations and registry |
| Execution | `orbit/execution/` | Multi-lane execution (dispatcher, lane runner) |
| Environment | `orbit/environment/` | Shared state, observations, conflict detection |
| Scheduler | `orbit/scheduler/` | Experiment loop and halt conditions |
| Scorers | `orbit/scorers/` | Generic security scorer + scenario-specific scorers |
| Wrapper | `orbit/wrapper/` | YAML loader, runner, CLI |
| Registry | `orbit/_registry.py` | Inspect entry point discovery |

## Conventions

- Python 3.11+
- All config models are frozen Pydantic (`BaseModel, frozen=True`)
- Async throughout (orchestrator, attacks, defenses)
- `TYPE_CHECKING` imports to avoid circular dependencies
- Literal types over enums for YAML-friendliness
- Extensibility via dict fields (`properties`, `config`, `metadata`)
