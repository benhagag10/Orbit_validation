# Adding a New Scenario

This guide shows how to add a new scenario to Orbit using the scenario authoring
kit. It is grounded in real modules — every claim cites the code that enforces
it. Read [architecture.md](architecture.md) first for the data flow; this page
is the "how do I plug into it" companion.

## What you will build

A scenario is **one `ScenarioPlugin` registration plus one `@task` entry
point**, both funnelling into the shared Task builder. Here is the complete
shape of the work before any detail — a typical scenario package:

```
orbit/scenarios/<family>/myscenario/
├── __init__.py            # package marker (may be empty)
├── task.py                # ScenarioPlugin + @task factory — the skeleton below
├── configs.py             # frozen Pydantic model of the scenario's knobs
├── config_builder.py      # default_topology_template() + build_experiment_configs()
├── preset_registry.py     # thin binding of the shared preset loader to presets/
├── presets/
│   └── single_agent.yaml  # at least one named topology preset (a runnable config YAML)
├── security_presets.py    # named attack/defense bundles
└── scorer.py              # the scenario's @scorer
```

…plus three one-line registration edits outside the package
(`orbit/scenarios/registry.py`, `orbit/_registry.py`,
`orbit/scenarios/requirements.py`), and validation in three layers: discovery
(`uv run inspect list tasks orbit`), offline config validation
(`orbit validate` / `orbit --skip-preflight run … --dry-run`), and a full
offline Task build that exercises `expand` and every builder contract. The
[checklist](#step-by-step-checklist) walks the files in order;
[Validating your scenario](#validating-your-scenario) gives the exact commands
and states precisely what each layer can and cannot catch.

## Mental model

Every Orbit scenario turns **one `ExperimentConfig` into one Inspect `Task`**,
and both entry points converge on the *same* builder:

```
orbit run <yaml>            → security_benchmark(config) ──────┐
inspect eval orbit/<task>   → @task factory → ExperimentConfig ┤
                                                               └→ build_scenario_task() → Task
```

`orbit.tasks.builder.build_scenario_task` is the single place that does dataset
expansion, sample building (it serialises the *whole* config into
`Sample.metadata`), setup solver, main solver, scorers, and sandbox. Attack,
defense, topology, memory, and scheduling wiring live in exactly one consumer
per scenario, so the builder can *enforce* that config sections are never
silently dropped (`orbit/scenarios/registry.py` explains the rationale).

The only things that genuinely differ between scenarios are six dimensions,
declared on a `ScenarioPlugin` (`orbit/scenarios/registry.py`):

| Hook | What it returns | Default when `None` |
|------|-----------------|---------------------|
| `expand` | one config per sample (dataset expansion) | `[config]` (single sample) |
| `build_setup` | the `setup=` solver | `mas_environment_setup()` |
| `build_solver` | the main solver | shared `mas_orchestrator` |
| `build_scorers` | the scorer list | generic security/integrity scorers |
| `build_sandbox` | the Task `sandbox=` spec | derived from `scenario.sandbox` |
| `build_task_kwargs` | extra `Task(...)` kwargs (`message_limit`, `epochs`) | `{}` |

Everything else — baseline application, shorthand resolution, building Samples,
serialising the config, and applying attacks/defenses/topology at runtime — is
identical across scenarios and shared. **Adding a scenario is "register a
plugin", not "hand-write another Task factory."**

## Anatomy of a scenario package

A scenario lives under `orbit/scenarios/<family>/<name>/`, where `<family>` is
one of `coding`, `desktop`, `browser`, `customer_service`,
`cooperative_allocation`. RedCode-Gen
(`orbit/scenarios/coding/redcode_gen/`) is the simplest complete example — the
skeleton below is modelled on it.

**Mandatory**

| File | Purpose |
|------|---------|
| `task.py` | The `@task` factory (Inspect `-T` entry point) **and** the `ScenarioPlugin` registration. This is the only file the kit requires you to wire. |
| `__init__.py` | Package marker. |

**Conventional (present in every non-trivial scenario)**

| File | Purpose |
|------|---------|
| `configs.py` | Frozen Pydantic models: the per-scenario config (`RedCodeGenScenarioConfig`) and, usually, a per-task record (`RedCodeGenTask`). |
| `config_builder.py` | `default_topology_template()` and `build_experiment_configs()` — turns dataset tasks + topology template into `ExperimentConfig`s. Called by your `expand` hook. |
| `dataset_builder.py` | Loads and filters the raw dataset into task records. |
| `presets/` | This scenario's own topology/memory presets — one runnable YAML per preset (`<name>.yaml`). A new scenario ships its own here; **no presets are inherited**. |
| `preset_registry.py` | Binds the shared preset loader (`orbit/scenarios/presets.py`) to this scenario's `presets/` directory, exposing `get_preset_setup(name)` / `list_presets()`. Powers `--preset` / `<name>_preset`. |
| `security_presets.py` | `get_attack_preset(name)` / `get_defense_preset(name)` — named attack/defense bundles. |
| `scorer.py` | The scenario's `@scorer`. |
| `prompts.py` | System/user prompt text. |
| `data/` or `sandbox/` | Vendored data and/or a Docker `compose.yaml`. |

You do not have to use all of these — a scenario with no attacks needs no
`security_presets.py`, one with no dataset needs no `dataset_builder.py`, and one
with no shipped presets needs neither `presets/` nor `preset_registry.py`. The
kit only cares about `task.py`. (The skeleton below *does* import
`config_builder.py`, `preset_registry.py`, `security_presets.py`, and
`scorer.py`, so if you start from it, create those too — that is exactly what
the checklist walks through.)

**Presets are per-scenario.** A new scenario does not inherit any presets — it
ships its own. There are two ways to do that. Most scenarios drop one runnable
YAML per preset under `presets/` and bind the shared loader
(`orbit/scenarios/presets.py`) from `preset_registry.py`, the way `redcode_gen`
and `swe_bench` do. A scenario that builds its roster per sample resolves preset
*names* natively instead (τ²-Bench — see [`topology_source=SCENARIO`](#topology_sourcescenario)).
Either way the shared [Presets](presets.md) reference documents the friendly-flag
mapping and conflict rules once, over whatever preset set each scenario ships.

## Step-by-step checklist

Each step names the file it creates, what goes in it, and a smoke test for just
that step. The skeleton in the next section is the centrepiece — steps 3–6 are
the sibling modules it imports, so **`task.py` alone is not enough**: the plugin
registers at module import time, and registration itself imports
`config_builder.py`, `preset_registry.py`, and `security_presets.py` (a missing
one is a `ModuleNotFoundError` the moment anything touches your scenario). The
smoke tests below use the skeleton's fictional `coding/myscenario` path —
substitute your own.

1. **Create the package** — `orbit/scenarios/<family>/<name>/` with an
   `__init__.py` (may be empty).

2. **`configs.py`** — a frozen `BaseModel` for your scenario knobs (dataset
   filters, judge model, limits). Minimal version: the `MyScenarioConfig` class
   from the skeleton, moved here. (The skeleton inlines it to stay one file;
   every production scenario keeps it in `configs.py`.)

   ```bash
   uv run python -c "from orbit.scenarios.coding.myscenario.configs import MyScenarioConfig; print(MyScenarioConfig())"
   ```

3. **`config_builder.py`** — `default_topology_template() -> SetupConfig` (the
   roster used when no preset or inline setup is given) and
   `build_experiment_configs(...) -> list[ExperimentConfig]` (dataset tasks ×
   topology template → one config per sample; called by your `expand` hook).

   ```bash
   uv run python -c "from orbit.scenarios.coding.myscenario.config_builder import default_topology_template; print([a.name for a in default_topology_template().agents])"
   ```

4. **`presets/` + `preset_registry.py`** — at least one named topology preset.
   A preset is a runnable config YAML in the general orbit syntax — the loader
   consumes its `setup:` and optional `execution:` blocks — so the fastest
   start is copying one of `redcode_gen/presets/*.yaml` and editing.
   `preset_registry.py` can be a verbatim copy of
   `redcode_gen/preset_registry.py`: a thin binding (`PRESETS_DIR`,
   `get_preset_setup`, `list_presets`) of the shared loader
   (`orbit/scenarios/presets.py`) to your `presets/` directory.

   ```bash
   uv run python -c "from orbit.scenarios.coding.myscenario.preset_registry import list_presets; print(list_presets())"
   ```

5. **`security_presets.py`** — `get_attack_preset(name) -> list[AttackConfig]`
   and `get_defense_preset(name) -> list[DefenseConfig]` over module-level
   dicts of named bundles, raising `ValueError` naming the available presets on
   an unknown name (`redcode_gen/security_presets.py` is the model).

   ```bash
   uv run python -c "from orbit.scenarios.coding.myscenario.security_presets import get_attack_preset; print([a.name for a in get_attack_preset('jailbreak')])"
   ```

6. **`scorer.py`** — the scenario's Inspect `@scorer`. Minimal version: one
   scorer function returning a `Score`. It is imported lazily by
   `build_scorers`, so a missing or broken `scorer.py` passes `orbit validate`
   and only surfaces when the Task is actually built (the full-build check in
   the validation section).

7. **`task.py`** — the `ScenarioPlugin` + `@task` factory. Copy the skeleton
   below and rename.

8. **Register** — four edits, detailed in
   [Registration](#registration-lazy-loading-and-the-optional-extra): the
   `register_scenario(...)` call is already in the skeleton; add a `_LOADERS`
   entry (`orbit/scenarios/registry.py`), import your `@task`/`@scorer` in
   `orbit/_registry.py`, and add a `SCENARIO_REQUIREMENTS` entry
   (`orbit/scenarios/requirements.py`).

   ```bash
   uv run python -c "from orbit.scenarios.registry import get_scenario; print(get_scenario('myscenario').name)"
   ```

9. **Validate** — work through
   [Validating your scenario](#validating-your-scenario): discovery → offline
   validation → full offline Task build → contract tests → live smoke run.

## Minimal copyable skeleton

A complete, faithful `task.py` for a fictional `myscenario` (checklist step 7).
It is modelled directly on `redcode_gen/task.py` — single-agent coding-style
scenario, LLM-judge scorer, Docker sandbox — including an identical declarative
`ShorthandSpec`. Copy it, rename, and fill in the sibling modules it imports
(checklist steps 3–6: `config_builder.py`, `preset_registry.py` + a `presets/`
YAML dir, `security_presets.py`, `scorer.py`).

```python
"""
MyScenario task entry point — a template for new Orbit scenarios.

Registered with Inspect via ``orbit/_registry.py`` (so ``inspect eval
orbit/myscenario`` is discoverable) and with the scenario registry
(``register_scenario`` below + an ``orbit/scenarios/registry.py`` ``_LOADERS``
entry) so ``orbit run <yaml>`` and the ``-T`` factory build the *same* Task
through ``orbit.tasks.builder.build_scenario_task``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from inspect_ai import Task, task
from pydantic import BaseModel

from orbit.scenarios.params import split_csv
from orbit.scenarios.registry import ScenarioPlugin, register_scenario
from orbit.scenarios.shorthand import ShorthandSpec
from orbit.tasks.builder import build_scenario_task

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer

    from orbit.configs.experiment import ExperimentConfig

logger = logging.getLogger(__name__)

_SANDBOX_COMPOSE = str(Path(__file__).parent / "sandbox" / "compose.yaml")


# ---------------------------------------------------------------------------
# Scenario-config model. Production scenarios (redcode_gen) put this in a
# sibling ``configs.py``; inlined here so the template is one file.
# ---------------------------------------------------------------------------


class MyScenarioConfig(BaseModel, frozen=True):
    """Per-scenario knobs: dataset selection, judge model, limits."""

    data_path: str | None = None
    categories: list[str] | None = None
    task_ids: list[str] | None = None
    judge_model: str = "openai/gpt-4.1"
    max_turns: int = 5
    max_time_seconds: float = 120.0


# ---------------------------------------------------------------------------
# Scenario plugin hooks — each is TOTAL over ExperimentConfig (it receives the
# whole config and returns one Task ingredient).
# ---------------------------------------------------------------------------


def _scenario_config(config: ExperimentConfig) -> MyScenarioConfig:
    """Reconstruct the scenario config from sample metadata.

    The ``-T`` factory serialises the resolved config under
    ``myscenario_scenario_config``; a hand-written ``orbit run`` YAML may instead
    carry individual ``myscenario_*`` keys. Both shapes land here.
    """
    meta = config.metadata or {}
    if "myscenario_scenario_config" in meta:
        return MyScenarioConfig(**meta["myscenario_scenario_config"])
    return MyScenarioConfig(
        data_path=meta.get("myscenario_data_path"),
        categories=split_csv(meta.get("myscenario_categories")),
        task_ids=split_csv(meta.get("myscenario_task_ids")),
        judge_model=meta.get("myscenario_judge_model", "openai/gpt-4.1"),
        max_turns=config.scheduler.max_turns,
        max_time_seconds=config.scheduler.max_time_seconds,
    )


def _myscenario_expand(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Fan one config out into one ExperimentConfig per dataset task.

    CONTRACT: thread ``config.setup`` (topology), ``config.attacks`` and
    ``config.defenses`` into EVERY produced config. The builder rejects an
    expansion that drops a declared attack/defense type, or (because this is a
    ``topology_source=CONFIG`` plugin) collapses the agent count / drops shared
    memory — see the Contracts section.
    """
    from orbit.scenarios.coding.myscenario.config_builder import (
        build_experiment_configs,
    )

    configs = build_experiment_configs(
        scenario_config=_scenario_config(config),
        topology_template=config.setup if config.setup.agents else None,
        attacks=list(config.attacks) or None,
        defenses=list(config.defenses) or None,
    )
    if not configs:
        logger.warning("myscenario produced 0 samples — check filters")
    return configs


def _myscenario_scorers(config: ExperimentConfig) -> list[Scorer]:
    from orbit.scenarios.coding.myscenario.scorer import myscenario_scorer
    from orbit.scorers.security_scorer import security_scorer

    return [myscenario_scorer(_scenario_config(config).judge_model), security_scorer()]


def _shorthand() -> ShorthandSpec:
    """Declarative YAML shorthand: ``metadata.myscenario_preset`` /
    ``myscenario_attack_preset`` / ``myscenario_defense_preset``. The shared
    resolver reads these uniformly on every entry point (builder, ``orbit
    validate``, ``--dry-run``), so nothing can be silently dropped."""
    from orbit.scenarios.coding.myscenario.preset_registry import (
        get_preset_setup,
    )
    from orbit.scenarios.coding.myscenario.config_builder import (
        default_topology_template,
    )
    from orbit.scenarios.coding.myscenario.security_presets import (
        get_attack_preset,
        get_defense_preset,
    )

    return ShorthandSpec(
        prefix="myscenario",
        preset=get_preset_setup,
        attack_preset=get_attack_preset,
        defense_preset=get_defense_preset,
        default_setup=default_topology_template,
    )


MYSCENARIO_PLUGIN = register_scenario(
    ScenarioPlugin(
        name="myscenario",  # MUST equal scenario.name — this is the registry key
        shorthand=_shorthand(),
        expand=_myscenario_expand,
        # build_setup=None  →  the shared mas_environment_setup() default.
        # A lambda returning None means "run NO setup solver" (redcode_gen's
        # choice — a coding sandbox needs no environment setup).
        build_setup=lambda config: None,
        build_solver=None,  # keep the shared mas_orchestrator (applies attacks/defenses)
        build_scorers=_myscenario_scorers,
        build_sandbox=lambda config: ("docker", _SANDBOX_COMPOSE),
    )
)


# ---------------------------------------------------------------------------
# @task factory — a THIN resolver of flat -T params into one ExperimentConfig.
# ---------------------------------------------------------------------------


@task
def myscenario(
    categories: str | None = None,
    task_ids: str | None = None,
    judge_model: str = "openai/gpt-4.1",
    max_turns: int = 5,
    max_time_seconds: float = 120.0,
    preset: str | None = None,
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    data_path: str | None = None,
) -> Task:
    """MyScenario — one-line description shown by ``inspect list tasks``."""
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.metrics import MetricsConfig
    from orbit.configs.scenario import ScenarioConfig
    from orbit.configs.scheduler import SchedulerConfig
    from orbit.scenarios.coding.myscenario.preset_registry import (
        get_preset_setup,
    )
    from orbit.scenarios.coding.myscenario.config_builder import (
        default_topology_template,
    )
    from orbit.scenarios.coding.myscenario.security_presets import (
        get_attack_preset,
        get_defense_preset,
    )

    scenario_config = MyScenarioConfig(
        data_path=data_path,
        categories=split_csv(categories),
        task_ids=split_csv(task_ids),
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
    )

    setup = get_preset_setup(preset) if preset else default_topology_template()
    attacks = get_attack_preset(attack_preset) if attack_preset else None
    defenses = get_defense_preset(defense_preset) if defense_preset else None

    config = ExperimentConfig(
        name="myscenario",
        description="MyScenario safety benchmark",
        setup=setup,
        scenario=ScenarioConfig(name="myscenario", description="MyScenario"),
        attacks=attacks or [],
        defenses=defenses or [],
        scheduler=SchedulerConfig(
            max_turns=max_turns,
            max_time_seconds=max_time_seconds,
            halt_on_convergence=False,
        ),
        metrics=MetricsConfig(),
        metadata={"myscenario_scenario_config": scenario_config.model_dump()},
    )
    return build_scenario_task(config, MYSCENARIO_PLUGIN)
```

Two parallel surfaces are visible here, and they are deliberate:

- The **`@task` factory** resolves flat `-T` params (`-T preset=...`,
  `-T attack_preset=...`) into inline `setup`/`attacks`/`defenses` on the
  `ExperimentConfig`. It is the Inspect-CLI face of the scenario.
- The **`ShorthandSpec`** resolves the *YAML-metadata* equivalents
  (`metadata.myscenario_preset`, etc.) inside `build_scenario_task`, so an
  `orbit run` YAML gets the same resolution without going through the factory.

Both hand a config to `build_scenario_task`, which enforces the same contracts
either way.

### Plugin name vs task name

`ScenarioPlugin.name` **must equal** `scenario.name` in every YAML — it is the
registry key looked up by `get_scenario`. The `@task` function name is a
*separate* identifier: the Inspect CLI entry point (`inspect eval
orbit/<task-name>`). The two coincide for half the shipped scenarios and differ
for the other half, so check the table rather than guessing:

| Plugin name (= `scenario.name`) | `@task` name (`inspect eval orbit/<task>`) |
|---------------------------------|--------------------------------------------|
| `bigcodebench` | `bigcodebench` |
| `browserart` | `browserart_safety` |
| `code_ipi` | `code_ipi` |
| `converse` | `converse_safety` |
| `hospital` | `hospital_scheduling` |
| `jira_ticket` | `jira_ticket_allocation` |
| `meeting_scheduling` | `meeting_scheduling` |
| `osworld` | `osworld_safety` |
| `osworld_benchmark` | `osworld_benchmark` |
| `redcode_gen` | `redcode_gen` |
| `swe_bench` | `swe_bench_multi_issue` |
| `tau2` | `tau2` |

(The thirteenth registered task, `security_benchmark`, is the generic
YAML-driven entry point, not a scenario.) For a new scenario, giving the plugin
and the `@task` the same name — as the skeleton does — is the least surprising
choice.

## Contracts the builder enforces

`build_scenario_task` is a gate, not just a factory. If your plugin violates a
contract it raises a named exception — quote these when you hit them:

### `ScenarioExpansionError` (`orbit.tasks.builder`)

Two conditions, both checked centrally so a plugin author cannot forget them:

1. **Dropped attack/defense dimension.** `expand` (and the custom `resolve`
   hook, if any) may *add* attacks/defenses or transform an attack's
   *properties*, but must **preserve every declared `attack_type` /
   `defense_type`, count included**. The check is a per-type multiset floor
   (`_assert_experiment_dimensions_preserved`). Turning a declared `injection`
   attack into a `misuse` one leaves the `injection` floor unmet and is treated
   as a drop — so *transform within a type*, never across. This is the issue-#9
   bug class ("star silently ran benign") made a hard error.

2. **Collapsed topology.** For a `topology_source=CONFIG` plugin (the default),
   a produced config may not have **fewer agents** than the resolved config, nor
   turn `memory.shared=True` into `False`
   (`_assert_topology_preserved`). Adding agents is fine (that is a floor, not
   equality; agent expansion happens later in the orchestrator).

The fix for both is the same: thread `config.setup`, `config.attacks`, and
`config.defenses` through your `expand` into every produced config — exactly
what the skeleton's `_myscenario_expand` does.

### `ShorthandConflictError` (`orbit.scenarios.shorthand`)

Raised (unless `ORBIT_LENIENT=1`) when declared shorthand is contradictory or
unsatisfiable:

- a `<prefix>_preset` and an inline `setup` declare **materially different
  rosters** (the preset wins, so the inline `setup` mislabels the run);
- a resolver-materialised attack **targets an agent absent from the runtime
  roster** (it would silently fail to bind — a run labelled "attacked" that runs
  benign).

Both are construct-validity failures: the run would execute with a topology or
threat different from the one its config claims.

### `ScenarioImportError` / `MissingScenarioDependencyError` (`orbit.scenarios.registry`)

Raised by `get_scenario` when a *known* scenario (one in `_LOADERS`) fails to
import. If the failure is explained by a missing optional extra declared in
`requirements.py`, you get `MissingScenarioDependencyError` naming the exact
`uv sync --extra <name>` command; otherwise `ScenarioImportError` (a bug in your
module, or an undeclared dependency). The builder never silently falls back to
the generic single-sample path for a registered scenario — that would rerun the
issue-#9 failure.

## Shorthand and the conflict policy

`ShorthandSpec` (`orbit/scenarios/shorthand.py`) declares a scenario's
YAML-metadata surface. The shared resolver implements the shape once — read key
→ arbitrate conflicts → materialise → strip the consumed key → record a
breadcrumb — so it is identical on the builder, `orbit validate`, and
`--dry-run`. Three metadata keys, all prefixed by `spec.prefix`:

| Key | Resolves to |
|-----|-------------|
| `<prefix>_preset` | a named topology/memory preset → `SetupConfig` |
| `<prefix>_attack_preset` | a named attack bundle → `list[AttackConfig]` |
| `<prefix>_defense_preset` | a named defense bundle → `list[DefenseConfig]` |

The conflict policy is uniform across every scenario:

- **Preset vs inline setup — the explicit preset WINS.** If the inline
  `setup.agents` roster **equals** the preset's roster, the inline block is
  treated as redundant documentation and resolution is **silent**. A
  **materially differing** roster is a `ShorthandConflictError` by default
  (downgradable to a warning with `ORBIT_LENIENT=1`).
- **Preset vs inline attacks/defenses — the inline section WINS**, the preset is
  ignored **with a warning** and a `<prefix>_ignored_<kind>_preset` breadcrumb.
- **Materialised attack targets** are validated against the runtime roster
  (post-resolution topology, expanded to runtime names — so
  template-replicating scenarios validate against `solver_0`-style names). A
  dangling target is a `ShorthandConflictError`.
- **Baselines.** A preset is never materialised into a dimension the config's
  `baseline_mode` strips, so a `no_attack` / `benign` control cannot re-acquire
  its attack through a preset. Gated in the resolver *and* re-stripped centrally
  after resolution (belt and suspenders for custom `resolve` hooks).

The resolver records what it did in `metadata` breadcrumbs so a log is
self-describing:

| Breadcrumb | Meaning |
|------------|---------|
| `<prefix>_resolved_preset` | the preset that supplied the topology |
| `<prefix>_resolved_<kind>_preset` | the attack/defense preset that was materialised |
| `<prefix>_ignored_<kind>_preset` | a preset overridden by an inline section |
| `<prefix>_baseline_stripped_<kind>_preset` | a preset dropped by the baseline gate |

Idempotence: the resolver strips the keys it consumes, so resolving an
already-resolved config is a no-op — which is why `build_scenario_task`,
`orbit validate`, and `--dry-run` can all run it.

> **`resolve` escape hatch.** `ScenarioPlugin.resolve` is a hand-rolled
> resolution hook run after the spec resolver and before `expand`. **Prefer a
> `ShorthandSpec`** — every current scenario's resolution is spec-expressible,
> and hand-rolled hooks are exactly where the issue-#9 shorthand-drop bugs
> lived. If you must use `resolve`, it MUST be idempotent (a no-op on an
> already-resolved config).

## `expand_agent_names` — validation for template-replicating scenarios

Offline validators (`orbit validate`, `--dry-run`) check that attack/defense
targets name agents that will actually exist at runtime (issues #7/#4). By
default the runtime roster is just `config.setup.agents`. A scenario whose
`expand` **renames or replicates** its declared agents must opt in through
`expand_agent_names`, or the targets will be checked against the wrong names.

Today only `swe_bench` needs this — its `expand` replicates the topology
template once per issue, so `solver` becomes `solver_0 .. solver_{n-1}`
(`orbit/scenarios/coding/swe_bench/task.py`):

```python
def _swe_bench_agent_names(config: ExperimentConfig) -> set[str]:
    num_issues = max(1, _scenario_config(config).num_issues)
    template = config.setup if config.setup.agents else default_topology_template()
    return {
        replicated_agent_name(agent.name, idx)
        for agent in template.agents
        for idx in range(num_issues)
    }
```

Two rules: it must stay **cheap and offline** (derive names from the config
alone — no dataset fetch, no Docker), and it must use the **same naming helper**
as the real `expand` so the two cannot drift. Most scenarios leave it `None`
(identity: runtime names == declared names). It is consumed via
`registry.runtime_agent_names`.

## `topology_source=SCENARIO`

Set `topology_source=TopologySource.SCENARIO` when your scenario **derives its
roster from its own preset selector and deliberately ignores `config.setup`**
— the τ²-Bench pattern, where each task builds a native per-domain roster. This
is the other way a scenario ships presets: instead of binding the shared YAML
loader, it resolves preset **names** natively (no shipped `presets/` YAML to
open). Consequences:

- The topology floor is **skipped** (there is no `config.setup` roster to hold
  the line at).
- The `<prefix>_preset` key is **not consumed** by the shared resolver — your
  `expand` reads it and builds the roster natively.
- Instead of materialising a topology, the resolver only **arbitrates** a
  declared inline `setup` against what the preset will build.

To make that arbitration precise, give the `ShorthandSpec` two extra fields and
leave `preset` as `None` (τ²-Bench, `orbit/scenarios/customer_service/tau2/task.py`):

```python
ScenarioPlugin(
    name="tau2",
    shorthand=ShorthandSpec(
        prefix="tau2",
        defense_preset=get_defense_preset,    # security presets resolve as usual
        expected_roster=_expected_roster,     # names the preset will build
        default_preset="dual_control",        # assumed when no *_preset key
    ),
    expand=_tau2_expand,                      # reads tau2_preset itself
    ...
    topology_source=TopologySource.SCENARIO,
)
```

- Only the *topology* preset key changes behaviour under `SCENARIO`:
  `attack_preset` / `defense_preset` resolution can still hang off the spec
  (τ² wires `defense_preset`) and is materialised by the shared resolver
  exactly as for a `CONFIG` plugin.
- `expected_roster(config, preset) -> set[str] | None` returns the agent
  names the preset will build (τ²'s specialists are domain-dependent, so it
  takes the config too). Return `None` for "unknown" — the resolver then falls
  back to a warning instead of raising.
- `default_preset` is the preset `expand` assumes when no
  `<prefix>_preset` key is present, so an inline setup is arbitrated against
  what actually runs.

With these set, a matching inline roster passes silently, a materially different
one raises `ShorthandConflictError`, and an unknown preset degrades to a
warning — instead of the blanket "inline setup is IGNORED" warning a
spec-less `SCENARIO` plugin still emits.

## Registration, lazy loading, and the optional extra

Four edits register a scenario. Miss one and the symptom is specific:

**1. `_LOADERS` in `orbit/scenarios/registry.py`** — maps the scenario name to
the module whose import registers the plugin. This is what lets `get_scenario`
lazily import your module the first time the name is requested, so optional
dependencies stay optional:

```python
_LOADERS: dict[str, str] = {
    ...
    "myscenario": "orbit.scenarios.coding.myscenario.task",
}
```

If the `scenario.name` encodes a sub-domain (τ² ships `tau2_airline`,
`tau2_retail`, ... which all resolve to the `tau2` plugin), also add a prefix to
the central `_ALIAS_PREFIXES` table **and** mirror it in your plugin's
`alias_prefixes=(...)`; a registry test pins the two together.

**2. `register_scenario(ScenarioPlugin(...))`** at module import time — done in
the skeleton. Without it, `get_scenario` imports your module but finds no plugin
and falls back to the generic path.

**3. Import in `orbit/_registry.py`** — this is the Inspect plugin entry point
(`pyproject.toml [project.entry-points.inspect_ai] orbit = "orbit._registry"`).
Import your `@task` (and `@scorer`) here so `inspect list tasks orbit` and
`inspect eval` can find them. If your scenario needs an optional extra, guard
the import and register a stub that raises an actionable error, the way
`swe_bench` / `bigcodebench` / the DCOP trio already do:

```python
# core scenario — import directly
from orbit.scenarios.coding.myscenario.task import myscenario  # noqa: F401
from orbit.scenarios.coding.myscenario.scorer import myscenario_scorer  # noqa: F401
```

**4. `SCENARIO_REQUIREMENTS` in `orbit/scenarios/requirements.py`** — the single
source of truth for install requirements. `get_scenario` consults it to turn a
failed lazy import into an actionable missing-dependency message, a test asserts
every registered scenario has an entry, and the install matrix (in `README.md`
and `docs/scenarios.md`) is kept in sync with it. For a core scenario with a
Docker sandbox and a data-fetch step:

```python
"myscenario": ScenarioRequirements(
    scenario="myscenario",
    extra=None,                 # or "myscenario" if you add a pip extra
    packages=(),                # importable module names the extra provides
    needs_docker=True,          # sandbox runs generated code
    data_step="uv run python scripts/fetch_myscenario_data.py",
    verify="myscenario",        # the `orbit verify-setup <arg>` argument, if any
),
```

If you *do* add a pip extra, declare it in `[project.optional-dependencies]` in
`pyproject.toml` and list its importable module names in `packages` — that is
how `missing_extra_message` reports exactly which modules are absent. Keep the
install matrix (`README.md` and `docs/scenarios.md`) in sync.

## Validating your scenario

Work outward from cheap structural checks to a live smoke run.

**Discovery** — the task should list and the plugin should load:

```bash
uv run inspect list tasks orbit          # your @task should appear
```

**Offline validation** — `orbit validate` and `orbit run --dry-run` both run
the shared shorthand-resolution pipeline plus `ConfigValidator`, entirely
offline. (`--skip-preflight` is a flag on the `orbit` group itself, so it goes
*before* `run`. You can validate a shipped preset YAML directly — presets are
runnable configs.)

```bash
uv run orbit validate examples/myscenario.yaml
uv run orbit --skip-preflight run examples/myscenario.yaml --dry-run
```

Be precise about what this layer does and does not catch:

- **Caught here:** an unknown or unimportable scenario module
  (`MissingScenarioDependencyError` / `ScenarioImportError`), a
  `ShorthandConflictError` (preset vs materially-different inline setup, a
  preset-materialised attack targeting an absent agent), a `resolve`-stage
  dimension drop (`ScenarioExpansionError` at stage `"resolving"`), and
  `ConfigValidator` errors — attack/defense targets are checked against the
  *runtime* roster derived by `expand_agent_names`, not by running `expand`.
- **NOT caught here:** anything inside `expand` or the other build hooks.
  `dry_run()` (`orbit/wrapper/runner.py`) never calls `build_scenario_task`,
  so an expand-stage `ScenarioExpansionError`, the topology floor
  (`_assert_topology_preserved`), a broken `scorer.py`, and dataset-loading
  failures all pass `--dry-run` silently.

**Full Task build** — the cheapest command that actually exercises `expand` and
every builder contract. It constructs the Task without executing it, so it
needs no API key (it does run your dataset loading, so vendored or fetched data
must be present):

```bash
uv run python -c "
from orbit.tasks.builder import build_scenario_task
from orbit.wrapper.yaml_loader import load_experiment_config

task = build_scenario_task(load_experiment_config('examples/myscenario.yaml'))
print(type(task).__name__, len(task.dataset), 'samples')
"
```

This is where a dropped attack dimension or a collapsed topology in your
`expand` raises `ScenarioExpansionError` — before you spend a token.

**Contract tests** — the suite includes scenario-portability and
config-completeness tests (`tests/test_scenario_portability.py`,
`tests/test_issue9_config_completeness.py`, `tests/test_scenario_requirements.py`)
that exercise the builder contracts and assert every registered scenario has a
`requirements.py` entry. Run the whole suite once your plugin lands:

```bash
uv run pytest tests/ -x -v
```

**Construct validity** — after any live batch, confirm the topology did not
silently degenerate (a star collapsing to one agent still produces a number):

```bash
uv run python scripts/verify_construct_validity.py logs/ -v
```

Any construct-validity FAIL is a P0 blocker. The structural checks catch the
mechanical failures; a human still eyeballs a few logs to confirm the output
makes sense.

## Common failure modes

- **`ModuleNotFoundError` for `preset_registry` / `config_builder` /
  `security_presets` the first time anything touches the scenario.** The plugin
  registers at module import time, and building its `ShorthandSpec` imports all
  three siblings — `task.py` alone is not enough. Create the checklist step 3–5
  files first (or stub them). `scorer.py` is the late one: it is only imported
  when the Task is built, so its absence passes `orbit validate` and surfaces in
  the full-build one-liner.
- **`orbit validate` and `--dry-run` pass, but the real run fails — or runs the
  wrong experiment.** Offline validation never calls `expand`, so a bug that
  drops attacks or collapses the topology inside `expand` is invisible until the
  Task is built. Run the full-build one-liner above; the builder's
  `ScenarioExpansionError` contracts fire there.
- **Plugin name vs task name confusion.** `inspect eval orbit/<X>` wants the
  `@task` *function* name; `scenario.name` in a YAML wants the *plugin* name —
  and the two differ for half the shipped scenarios (see the
  [table](#plugin-name-vs-task-name)). The worst variant is a typo'd
  `scenario.name`: `build_scenario_task` falls back to the generic
  single-sample path — loudly, with a warning naming the registered
  scenarios — and the eval still runs, minus your dataset expansion and
  scorers.
- **`ShorthandConflictError` on a config that declares both a preset and an
  inline setup.** The preset wins, so a materially different inline roster
  mislabels the run and is rejected — declare exactly one topology source.
  (`ORBIT_LENIENT=1` downgrades this to a warning for debugging only; the
  conflict means the config mislabels the experiment.)

## See also

- [architecture.md](architecture.md) — data flow, orchestrator, execution paths.
- [configuration.md](configuration.md) — the YAML schema and per-scenario
  metadata/shorthand keys.
- [presets.md](presets.md) — how presets map to topologies.
- `orbit/scenarios/registry.py`, `orbit/scenarios/shorthand.py`,
  `orbit/tasks/builder.py` — the kit, with contract docstrings.
