"""
OSWorld task entry points.

Provides two Inspect tasks registered via _registry.py:

- ``osworld_safety`` — OS-Harm safety evaluation (harmful task refusal)
- ``osworld_benchmark`` — Standard OSWorld capability evaluation
  (benign desktop tasks, optionally with injection attacks layered on)

Both use the OSWorld desktop environment from ``inspect_evals`` for
sandbox execution with the ``computer`` tool.

Construction is unified through the scenario-agnostic builder
(``orbit.tasks.builder.build_scenario_task``): each scenario's ``expand`` hook
reads the task-selection config from ``config.metadata`` and threads
``config.setup`` (topology), ``config.attacks`` and ``config.defenses`` into
every per-task config, so ``inspect eval -T ...`` and ``orbit run <yaml>``
construct identical Tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from inspect_ai import Task, task

from orbit.scenarios.desktop.osworld.config_builder import (
    _resolve_osworld_compose_file,
    build_benchmark_configs,
    build_experiment_configs,
    default_topology_template,
)
from orbit.scenarios.desktop.osworld.configs import OSWorldScenarioConfig
from orbit.scenarios.registry import ScenarioPlugin, register_scenario
from orbit.tasks.builder import build_scenario_task

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer

    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.setup import SetupConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _split(value: object) -> list[str] | None:
    """Normalise a comma-string or list into a list of strings (or None)."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _scenario_config(
    config: ExperimentConfig, *, default_dataset: str
) -> OSWorldScenarioConfig:
    """Reconstruct the task-selection config from metadata + scheduler.

    The -T factory serializes the resolved ``OSWorldScenarioConfig`` under
    ``osworld_scenario_config``; YAML/​orbit-run configs use the individual
    ``osworld_*`` keys.
    """
    meta = config.metadata or {}
    if "osworld_scenario_config" in meta:
        return OSWorldScenarioConfig(**meta["osworld_scenario_config"])
    return OSWorldScenarioConfig(
        dataset=meta.get("osworld_dataset", default_dataset),
        apps=_split(meta.get("osworld_app")),
        threat_categories=_split(meta.get("osworld_threat_category")),
        violation_types=_split(meta.get("osworld_violation_type")),
        task_ids=_split(meta.get("osworld_task_ids")),
        seed=meta.get("osworld_seed"),
        judge_model=meta.get("osworld_judge_model", "openai/gpt-4.1"),
        max_turns=config.scheduler.max_turns,
        max_time_seconds=config.scheduler.max_time_seconds,
        max_screenshots=meta.get("osworld_max_screenshots", 1),
        computer_timeout=meta.get("osworld_computer_timeout", 180),
    )


def _judge_model(config: ExperimentConfig) -> str:
    meta = config.metadata or {}
    if "osworld_scenario_config" in meta:
        return meta["osworld_scenario_config"].get("judge_model", "openai/gpt-4.1")
    return meta.get("osworld_judge_model", "openai/gpt-4.1")


def _resolve_topology(
    condition: str | None,
    agents: str | None,
    topology: str | None,
    memory: str | None,
    instructions: str | None,
    topology_file: str,
) -> tuple[SetupConfig, str | None]:
    """Resolve topology template and condition name from task parameters.

    Returns (topology_template, resolved_condition_name).
    """
    from orbit.configs.setup import SetupConfig

    has_condition = condition is not None
    has_agents = agents is not None
    has_custom_topo = topology_file != "default"
    has_sub_params = any(p is not None for p in [topology, memory, instructions])

    if has_condition and has_agents:
        raise ValueError("Cannot specify both --condition and --agents.")
    if has_condition and has_custom_topo:
        raise ValueError("Cannot specify both --condition and --topology_file.")
    if has_agents and has_custom_topo:
        raise ValueError("Cannot specify both --agents and --topology_file.")
    if (has_condition or has_custom_topo) and has_sub_params:
        raise ValueError(
            "--topology, --memory, --instructions are only used with --agents."
        )
    if not has_agents and has_sub_params:
        raise ValueError("--topology, --memory, --instructions require --agents.")

    resolved_condition: str | None = None

    if has_condition:
        from orbit.scenarios.desktop.osworld.condition_presets import (
            get_condition_setup,
        )
        topo_template = get_condition_setup(condition)
        resolved_condition = condition
    elif has_agents:
        from orbit.scenarios.desktop.osworld.condition_presets import (
            get_condition_setup,
            resolve_condition,
        )
        resolved_condition = resolve_condition(
            agents=agents,
            topology=topology or "star",
            memory=memory or "none",
            instructions=instructions or "detailed",
        )
        topo_template = get_condition_setup(resolved_condition)
    elif has_custom_topo:
        import yaml

        with open(topology_file, encoding="utf-8") as f:
            topo_data = yaml.safe_load(f)
        if "setup" in topo_data and ("scenario" in topo_data or "name" in topo_data):
            topo_data = topo_data["setup"]
        topo_template = SetupConfig(**topo_data)
    else:
        topo_template = default_topology_template()

    return topo_template, resolved_condition


def _osworld_template_config(
    *,
    scenario_name: str,
    scenario_config: OSWorldScenarioConfig,
    topology_template: SetupConfig,
    attacks: list | None,
    defenses: list | None,
) -> ExperimentConfig:
    """Build the single template ExperimentConfig passed to the builder."""
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.metrics import MetricsConfig
    from orbit.configs.scenario import ScenarioConfig
    from orbit.configs.scheduler import SchedulerConfig

    return ExperimentConfig(
        name=scenario_name,
        description=f"OSWorld {scenario_name}",
        setup=topology_template,
        scenario=ScenarioConfig(name=scenario_name, description=f"OSWorld {scenario_name}"),
        attacks=attacks or [],
        defenses=defenses or [],
        scheduler=SchedulerConfig(
            max_turns=scenario_config.max_turns,
            max_time_seconds=scenario_config.max_time_seconds,
            halt_on_convergence=False,
        ),
        metrics=MetricsConfig(),
        metadata={
            "osworld_scenario_config": scenario_config.model_dump(),
        },
    )


# ---------------------------------------------------------------------------
# osworld_safety plugin + factory
# ---------------------------------------------------------------------------


def _osworld_safety_expand(config: ExperimentConfig) -> list[ExperimentConfig]:
    scenario_config = _scenario_config(config, default_dataset="osharm")
    topology_template = config.setup if config.setup.agents else default_topology_template()
    configs = build_experiment_configs(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=list(config.attacks) or None,
        defenses=list(config.defenses) or None,
    )
    if not configs:
        logger.warning("No samples built — check scenario config and available tasks")
    return configs


def _osworld_safety_scorers(config: ExperimentConfig) -> list[Scorer]:
    from orbit.scenarios.desktop.osworld.scorer import osworld_scorer
    from orbit.scorers.security_scorer import security_scorer

    return [osworld_scorer(_judge_model(config)), security_scorer()]


OSWORLD_SAFETY_PLUGIN = register_scenario(
    ScenarioPlugin(
        name="osworld",
        expand=_osworld_safety_expand,
        build_setup=lambda config: None,
        build_solver=None,  # default mas_orchestrator
        build_scorers=_osworld_safety_scorers,
        build_sandbox=lambda config: ("docker", _resolve_osworld_compose_file()),
    )
)


@task
def osworld_safety(
    dataset: str = "osharm",
    app: str | None = None,
    threat_category: str | None = None,
    violation_type: str | None = None,
    task_ids: str | None = None,
    seed: int | None = None,
    max_turns: int = 100,
    max_time_seconds: float = 600.0,
    max_screenshots: int = 1,
    computer_timeout: int = 180,
    topology_file: str = "default",
    condition: str | None = None,
    agents: str | None = None,
    topology: str | None = None,
    memory: str | None = None,
    instructions: str | None = None,
    defense_preset: str | None = None,
    attack_preset: str | None = None,
    judge_model: str = "openai/gpt-4.1",
    orchestrator: str = "v1",
) -> Task:
    """OSWorld multi-agent safety benchmark task.

    Evaluates computer-use agent safety against harmful task instructions
    from the OS-Harm dataset on the OSWorld desktop environment.

    Topology can be specified in three mutually exclusive ways:

    1. **Named condition** (``condition``): a pre-validated multi-agent
       topology preset (e.g. ``"star_specialist"``).
    2. **Friendly parameters** (``agents``, ``topology``, ``memory``,
       ``instructions``): human-readable knobs that resolve to a named
       condition.
    3. **Custom YAML** (``topology_file``): path to a SetupConfig YAML.

    Examples:
        # Basic evaluation (requires OS-Harm data):
        inspect eval orbit/osworld_safety \\
            --limit 5 --model openai/gpt-4o

        # Named condition preset:
        inspect eval orbit/osworld_safety \\
            -T condition=star_specialist --limit 5 --model openai/gpt-4o

    Args:
        dataset: Dataset to load. ``"osharm"`` loads the full OS-Harm
            dataset; sub-variants ``"osharm_misuse"``, ``"osharm_injection"``,
            ``"osharm_misbehavior"`` pre-filter by threat category.
        app: Filter tasks by target application (comma-separated).
        task_ids: Comma-separated list of task IDs to include.
        threat_category: Filter by threat category (comma-separated).
        violation_type: Filter by violation type (comma-separated).
        seed: Random seed for task sampling.
        max_turns: Maximum turns per task.
        max_time_seconds: Maximum wall-clock seconds per task.
        max_screenshots: Maximum screenshots per turn for the computer tool.
        computer_timeout: Timeout in seconds for the computer tool.
        topology_file: Topology template: ``"default"`` or path to YAML.
        condition: Named condition preset (e.g. ``"star_specialist"``).
        agents: Agent configuration style.
        topology: Topology type (used with ``agents``).
        memory: Memory level (used with ``agents``).
        instructions: Instruction detail level (used with ``agents``).
        defense_preset: Defense preset name (e.g. ``"basic"``).
        attack_preset: Attack preset name (e.g. ``"prompt_injection"``).
        judge_model: Model for the safety judge scorer.
        orchestrator: ``"v1"`` or ``"v2"`` — selects the orchestrator solver.
    """
    scenario_config = OSWorldScenarioConfig(
        dataset=dataset,
        apps=_split(app),
        threat_categories=_split(threat_category),
        violation_types=_split(violation_type),
        task_ids=_split(task_ids),
        seed=seed,
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
        max_screenshots=max_screenshots,
        computer_timeout=computer_timeout,
    )

    topology_template, resolved_condition = _resolve_topology(
        condition=condition,
        agents=agents,
        topology=topology,
        memory=memory,
        instructions=instructions,
        topology_file=topology_file,
    )

    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.desktop.osworld.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset, condition=resolved_condition)
    if defense_preset:
        from orbit.scenarios.desktop.osworld.presets import get_defense_preset

        defenses = get_defense_preset(defense_preset)

    config = _osworld_template_config(
        scenario_name="osworld",
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=attacks,
        defenses=defenses,
    )
    return build_scenario_task(
        config, OSWORLD_SAFETY_PLUGIN, orchestrator=orchestrator
    )


# ---------------------------------------------------------------------------
# osworld_benchmark plugin + factory
# ---------------------------------------------------------------------------


def _osworld_benchmark_expand(config: ExperimentConfig) -> list[ExperimentConfig]:
    scenario_config = _scenario_config(config, default_dataset="osworld")
    topology_template = config.setup if config.setup.agents else default_topology_template()
    configs = build_benchmark_configs(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=list(config.attacks) or None,
        defenses=list(config.defenses) or None,
    )
    if not configs:
        logger.warning("No benchmark samples built — check corpus and filters")
    return configs


def _osworld_benchmark_scorers(config: ExperimentConfig) -> list[Scorer]:
    from orbit.scenarios.desktop.osworld.scorer import (
        osworld_capability_scorer,
        osworld_scorer,
    )

    scorers: list[Scorer] = [osworld_capability_scorer()]
    if config.attacks:
        from orbit.scorers.security_scorer import security_scorer

        scorers.append(osworld_scorer(_judge_model(config)))
        scorers.append(security_scorer())
    return scorers


OSWORLD_BENCHMARK_PLUGIN = register_scenario(
    ScenarioPlugin(
        name="osworld_benchmark",
        expand=_osworld_benchmark_expand,
        build_setup=lambda config: None,
        build_solver=None,
        build_scorers=_osworld_benchmark_scorers,
        build_sandbox=lambda config: ("docker", _resolve_osworld_compose_file()),
    )
)


@task
def osworld_benchmark(
    dataset: str = "osworld",
    app: str | None = None,
    task_ids: str | None = None,
    seed: int | None = None,
    max_turns: int = 100,
    max_time_seconds: float = 600.0,
    max_screenshots: int = 1,
    computer_timeout: int = 180,
    condition: str | None = None,
    agents: str | None = None,
    topology: str | None = None,
    memory: str | None = None,
    instructions: str | None = None,
    topology_file: str = "default",
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    judge_model: str = "openai/gpt-4.1",
    orchestrator: str = "v1",
) -> Task:
    """OSWorld standard benchmark task for capability evaluation.

    Evaluates computer-use agents on benign desktop automation tasks
    from the standard OSWorld benchmark (xlang-ai/OSWorld). Tasks are
    scored on functional correctness by the OSWorld built-in evaluator.

    Can also be used for injection experiments: when ``attack_preset``
    is set, attacks are layered onto the benign tasks and both
    capability and safety scorers run.

    Args:
        dataset: OSWorld dataset variant (``"osworld"`` for the full
            benchmark or ``"osworld_small"`` for the small corpus).
        app: Filter by app (comma-separated).
        task_ids: Comma-separated task IDs to include.
        seed: Random seed for sampling.
        max_turns: Maximum turns per task.
        max_time_seconds: Maximum wall-clock seconds per task.
        max_screenshots: Maximum screenshots per turn.
        computer_timeout: Timeout for the computer tool.
        condition: Named condition preset.
        agents: Agent style (single, batch, specialist, step).
        topology: Topology type (star, round_robin, delegation).
        memory: Memory level.
        instructions: Instruction detail level.
        topology_file: Custom topology YAML path.
        attack_preset: Attack preset for injection experiments.
        defense_preset: Defense preset name.
        judge_model: Model for safety judge (used in injection mode).
        orchestrator: ``"v1"`` or ``"v2"`` — selects the orchestrator solver.
    """
    osworld_corpus = "small" if dataset == "osworld_small" else "all"

    scenario_config = OSWorldScenarioConfig(
        dataset=dataset,
        apps=_split(app),
        task_ids=_split(task_ids),
        seed=seed,
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
        max_screenshots=max_screenshots,
        computer_timeout=computer_timeout,
        osworld_corpus=osworld_corpus,
    )

    topology_template, resolved_condition = _resolve_topology(
        condition=condition,
        agents=agents,
        topology=topology,
        memory=memory,
        instructions=instructions,
        topology_file=topology_file,
    )

    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.desktop.osworld.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset, condition=resolved_condition)
    if defense_preset:
        from orbit.scenarios.desktop.osworld.presets import get_defense_preset

        defenses = get_defense_preset(defense_preset)

    config = _osworld_template_config(
        scenario_name="osworld_benchmark",
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=attacks,
        defenses=defenses,
    )
    return build_scenario_task(
        config, OSWORLD_BENCHMARK_PLUGIN, orchestrator=orchestrator
    )
