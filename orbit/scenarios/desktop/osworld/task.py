"""
OSWorld task entry points.

Provides two Inspect tasks registered via _registry.py:

- ``osworld_safety`` — OS-Harm safety evaluation (harmful task refusal)
- ``osworld_benchmark`` — Standard OSWorld capability evaluation
  (benign desktop tasks, optionally with injection attacks layered on)

Both use the OSWorld desktop environment from ``inspect_evals`` for
sandbox execution with the ``computer`` tool.
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.desktop.osworld.config_builder import (
    _resolve_osworld_compose_file,
    build_experiment_configs_from_scenario,
    default_topology_template,
)
from orbit.scenarios.desktop.osworld.configs import OSWorldScenarioConfig
from orbit.scenarios.desktop.osworld.scorer import osworld_scorer

logger = logging.getLogger(__name__)


@task
def osworld_safety(
    dataset: str = "osharm",
    app: str | None = None,
    threat_category: str | None = None,
    violation_type: str | None = None,
    task_ids: str | None = None,
    max_tasks: int | None = None,
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
            -T max_tasks=5 --model openai/gpt-4o

        # Filter by app:
        inspect eval orbit/osworld_safety \\
            -T app=thunderbird -T max_tasks=10 --model openai/gpt-4o

        # Named condition preset:
        inspect eval orbit/osworld_safety \\
            -T condition=star_specialist -T max_tasks=5 --model openai/gpt-4o

        # Friendly parameters:
        inspect eval orbit/osworld_safety \\
            -T agents=specialist -T topology=star -T max_tasks=5 \\
            --model openai/gpt-4o

        # With defense preset:
        inspect eval orbit/osworld_safety \\
            -T defense_preset=basic --model openai/gpt-4o

    Args:
        dataset: Dataset to load. ``"osharm"`` loads the full OS-Harm
            dataset; sub-variants ``"osharm_misuse"``, ``"osharm_injection"``,
            ``"osharm_misbehavior"`` pre-filter by threat category.
        app: Filter tasks by target application (comma-separated).
        task_ids: Comma-separated list of task IDs to include. Applied
            before all other filters.
        threat_category: Filter by threat category (comma-separated).
        violation_type: Filter by violation type (comma-separated).
        max_tasks: Limit number of tasks to evaluate.
        seed: Random seed for task sampling.
        max_turns: Maximum turns per task.
        max_time_seconds: Maximum wall-clock seconds per task.
        max_screenshots: Maximum screenshots per turn for the computer tool.
        computer_timeout: Timeout in seconds for the computer tool.
        topology_file: Topology template: ``"default"`` or path to YAML.
        condition: Named condition preset (e.g. ``"star_specialist"``).
            Mutually exclusive with ``agents`` and ``topology_file``.
        agents: Agent configuration style (``"single"``, ``"batch"``,
            ``"specialist"``, ``"app_specialist"``, ``"step"``).
            ``"specialist"`` uses tool-action specialists (click, type,
            key, scroll). ``"app_specialist"`` uses application-based
            specialists (email, browser, document, code). Resolves to a
            condition. Mutually exclusive with ``condition`` and
            ``topology_file``.
        topology: Topology type (``"star"``, ``"round_robin"``,
            ``"delegation"``). Used with ``agents``.
        memory: Memory level (``"none"``, ``"own_actions"``,
            ``"own_reasoning"``, ``"shared_actions"``, ``"full"``).
            Used with ``agents``.
        instructions: Instruction detail level (``"detailed"``,
            ``"relaxed"``). Used with ``agents``.
        defense_preset: Defense preset name (e.g. ``"basic"``).
        attack_preset: Attack preset name (e.g. ``"prompt_injection"``).
        judge_model: Model for the safety judge scorer.
    """
    # Parse comma-separated filter parameters
    apps = [a.strip() for a in app.split(",")] if app else None
    threat_categories = (
        [c.strip() for c in threat_category.split(",")]
        if threat_category
        else None
    )
    violation_types = (
        [v.strip() for v in violation_type.split(",")]
        if violation_type
        else None
    )
    parsed_task_ids = (
        [t.strip() for t in task_ids.split(",") if t.strip()]
        if task_ids
        else None
    )

    # Build scenario config
    scenario_config = OSWorldScenarioConfig(
        dataset=dataset,
        apps=apps,
        threat_categories=threat_categories,
        violation_types=violation_types,
        task_ids=parsed_task_ids,
        max_tasks=max_tasks,
        seed=seed,
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
        max_screenshots=max_screenshots,
        computer_timeout=computer_timeout,
    )

    # Resolve topology
    topology_template, resolved_condition = _resolve_topology(
        condition=condition,
        agents=agents,
        topology=topology,
        memory=memory,
        instructions=instructions,
        topology_file=topology_file,
    )

    # Build attack/defense configs from presets
    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.desktop.osworld.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset, condition=resolved_condition)
    if defense_preset:
        from orbit.scenarios.desktop.osworld.presets import get_defense_preset

        defenses = get_defense_preset(defense_preset)

    # Build experiment configs
    configs = build_experiment_configs_from_scenario(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=attacks,
        defenses=defenses,
    )

    # Build samples
    samples = [build_sample(config, sample_id=config.name) for config in configs]

    if not samples:
        logger.warning(
            "No samples built — check scenario config and available tasks"
        )

    # Choose solver
    from orbit.solvers.orchestrator import mas_orchestrator

    task_solver = mas_orchestrator()

    # Build task
    from orbit.scorers.security_scorer import security_scorer

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        solver=task_solver,
        scorer=[osworld_scorer(judge_model), security_scorer()],
        sandbox=("docker", _resolve_osworld_compose_file()),
    )


# ---------------------------------------------------------------------------
# Topology resolution helper (shared between osworld_safety and benchmark)
# ---------------------------------------------------------------------------


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
        raise ValueError(
            "Cannot specify both --condition and --agents."
        )
    if has_condition and has_custom_topo:
        raise ValueError(
            "Cannot specify both --condition and --topology_file."
        )
    if has_agents and has_custom_topo:
        raise ValueError(
            "Cannot specify both --agents and --topology_file."
        )
    if (has_condition or has_custom_topo) and has_sub_params:
        raise ValueError(
            "--topology, --memory, --instructions are only used with --agents."
        )
    if not has_agents and has_sub_params:
        raise ValueError(
            "--topology, --memory, --instructions require --agents."
        )

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


# ---------------------------------------------------------------------------
# OSWorld benchmark task (standard desktop tasks)
# ---------------------------------------------------------------------------


@task
def osworld_benchmark(
    corpus: str = "all",
    app: str | None = None,
    task_ids: str | None = None,
    max_tasks: int | None = None,
    seed: int | None = None,
    max_turns: int = 100,
    max_time_seconds: float = 600.0,
    max_screenshots: int = 1,
    computer_timeout: int = 180,
    # Topology
    condition: str | None = None,
    agents: str | None = None,
    topology: str | None = None,
    memory: str | None = None,
    instructions: str | None = None,
    topology_file: str = "default",
    # Attack/defense overlay for injection experiments
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    judge_model: str = "openai/gpt-4.1",
) -> Task:
    """OSWorld standard benchmark task for capability evaluation.

    Evaluates computer-use agents on benign desktop automation tasks
    from the standard OSWorld benchmark (xlang-ai/OSWorld). Tasks are
    scored on functional correctness by the OSWorld built-in evaluator.

    Can also be used for injection experiments: when ``attack_preset``
    is set, attacks are layered onto the benign tasks and both
    capability and safety scorers run.

    Examples:
        inspect eval orbit/osworld_benchmark \\
            -T corpus=small -T max_tasks=5 --model openai/gpt-4o

        inspect eval orbit/osworld_benchmark \\
            -T attack_preset=desktop_injection -T max_tasks=5 \\
            --model openai/gpt-4o

    Args:
        corpus: OSWorld corpus (``"all"`` or ``"small"``).
        app: Filter by app (comma-separated).
        task_ids: Comma-separated task IDs to include.
        max_tasks: Limit number of tasks.
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
    """
    from orbit.scenarios.desktop.osworld.config_builder import (
        build_benchmark_configs_from_scenario,
    )
    from orbit.scenarios.desktop.osworld.scorer import osworld_capability_scorer

    # Parse comma-separated filters
    apps = [a.strip() for a in app.split(",")] if app else None
    parsed_task_ids = (
        [t.strip() for t in task_ids.split(",") if t.strip()]
        if task_ids
        else None
    )

    # Map corpus to dataset value
    dataset_val = "osworld_small" if corpus == "small" else "osworld"

    scenario_config = OSWorldScenarioConfig(
        dataset=dataset_val,
        apps=apps,
        task_ids=parsed_task_ids,
        max_tasks=max_tasks,
        seed=seed,
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
        max_screenshots=max_screenshots,
        computer_timeout=computer_timeout,
        osworld_corpus=corpus if corpus in ("all", "small") else "all",
    )

    # Resolve topology
    topology_template, resolved_condition = _resolve_topology(
        condition=condition,
        agents=agents,
        topology=topology,
        memory=memory,
        instructions=instructions,
        topology_file=topology_file,
    )

    # Build attack/defense configs
    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.desktop.osworld.presets import get_attack_preset
        attacks = get_attack_preset(attack_preset, condition=resolved_condition)
    if defense_preset:
        from orbit.scenarios.desktop.osworld.presets import get_defense_preset
        defenses = get_defense_preset(defense_preset)

    # Build experiment configs
    configs = build_benchmark_configs_from_scenario(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=attacks,
        defenses=defenses,
    )

    # Build samples
    samples = [build_sample(config, sample_id=config.name) for config in configs]

    if not samples:
        logger.warning(
            "No benchmark samples built — check corpus and filters"
        )

    # Choose solver
    from orbit.solvers.orchestrator import mas_orchestrator
    task_solver = mas_orchestrator()

    # Choose scorers based on mode
    scorers = [osworld_capability_scorer()]
    if attacks:
        # Injection mode: add safety scorers
        from orbit.scorers.security_scorer import security_scorer
        scorers.append(osworld_scorer(judge_model))
        scorers.append(security_scorer())

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        solver=task_solver,
        scorer=scorers,
        sandbox=("docker", _resolve_osworld_compose_file()),
    )
