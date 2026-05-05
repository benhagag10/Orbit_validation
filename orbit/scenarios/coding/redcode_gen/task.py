"""
RedCode-Gen task entry point.

Registered with Inspect via ``_registry.py``. Creates a Task from RedCode-Gen
malware code generation prompts with configurable topology, attacks, and
defenses.
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.coding.redcode_gen.config_builder import (
    build_experiment_configs_from_scenario,
    default_topology_template,
)
from orbit.scenarios.coding.redcode_gen.configs import RedCodeGenScenarioConfig
from orbit.scenarios.coding.redcode_gen.scorer import redcode_gen_scorer

logger = logging.getLogger(__name__)


@task
def redcode_gen(
    categories: str | None = None,
    task_ids: str | None = None,
    max_tasks: int | None = None,
    seed: int | None = None,
    judge_model: str = "openai/gpt-4.1",
    max_turns: int = 5,
    max_time_seconds: float = 120.0,
    topology_file: str = "default",
    condition: str | None = None,
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    data_path: str | None = None,
    orchestrator: str = "v1",
) -> Task:
    """RedCode-Gen malware code generation safety benchmark task.

    Evaluates whether coding agents will generate malicious code when given
    function signatures describing malware behavior.

    Args:
        categories: Comma-separated malware categories to include
            (e.g. ``"spyware,ransomware"``). Default: all 8 categories.
        task_ids: Comma-separated task IDs to include
            (e.g. ``"spyware_1_keylogger,ddos_5_syn_flood"``).
        max_tasks: Maximum number of tasks to evaluate.
        seed: Random seed for deterministic sampling when ``max_tasks`` is set.
        judge_model: Model for the LLM judge scorer.
        max_turns: Maximum turns per task.
        max_time_seconds: Maximum wall-clock time per task.
        topology_file: Custom topology YAML path or ``"default"``.
        condition: Named ICML condition (e.g. ``"single_agent"``,
            ``"star_specialist"``). Overrides ``topology_file``.
        attack_preset: Attack preset name (e.g. ``"jailbreak"``).
        defense_preset: Defense preset name (e.g. ``"prompt_vaccination"``).
        data_path: Path to RedCode-Gen dataset directory. Default: bundled data.
    """
    # Parse comma-separated params
    category_list = (
        [c.strip() for c in categories.split(",")]
        if categories
        else None
    )
    task_id_list = (
        [t.strip() for t in task_ids.split(",")]
        if task_ids
        else None
    )

    # Build scenario config
    scenario_config = RedCodeGenScenarioConfig(
        data_path=data_path,
        categories=category_list,
        task_ids=task_id_list,
        max_tasks=max_tasks,
        seed=seed,
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
    )

    # Resolve topology: condition > topology_file > default
    if condition:
        from orbit.scenarios.coding.redcode_gen.condition_presets import (
            get_condition_setup,
        )
        topology_template = get_condition_setup(condition)
    elif topology_file == "default":
        topology_template = default_topology_template()
    else:
        import yaml
        from orbit.configs.setup import SetupConfig

        with open(topology_file) as f:
            topo_data = yaml.safe_load(f)
        topology_template = SetupConfig(**topo_data)

    # Build attack/defense configs from presets
    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.coding.redcode_gen.presets import get_attack_preset
        attacks = get_attack_preset(attack_preset)
    if defense_preset:
        from orbit.scenarios.coding.redcode_gen.presets import get_defense_preset
        defenses = get_defense_preset(defense_preset)

    # Build experiment configs
    configs = build_experiment_configs_from_scenario(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=attacks,
        defenses=defenses,
    )

    # Build samples
    samples = []
    for config in configs:
        sample = build_sample(config, sample_id=config.name)
        samples.append(sample)

    if not samples:
        logger.warning("No samples built — check scenario config and filters")

    # Build task
    from pathlib import Path

    from orbit.scorers.security_scorer import security_scorer
    from orbit.solvers.orchestrator import mas_orchestrator
    from orbit.solvers.orchestrator_v2 import mas_orchestrator_v2

    sandbox_compose = str(
        Path(__file__).parent / "sandbox" / "compose.yaml"
    )

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        solver=mas_orchestrator_v2() if orchestrator == "v2" else mas_orchestrator(),
        scorer=[redcode_gen_scorer(judge_model), security_scorer()],
        sandbox=("docker", sandbox_compose),
    )
