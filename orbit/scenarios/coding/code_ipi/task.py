"""
CodeIPI task entry point.

Registered with Inspect via _registry.py. Creates a Task from CodeIPI
samples with configurable topology, attacks, and defenses.
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.coding.code_ipi.config_builder import (
    build_experiment_configs_from_scenario,
    default_topology_template,
)
from orbit.scenarios.coding.code_ipi.configs import CodeIPIScenarioConfig
from orbit.scenarios.coding.code_ipi.scorer import code_ipi_scorer

logger = logging.getLogger(__name__)


@task
def code_ipi(
    vectors: str | None = None,
    severities: str | None = None,
    defense_prompt: str = "true",
    max_tasks: int | None = None,
    seed: int | None = None,
    include_benign: str = "true",
    condition: str = "single_agent",
    topology: str | None = None,
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    max_turns: int = 30,
    max_time: float = 300.0,
) -> Task:
    """Indirect Prompt Injection benchmark for coding agents.

    Evaluates whether coding agents execute hidden malicious instructions
    embedded in software engineering artifacts (issues, code comments,
    README files, config files) while performing legitimate bug-fixing tasks.

    Args:
        vectors: Comma-separated injection vectors to include.
            Options: issue_text, code_comment, readme, config_file.
        severities: Comma-separated severity tiers.
            Options: S1, S2, S3.
        defense_prompt: Whether to include anti-injection warning.
            "true" (default) adds explicit instruction. "false" uses neutral.
        max_tasks: Maximum number of samples to include.
        seed: Random seed for reproducible sampling.
        include_benign: Whether to include benign control samples.
        condition: Named topology condition.
            Options: single_agent, star_single_executor, star_specialists,
            mesh_round_robin, mesh_delegation.
        topology: Path to a YAML topology file (overrides condition).
        attack_preset: Named attack preset to layer on top of sample injection.
        defense_preset: Named defense preset.
            Options: sysprompt, llm_monitor, guardian, dual_llm, full.
        max_turns: Maximum turns per agent.
        max_time: Maximum wall-clock time in seconds.
    """
    vector_list = [v.strip() for v in vectors.split(",")] if vectors else None
    severity_list = [s.strip() for s in severities.split(",")] if severities else None
    use_defense = str(defense_prompt).lower() != "false"
    use_benign = str(include_benign).lower() != "false"

    scenario_config = CodeIPIScenarioConfig(
        vectors=vector_list,
        severities=severity_list,
        max_tasks=max_tasks,
        seed=seed,
        include_benign=use_benign,
        defense_prompt=use_defense,
        max_turns=max_turns,
        max_time_seconds=max_time,
    )

    # Resolve topology
    if topology:
        import yaml

        from orbit.configs.setup import SetupConfig

        with open(topology) as f:
            topo_data = yaml.safe_load(f)
        topology_template = SetupConfig(**topo_data)
    elif condition != "single_agent":
        from orbit.scenarios.coding.code_ipi.condition_presets import (
            get_condition_setup,
        )

        topology_template = get_condition_setup(condition)
    else:
        topology_template = default_topology_template(
            defense_prompt=use_defense
        )

    # Resolve presets
    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.coding.code_ipi.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset)
    if defense_preset:
        from orbit.scenarios.coding.code_ipi.presets import get_defense_preset

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

    from orbit.scorers.security_scorer import security_scorer
    from orbit.solvers.orchestrator import mas_orchestrator

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        solver=mas_orchestrator(),
        scorer=[code_ipi_scorer(), security_scorer()],
    )
