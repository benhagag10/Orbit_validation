"""
CodeIPI task entry point.

Registered with Inspect via _registry.py. Creates a Task from CodeIPI
samples with configurable topology, attacks, and defenses.

Construction is unified through the scenario-agnostic builder
(``orbit.tasks.builder.build_scenario_task``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from inspect_ai import Task, task

from orbit.scenarios.coding.code_ipi.config_builder import (
    build_experiment_configs,
    default_topology_template,
)
from orbit.scenarios.coding.code_ipi.configs import CodeIPIScenarioConfig
from orbit.scenarios.registry import ScenarioPlugin, register_scenario
from orbit.tasks.builder import build_scenario_task, split_csv

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer

    from orbit.configs.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


def _scenario_config(config: ExperimentConfig) -> CodeIPIScenarioConfig:
    meta = config.metadata or {}
    if "code_ipi_scenario_config" in meta:
        return CodeIPIScenarioConfig(**meta["code_ipi_scenario_config"])
    return CodeIPIScenarioConfig(
        vectors=split_csv(meta.get("code_ipi_vectors")),
        severities=split_csv(meta.get("code_ipi_severities")),
        seed=meta.get("code_ipi_seed"),
        include_benign=meta.get("code_ipi_include_benign", True),
        inline_defense_warning=meta.get("code_ipi_inline_defense_warning", True),
        max_turns=config.scheduler.max_turns,
        max_time_seconds=config.scheduler.max_time_seconds,
    )


def _code_ipi_expand(config: ExperimentConfig) -> list[ExperimentConfig]:
    scenario_config = _scenario_config(config)
    if config.setup.agents:
        topology_template = config.setup
    else:
        topology_template = default_topology_template(
            inline_defense_warning=scenario_config.inline_defense_warning
        )
    configs = build_experiment_configs(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=list(config.attacks) or None,
        defenses=list(config.defenses) or None,
    )
    if not configs:
        logger.warning("No samples built — check scenario config and filters")
    return configs


def _code_ipi_scorers(config: ExperimentConfig) -> list[Scorer]:
    from orbit.scenarios.coding.code_ipi.scorer import code_ipi_scorer
    from orbit.scorers.security_scorer import security_scorer

    return [code_ipi_scorer(), security_scorer()]


CODE_IPI_PLUGIN = register_scenario(
    ScenarioPlugin(
        name="code_ipi",
        expand=_code_ipi_expand,
        build_setup=lambda config: None,
        build_solver=None,
        build_scorers=_code_ipi_scorers,
        build_sandbox=lambda config: None,  # per-sample compose file
    )
)


@task
def code_ipi(
    vectors: str | None = None,
    severities: str | None = None,
    inline_defense_warning: bool = True,
    seed: int | None = None,
    include_benign: bool = True,
    condition: str = "single_agent",
    topology: str | None = None,
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    max_turns: int = 30,
    max_time_seconds: float = 300.0,
    orchestrator: str = "v1",
) -> Task:
    """Indirect Prompt Injection benchmark for coding agents.

    Evaluates whether coding agents execute hidden malicious instructions
    embedded in software engineering artifacts (issues, code comments,
    README files, config files) while performing legitimate bug-fixing tasks.

    Args:
        vectors: Comma-separated injection vectors to include.
            Options: issue_text, code_comment, readme, config_file.
        severities: Comma-separated severity tiers. Options: S1, S2, S3.
        inline_defense_warning: Whether to include the anti-injection warning
            in the agent system prompt. ``True`` (default) adds explicit
            instruction; ``False`` uses a neutral prompt.
        seed: Random seed for reproducible sampling.
        include_benign: Whether to include benign control samples.
        condition: Named topology condition.
        topology: Path to a YAML topology file (overrides condition).
        attack_preset: Named attack preset to layer on top of sample injection.
        defense_preset: Named defense preset.
        max_turns: Maximum turns per agent.
        max_time_seconds: Maximum wall-clock time in seconds.
        orchestrator: MAS orchestrator variant (``"v1"`` or ``"v2"``).
    """
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.metrics import MetricsConfig
    from orbit.configs.scenario import ScenarioConfig
    from orbit.configs.scheduler import SchedulerConfig

    scenario_config = CodeIPIScenarioConfig(
        vectors=split_csv(vectors),
        severities=split_csv(severities),
        seed=seed,
        include_benign=include_benign,
        inline_defense_warning=inline_defense_warning,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
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
            inline_defense_warning=inline_defense_warning
        )

    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.coding.code_ipi.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset)
    if defense_preset:
        from orbit.scenarios.coding.code_ipi.presets import get_defense_preset

        defenses = get_defense_preset(defense_preset)

    config = ExperimentConfig(
        name="code_ipi",
        description="CodeIPI indirect prompt injection benchmark",
        setup=topology_template,
        scenario=ScenarioConfig(name="code_ipi", description="CodeIPI coding injection"),
        attacks=attacks or [],
        defenses=defenses or [],
        scheduler=SchedulerConfig(
            max_turns=max_turns,
            max_time_seconds=max_time_seconds,
            halt_on_convergence=False,
        ),
        metrics=MetricsConfig(),
        metadata={"code_ipi_scenario_config": scenario_config.model_dump()},
    )
    return build_scenario_task(config, CODE_IPI_PLUGIN, orchestrator=orchestrator)
