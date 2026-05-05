"""Config builder for AgentHarm experiments.

Turns ``AgentHarmBehavior`` + scenario options into one
``ExperimentConfig`` per behavior, with per-sample topology and tools.
"""

from __future__ import annotations

import logging

from orbit.configs.attack import AttackConfig
from orbit.configs.defense import DefenseConfig
from orbit.configs.execution import (
    AgentGroup,
    ExecutionConfig,
    ObservationConfig,
)
from orbit.configs.experiment import ExperimentConfig
from orbit.configs.metrics import MetricsConfig
from orbit.configs.scenario import ScenarioConfig
from orbit.configs.scheduler import SchedulerConfig
from orbit.configs.setup import SetupConfig
from orbit.scenarios.agentharm.condition_presets import get_condition_setup
from orbit.scenarios.agentharm.configs import (
    AgentHarmBehavior,
    AgentHarmScenarioConfig,
)
from orbit.scenarios.agentharm.dataset_builder import load_behaviors

logger = logging.getLogger(__name__)


def _build_execution_config(
    topology: SetupConfig,
    max_turns: int,
) -> tuple[ExecutionConfig, int]:
    """Wire mesh conditions into Path C (AgentScheduler).

    Mirrors the pattern used by other scenarios: topology_type in
    {``mesh_round_robin``, ``mesh_delegation``} routes through an
    ``AgentGroup`` with round-robin scheduling and ``observation=summary``.
    """
    topology_type = topology.properties.get("topology_type", "")

    if topology_type in {"mesh_round_robin", "mesh_delegation"}:
        agent_names = [a.name for a in topology.agents]
        execution = ExecutionConfig(
            scheduling_mode="round_robin",
            agent_groups=[AgentGroup(
                name="mesh_peers",
                agents=agent_names,
                goal="",
            )],
            observation=ObservationConfig(mode="summary"),
        )
        effective_max_turns = max_turns * len(agent_names)
        return execution, effective_max_turns

    return ExecutionConfig(), max_turns


def build_experiment_config(
    behavior: AgentHarmBehavior,
    scenario_config: AgentHarmScenarioConfig,
    *,
    condition: str = "single_agent",
    attacks: list[AttackConfig] | None = None,
    defenses: list[DefenseConfig] | None = None,
) -> ExperimentConfig:
    """Build one ``ExperimentConfig`` for a single AgentHarm behavior."""
    topology = get_condition_setup(
        condition,
        behavior,
        split=scenario_config.split,
        tool_mode=scenario_config.tool_mode,
        n_irrelevant=scenario_config.n_irrelevant_tools,
        seed=scenario_config.seed,
        system_prompt_variant=scenario_config.system_prompt_variant,
        prompt_technique=scenario_config.prompt_technique,
    )

    execution, effective_max_turns = _build_execution_config(
        topology, scenario_config.max_turns,
    )

    scenario_meta = {
        "agentharm_behavior_id": behavior.id,
        "agentharm_id_original": behavior.id_original,
        "agentharm_name": behavior.name,
        "agentharm_category": behavior.category,
        "agentharm_split": scenario_config.split,
        "agentharm_partition": scenario_config.partition,
        "agentharm_target_functions": list(behavior.target_functions),
        "agentharm_grading_function": behavior.grading_function or "",
        "agentharm_detailed_prompt": behavior.detailed_prompt,
        "agentharm_hint_included": behavior.hint_included,
        "agentharm_judge_model": scenario_config.judge_model,
        "agentharm_condition": condition,
        "agentharm_tool_mode": scenario_config.tool_mode,
    }

    return ExperimentConfig(
        name=f"agentharm_{scenario_config.split}_{behavior.id}_{condition}",
        description=(
            f"AgentHarm ({scenario_config.split}) {behavior.category}: "
            f"{behavior.name} [{condition}]"
        ),
        setup=topology,
        scenario=ScenarioConfig(
            name="agentharm",
            description=(
                f"AgentHarm {scenario_config.split} tool-use benchmark "
                f"(category: {behavior.category})"
            ),
        ),
        attacks=attacks or [],
        defenses=defenses or [],
        execution=execution,
        scheduler=SchedulerConfig(
            max_turns=effective_max_turns,
            max_time_seconds=scenario_config.max_time_seconds,
            halt_on_convergence=False,
        ),
        metrics=MetricsConfig(),
        user_prompt=behavior.prompt,
        metadata=scenario_meta,
    )


def build_experiment_configs_from_scenario(
    scenario_config: AgentHarmScenarioConfig | None = None,
    *,
    condition: str = "single_agent",
    attacks: list[AttackConfig] | None = None,
    defenses: list[DefenseConfig] | None = None,
    behaviors: list[AgentHarmBehavior] | None = None,
) -> list[ExperimentConfig]:
    """Build one ``ExperimentConfig`` per filtered behavior."""
    if scenario_config is None:
        scenario_config = AgentHarmScenarioConfig()
    if behaviors is None:
        behaviors = load_behaviors(scenario_config)

    configs: list[ExperimentConfig] = []
    for b in behaviors:
        configs.append(
            build_experiment_config(
                behavior=b,
                scenario_config=scenario_config,
                condition=condition,
                attacks=attacks,
                defenses=defenses,
            )
        )

    logger.info(
        "Built %d AgentHarm experiment configs from %d behaviors "
        "(split=%s, condition=%s)",
        len(configs), len(behaviors), scenario_config.split, condition,
    )
    return configs
