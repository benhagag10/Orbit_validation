"""AgentHarm task entry point.

Registered via :mod:`orbit._registry` so ``inspect eval orbit/agentharm``
is discoverable from the Inspect CLI.

Construction is unified through the scenario-agnostic builder
(``orbit.tasks.builder.build_scenario_task``). AgentHarm's topology is derived
per behavior from a named ``condition`` (each behavior gets its own mocked tool
set), so the ``expand`` hook reads the condition and the task-selection config
from ``config.metadata`` rather than from ``config.setup``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from inspect_ai import Task, task

from orbit.scenarios.agentharm.config_builder import (
    build_experiment_configs,
)
from orbit.scenarios.agentharm.configs import AgentHarmScenarioConfig
from orbit.scenarios.registry import ScenarioPlugin, register_scenario
from orbit.tasks.builder import build_scenario_task

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer

    from orbit.configs.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


def _scenario_config(config: ExperimentConfig) -> AgentHarmScenarioConfig:
    meta = config.metadata or {}
    if "agentharm_scenario_config" in meta:
        return AgentHarmScenarioConfig(**meta["agentharm_scenario_config"])
    # Bare config (e.g. a hand-written orbit-run YAML): use defaults but honour
    # the experiment's turn/time budget.
    return AgentHarmScenarioConfig(
        max_turns=config.scheduler.max_turns,
        max_time_seconds=config.scheduler.max_time_seconds,
    )


def _agentharm_expand(config: ExperimentConfig) -> list[ExperimentConfig]:
    meta = config.metadata or {}
    condition = meta.get("agentharm_condition", "single_agent")
    # Topology is derived per-behavior from the named condition (not config.setup),
    # but the experiment-level attacks/defenses MUST still reach every behavior's
    # config — otherwise an `orbit run` YAML that declares them is silently a
    # benign run (issue #9). The builder enforces this; we satisfy it here.
    configs = build_experiment_configs(
        scenario_config=_scenario_config(config),
        condition=condition,
        attacks=list(config.attacks) or None,
        defenses=list(config.defenses) or None,
    )
    if not configs:
        logger.warning("AgentHarm produced 0 samples — check split/partition/filters")
    return configs


def _agentharm_scorers(config: ExperimentConfig) -> list[Scorer]:
    from orbit.scenarios.agentharm.scorer import agentharm_scorer
    from orbit.scorers.security_scorer import security_scorer

    return [agentharm_scorer(_scenario_config(config).judge_model), security_scorer()]


AGENTHARM_PLUGIN = register_scenario(
    ScenarioPlugin(
        name="agentharm",
        expand=_agentharm_expand,
        build_setup=lambda config: None,
        build_solver=None,  # default mas_orchestrator (v1/v2)
        build_scorers=_agentharm_scorers,
        build_sandbox=lambda config: None,  # tools are mocked; no sandbox
    )
)


@task
def agentharm(
    split: str = "harmful",
    partition: str = "test_public",
    condition: str = "single_agent",
    categories: str | None = None,
    task_ids: str | None = None,
    detailed_prompt: bool | None = None,
    hint_included: bool | None = None,
    seed: int = 0,
    judge_model: str = "openai/gpt-4.1",
    n_irrelevant_tools: int = 0,
    tool_mode: str = "filtered",
    prompt_technique: str = "standard",
    system_prompt_variant: str = "default",
    max_turns: int = 10,
    max_time_seconds: float = 300.0,
    data_path: str | None = None,
    orchestrator: str = "v1",
) -> Task:
    """AgentHarm harmful tool-use safety benchmark.

    Orbit port of AgentHarm (Andriushchenko et al., ICLR 2025). Every
    tool is mocked — no sandbox, no network, no filesystem side effects.

    Args:
        split: ``harmful`` (default), ``benign``, or ``chat``.
        partition: ``test_public`` (default, 176 samples) or ``validation``.
        condition: Topology preset — see
            :func:`orbit.scenarios.agentharm.condition_presets.list_conditions`.
        categories: Comma-separated harm-category filter.
        task_ids: Comma-separated behavior-ID filter.
        detailed_prompt: Keep only behaviors with matching ``detailed_prompt``.
        hint_included: Keep only behaviors with matching ``hint_included``.
        seed: Seed for deterministic irrelevant-tool selection.
        judge_model: Model used for the refusal judge and graders.
        n_irrelevant_tools: Pad each agent's tool list with distractor tools.
        tool_mode: ``filtered`` (default) or ``full_domain``.
        prompt_technique: ``standard`` / ``cot`` / ``react`` / ``refusal``.
        system_prompt_variant: ``default`` or ``llama-3.1``.
        max_turns: Maximum agent turns per sample.
        max_time_seconds: Maximum wall-clock time per sample.
        data_path: Override for the pinned ``data/`` directory.
    """
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.metrics import MetricsConfig
    from orbit.configs.scenario import ScenarioConfig
    from orbit.configs.scheduler import SchedulerConfig

    category_list = (
        tuple(c.strip() for c in categories.split(",") if c.strip())
        if categories else None
    )
    id_list = (
        tuple(b.strip() for b in task_ids.split(",") if b.strip())
        if task_ids else None
    )

    scenario_config = AgentHarmScenarioConfig(
        data_path=data_path,
        split=split,  # type: ignore[arg-type]
        partition=partition,  # type: ignore[arg-type]
        categories=category_list,
        task_ids=id_list,
        detailed_prompt=detailed_prompt,
        hint_included=hint_included,
        seed=seed,
        judge_model=judge_model,
        n_irrelevant_tools=n_irrelevant_tools,
        tool_mode=tool_mode,  # type: ignore[arg-type]
        prompt_technique=prompt_technique,  # type: ignore[arg-type]
        system_prompt_variant=system_prompt_variant,  # type: ignore[arg-type]
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
    )

    config = ExperimentConfig(
        name="agentharm",
        description="AgentHarm harmful tool-use safety benchmark",
        scenario=ScenarioConfig(name="agentharm", description="AgentHarm tool-use safety"),
        scheduler=SchedulerConfig(
            max_turns=max_turns,
            max_time_seconds=max_time_seconds,
            halt_on_convergence=False,
        ),
        metrics=MetricsConfig(),
        metadata={
            "agentharm_scenario_config": scenario_config.model_dump(),
            "agentharm_condition": condition,
        },
    )
    return build_scenario_task(config, AGENTHARM_PLUGIN, orchestrator=orchestrator)
