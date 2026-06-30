"""BigCodeBench task entry point.

Registered with Inspect via ``_registry.py``. Builds a Task from the
BigCodeBench dataset using the InspectMAS orchestrator, with the same
topology and preset surface as the ``redcode_gen`` scenario.

Construction is unified through the scenario-agnostic builder
(``orbit.tasks.builder.build_scenario_task``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from inspect_ai import Task, task

from orbit.scenarios.coding.bigcodebench.config_builder import (
    build_experiment_configs,
)
from orbit.scenarios.coding.bigcodebench.configs import BigCodeBenchScenarioConfig
from orbit.scenarios.registry import ScenarioPlugin, register_scenario
from orbit.tasks.builder import build_scenario_task

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer

    from orbit.configs.experiment import ExperimentConfig

logger = logging.getLogger(__name__)

_SANDBOX_COMPOSE = str(Path(__file__).parent / "sandbox" / "compose.yaml")


def _scenario_config(config: ExperimentConfig) -> BigCodeBenchScenarioConfig:
    meta = config.metadata or {}
    if "bigcodebench_scenario_config" in meta:
        return BigCodeBenchScenarioConfig(**meta["bigcodebench_scenario_config"])
    subset = meta.get("bigcodebench_subset")
    if isinstance(subset, str):
        subset = [int(s.strip()) for s in subset.split(",") if s.strip()]
    return BigCodeBenchScenarioConfig(
        version=meta.get("bigcodebench_version", "v0.1.2"),
        subset=subset,
        max_tasks=meta.get("bigcodebench_max_tasks"),
        seed=meta.get("bigcodebench_seed"),
        max_turns=config.scheduler.max_turns,
        max_time_seconds=config.scheduler.max_time_seconds,
    )


def _bigcodebench_expand(config: ExperimentConfig) -> list[ExperimentConfig]:
    configs = build_experiment_configs(
        scenario_config=_scenario_config(config),
        topology_template=config.setup if config.setup.agents else None,
        attacks=list(config.attacks) or None,
        defenses=list(config.defenses) or None,
    )
    if not configs:
        logger.warning("No samples built — check scenario config and filters")
    return configs


def _bigcodebench_scorers(config: ExperimentConfig) -> list[Scorer]:
    from orbit.scenarios.coding.bigcodebench.scorer import bigcodebench_scorer
    from orbit.scorers.security_scorer import security_scorer

    return [bigcodebench_scorer(), security_scorer()]


BIGCODEBENCH_PLUGIN = register_scenario(
    ScenarioPlugin(
        name="bigcodebench",
        expand=_bigcodebench_expand,
        build_setup=lambda config: None,
        build_solver=None,
        build_scorers=_bigcodebench_scorers,
        build_sandbox=lambda config: ("docker", _SANDBOX_COMPOSE),
    )
)


@task
def bigcodebench(
    subset: str | None = None,
    max_tasks: int | None = None,
    seed: int | None = None,
    max_turns: int = 5,
    max_time_seconds: float = 300.0,
    topology_file: str = "default",
    condition: str | None = None,
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    version: str = "v0.1.2",
) -> Task:
    """BigCodeBench benign code generation baseline.

    Evaluates coding agents on legitimate function-completion tasks with
    real unit-test execution. Serves as the benign counterpart to
    ``redcode_gen``: same input format, same topology surface, but
    measures pass@1 on normal coding workloads instead of refusal on
    malware requests.

    Args:
        subset: Comma-separated list of BigCodeBench task ids (integers)
            to include, e.g. ``"0,1,2"``.
        max_tasks: Maximum number of tasks to evaluate.
        seed: Random seed for deterministic sampling when ``max_tasks``
            is set.
        max_turns: Maximum turns per task.
        max_time_seconds: Maximum wall-clock time per task.
        topology_file: Custom topology YAML path or ``"default"``.
        condition: Named ICML condition (e.g. ``"single_agent"``,
            ``"star_specialist"``). Overrides ``topology_file``.
        attack_preset: Attack preset name (e.g. ``"jailbreak"``).
        defense_preset: Defense preset name (e.g. ``"prompt_vaccination"``).
        version: HuggingFace dataset version (e.g. ``"v0.1.2"``).
    """
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.metrics import MetricsConfig
    from orbit.configs.scenario import ScenarioConfig
    from orbit.configs.scheduler import SchedulerConfig
    from orbit.scenarios.coding.bigcodebench.config_builder import default_topology_template

    subset_ids: list[int] | None = None
    if subset:
        subset_ids = [int(s.strip()) for s in subset.split(",") if s.strip()]

    scenario_config = BigCodeBenchScenarioConfig(
        version=version,
        subset=subset_ids,
        max_tasks=max_tasks,
        seed=seed,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
    )

    if condition:
        from orbit.scenarios.coding.bigcodebench.condition_presets import (
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

    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.coding.bigcodebench.presets import get_attack_preset
        attacks = get_attack_preset(attack_preset)
    if defense_preset:
        from orbit.scenarios.coding.bigcodebench.presets import get_defense_preset
        defenses = get_defense_preset(defense_preset)

    config = ExperimentConfig(
        name="bigcodebench",
        description="BigCodeBench benign code generation baseline",
        setup=topology_template,
        scenario=ScenarioConfig(name="bigcodebench", description="BigCodeBench coding baseline"),
        attacks=attacks or [],
        defenses=defenses or [],
        scheduler=SchedulerConfig(
            max_turns=max_turns,
            max_time_seconds=max_time_seconds,
            halt_on_convergence=False,
        ),
        metrics=MetricsConfig(),
        metadata={"bigcodebench_scenario_config": scenario_config.model_dump()},
    )
    return build_scenario_task(config, BIGCODEBENCH_PLUGIN)
