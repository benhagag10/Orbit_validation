"""BigCodeBench task entry point.

Registered with Inspect via ``_registry.py``. Builds a Task from the
BigCodeBench dataset using the InspectMAS orchestrator, with the same
topology and preset surface as the ``redcode_gen`` scenario.
"""

from __future__ import annotations

import logging
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.coding.bigcodebench.config_builder import (
    build_experiment_configs_from_scenario,
    default_topology_template,
)
from orbit.scenarios.coding.bigcodebench.configs import (
    BigCodeBenchScenarioConfig,
)
from orbit.scenarios.coding.bigcodebench.scorer import bigcodebench_scorer

logger = logging.getLogger(__name__)


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

    # Resolve topology: condition > topology_file > default
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
        from orbit.scenarios.coding.bigcodebench.presets import (
            get_attack_preset,
        )
        attacks = get_attack_preset(attack_preset)
    if defense_preset:
        from orbit.scenarios.coding.bigcodebench.presets import (
            get_defense_preset,
        )
        defenses = get_defense_preset(defense_preset)

    configs = build_experiment_configs_from_scenario(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=attacks,
        defenses=defenses,
    )

    samples = [build_sample(c, sample_id=c.name) for c in configs]

    if not samples:
        logger.warning("No samples built — check scenario config and filters")

    from orbit.scorers.security_scorer import security_scorer
    from orbit.solvers.orchestrator import mas_orchestrator

    sandbox_compose = str(Path(__file__).parent / "sandbox" / "compose.yaml")

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        solver=mas_orchestrator(),
        scorer=[bigcodebench_scorer(), security_scorer()],
        sandbox=("docker", sandbox_compose),
    )
