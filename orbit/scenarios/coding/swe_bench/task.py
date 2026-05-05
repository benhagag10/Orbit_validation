"""
SWE-Bench multi-issue task entry point.

Registered with Inspect via _registry.py. Creates a Task from SWE-Bench
multi-issue groups with configurable topology, attacks, and defenses.
"""

from __future__ import annotations

import logging
import platform
from importlib.util import find_spec
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.coding.swe_bench.config_builder import (
    build_experiment_configs_from_scenario,
    default_topology_template,
)
from inspect_ai.solver import chain

from orbit.scenarios.coding.swe_bench.configs import SWEBenchScenarioConfig
from orbit.scenarios.coding.swe_bench.network_solver import (
    swe_bench_disconnect_network,
    swe_bench_pre_agent_validation,
)
from orbit.scenarios.coding.swe_bench.scorer import swe_bench_multi_issue_scorer

logger = logging.getLogger(__name__)


def _check_swe_bench_prerequisites() -> None:
    """Verify SWE-Bench prerequisites before running the task.

    Checks that swebench is importable and Docker is reachable. Logs a
    warning on ARM hosts about Rosetta requirements.

    Raises:
        RuntimeError: If swebench is not installed or Docker is unreachable.
    """
    if find_spec("swebench") is None:
        raise RuntimeError(
            "swebench is not installed.\n"
            "Install with: uv sync --extra swebench\n"
            "See docs/scenarios.md for full setup instructions."
        )

    machine = platform.machine()
    if machine in ("arm64", "aarch64"):
        logger.warning(
            "Running on ARM (%s). SWE-Bench images are primarily x86_64 — "
            "you may need Docker Desktop with Rosetta emulation enabled. "
            "See docs/scenarios.md for Apple Silicon setup instructions.",
            machine,
        )

    try:
        import docker

        client = docker.from_env()
        client.ping()
    except ImportError:
        raise RuntimeError(
            "docker Python package is not installed.\n"
            "Install with: uv sync --extra swebench"
        )
    except Exception as e:
        raise RuntimeError(
            f"Docker is not reachable: {e}\n"
            "Make sure Docker Desktop is running.\n"
            "On Apple Silicon, enable Rosetta in Docker Desktop → Settings → General → "
            "'Use Rosetta for x86_64/amd64 emulation on Apple Silicon'."
        )


@task
def swe_bench_multi_issue(
    scenario_config_path: str | None = None,
    num_issues: int = 2,
    seed: int | None = None,
    repos: str | None = None,
    instance_ids: str | None = None,
    instance_set: str | None = None,
    model: str | None = None,
    mode: str = "round_robin",
    topology: str | SetupConfig = "star",
    topology_file: str = "default",
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    max_turns: int = 30,
    max_time: float = 900.0,
    max_groups: int | None = None,
    force_arch: str = "",
    condition: str | None = None,
    agents: str | None = None,
    memory: str = "none",
    instructions: str = "detailed",
) -> Task:
    """SWE-Bench multi-issue security benchmark task.

    Args:
        topology: Topology style (star, round_robin, delegation).
        topology_file: Custom topology YAML path or ``"default"``.
        condition: Condition preset name (e.g. ``"star_specialist"``,
            ``"memory_full"``). When set, overrides topology with the
            condition's setup.
        agents: Human-readable agent type (single, batch, specialist, step).
            Cannot be combined with ``condition``.
        memory: Memory sharing level (none, own_actions, own_reasoning,
            shared_actions, full). Only used with ``agents``.
        instructions: Instruction detail level (detailed, relaxed).
            Only used with ``agents``.
    """
    from orbit.configs.setup import SetupConfig

    # ---- Resolve human-readable params to condition name ----
    if agents is not None and condition is not None:
        raise ValueError(
            "Cannot specify both --condition and --agents. Use one or the other."
        )
    if agents is not None:
        from orbit.scenarios.coding.swe_bench.condition_presets import resolve_condition

        condition = resolve_condition(
            agents=agents, topology=topology,
            memory=memory, instructions=instructions,
        )

    _check_swe_bench_prerequisites()

    # Build scenario config
    if scenario_config_path:
        import yaml

        with open(scenario_config_path) as f:
            data = yaml.safe_load(f)
        scenario_config = SWEBenchScenarioConfig(**data)
    else:
        repo_list = [r.strip() for r in repos.split(",")] if repos else None

        # Resolve instance IDs from various sources
        id_list: list[str] | None = None
        if instance_ids:
            id_list = [i.strip() for i in instance_ids.split(",")]
        elif instance_set:
            import json

            set_path = Path(instance_set)
            if not set_path.is_absolute():
                set_path = (
                    Path(__file__).parent / "data" / instance_set
                )
            with open(set_path) as f:
                id_list = json.load(f)["instance_ids"]
            logger.info("Loaded %d instance IDs from %s", len(id_list), set_path)

        scenario_config = SWEBenchScenarioConfig(
            num_issues=num_issues,
            seed=seed,
            repos=repo_list,
            instance_ids=id_list,
            execution_mode=mode,
            max_turns_per_agent=max_turns,
            max_time_seconds=max_time,
            max_groups=max_groups,
            force_arch=force_arch,
        )

    # Load topology template
    if condition is not None:
        from orbit.scenarios.coding.swe_bench.condition_presets import (
            get_condition_setup,
        )

        topology_template = get_condition_setup(condition)
    elif isinstance(topology, SetupConfig):
        topology_template = topology
    elif topology_file == "default":
        topology_template = default_topology_template()
    else:
        # Load from YAML file
        import yaml

        with open(topology_file) as f:
            topo_data = yaml.safe_load(f)
        topology_template = SetupConfig(**topo_data)

    # Build attack/defense configs from presets
    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.coding.swe_bench.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset)
    if defense_preset:
        from orbit.scenarios.coding.swe_bench.presets import get_defense_preset

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
        logger.warning("No samples built — check scenario config and available groups")

    # Build scorer pipeline
    from orbit.scorers.security_scorer import security_scorer
    from orbit.solvers.orchestrator import mas_orchestrator

    scorers = [swe_bench_multi_issue_scorer()]

    has_codebase_injection = attacks and any(
        a.attack_type == "codebase_injection" for a in attacks
    )
    if has_codebase_injection:
        from orbit.scenarios.coding.swe_bench.injection_scorer import (
            codebase_injection_scorer,
        )

        scorers.append(codebase_injection_scorer())

    has_compromised_or_collusion = attacks and any(
        a.attack_type in ("compromised_agent", "collusion")
        and a.properties.get("has_covert_report_tool")
        for a in attacks
    )
    if has_compromised_or_collusion:
        from orbit.scenarios.coding.swe_bench.attacks.vulnerability_scorer import (
            vulnerability_scorer,
        )

        scorers.append(vulnerability_scorer())

    scorers.append(security_scorer())

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        setup=chain(
            swe_bench_disconnect_network(),
            swe_bench_pre_agent_validation(),
        ),
        solver=mas_orchestrator(),
        scorer=scorers,
    )
