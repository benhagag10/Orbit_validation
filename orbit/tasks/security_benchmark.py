"""
Main @task entry point for Orbit security benchmarks.

Composes all components into a standard Inspect Task:
    - Dataset: built from ExperimentConfig via sample_factory
    - Setup: mas_environment_setup solver (pre-flight validation + env init)
    - Solver: mas_orchestrator (builds agents, runs scheduler, manages attacks/defenses)
    - Scorer: security_scorer + integrity_scorer

Interaction with Inspect:
    - Decorated with @task to register with Inspect's task registry
    - Returns Task(...) with dataset, setup, solver, scorer
    - Discovered via entry-point: orbit._registry imports this
    - Can be run with: inspect eval orbit/security_benchmark --task-config ...
    - Or via CLI: orbit run experiment.yaml
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.baselines.baselines import apply_baseline
from orbit.configs.baseline import BaselineMode
from orbit.configs.experiment import ExperimentConfig
from orbit.dataset.sample_factory import build_sample
from orbit.scorers.collusion_scorer import collusion_scorer
from orbit.scorers.security_scorer import security_scorer
from orbit.solvers.orchestrator import mas_orchestrator
from orbit.solvers.orchestrator_v2 import mas_orchestrator_v2
from orbit.solvers.setup_solver import mas_environment_setup
from orbit.validation.integrity_scorer import integrity_scorer


@task
def security_benchmark(
    config: ExperimentConfig | None = None,
    config_path: str | None = None,
    baseline: str = "none",
    orchestrator: str = "v1",
) -> Task:
    """Create an Inspect Task for a security benchmark experiment.

    This is the main entry point. Everything flows through here:
        YAML -> ExperimentConfig -> Task -> eval()

    Args:
        config: Pre-built ExperimentConfig. Used when called programmatically.
        config_path: Path to YAML config file. Used via Inspect CLI.
        baseline: Baseline mode: "none", "no_defense", "no_attack", "benign".

    Returns:
        Inspect Task configured for the security benchmark.
    """
    # 1. Load config
    if config is not None and config_path is not None:
        raise ValueError("Provide either config or config_path, not both")

    if config is None and config_path is not None:
        from orbit.wrapper.yaml_loader import load_experiment_config

        config = load_experiment_config(config_path)

    if config is None:
        raise ValueError("Either config or config_path must be provided")

    # 2. Apply baseline
    mode = BaselineMode(baseline)
    config = apply_baseline(config, mode)

    # 3. Build dataset (single sample from config)
    sample = build_sample(config, variant=mode.value)
    dataset = MemoryDataset([sample])

    # 4. Build sandbox spec
    sandbox = None
    if config.scenario.sandbox and config.scenario.sandbox.compose_file:
        sandbox = ("docker", config.scenario.sandbox.compose_file)
    elif config.scenario.sandbox:
        sandbox = "docker"

    # 5. Build scorer list
    # IMPORTANT: Scorer ordering matters! Scenario-specific scorers populate
    # CollusionLog and DCOPState, so they must run BEFORE collusion_scorer and
    # dcop_scorer which read from those stores.
    scorers = [security_scorer(), integrity_scorer()]

    # Add scenario-specific scorers FIRST (they populate CollusionLog/DCOPState)
    # DCOP-compatible scenarios (jira_ticket, hospital, meeting_scheduling) get both
    # their domain-specific scorer AND the generic dcop_scorer for DCOP metrics.
    # Non-DCOP scenarios (swe_bench, browserart) only get their own scorer.
    scenario_name = config.scenario.name
    if scenario_name == "jira_ticket":
        from orbit.scenarios.cooperative_allocation.jira_ticket.scorer import jira_ticket_scorer
        scorers.append(jira_ticket_scorer())
    elif scenario_name == "hospital":
        from orbit.scenarios.cooperative_allocation.hospital.scorer import hospital_scorer
        scorers.append(hospital_scorer())
    elif scenario_name == "meeting_scheduling":
        from orbit.scenarios.cooperative_allocation.meeting_scheduling.scorer import meeting_scheduling_scorer
        scorers.append(meeting_scheduling_scorer())

    # Add DCOP scorer AFTER scenario scorer (reads DCOPState populated above)
    if scenario_name in ("jira_ticket", "hospital", "meeting_scheduling"):
        from orbit.scenarios.cooperative_allocation.dcop.scorer import dcop_scorer
        scorers.append(dcop_scorer())

    # Add collusion scorer LAST (reads CollusionLog populated by scenario scorer)
    has_collusion = any(a.attack_type == "collusion" for a in config.attacks)
    if has_collusion:
        scorers.append(collusion_scorer())

    # 6. Compose Task
    return Task(
        dataset=dataset,
        setup=mas_environment_setup(),
        solver=mas_orchestrator_v2() if orchestrator == "v2" else mas_orchestrator(),
        scorer=scorers,
        sandbox=sandbox,
        max_messages=config.scheduler.max_turns * 10,
        epochs=config.epochs,
    )
