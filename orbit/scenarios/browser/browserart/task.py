"""
BrowserART safety task entry point.

Registered with Inspect via _registry.py. Creates a Task from BrowserART
behaviors with configurable topology, attacks, and defenses.

Every condition (single-agent, star, memory, mesh) drives the persistent
``browserart-service`` Docker container via the shared ``mas_orchestrator``:
the task solver is ``chain(browserart_setup, mas_orchestrator,
browserart_teardown)``. Star and memory conditions run Path A (orchestrator
is the ``react()`` root, specialists are ``as_tool()`` sub-agents); mesh
conditions run Path C (``AgentScheduler`` with round_robin + observation
summary). Start the service with ``scripts/browserart_service.sh up`` and
point ``BROWSERART_SERVICE_URL`` at it (default ``http://localhost:7878``).

Construction is unified through the scenario-agnostic builder
(``orbit.tasks.builder.build_scenario_task``): the ``browserart`` plugin's
``expand`` hook is total over ``ExperimentConfig`` — it reads the dataset
selection from ``config.metadata`` and threads ``config.setup`` (topology),
``config.attacks`` and ``config.defenses`` into every per-behavior config. Both
``inspect eval orbit/browserart_safety -T ...`` and ``orbit run <yaml>`` reach
that same hook, so no config section can be silently dropped.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from inspect_ai import Task, task

from orbit.scenarios.browser.browserart.config_builder import (
    build_experiment_configs_from_scenario,
    default_topology_template,
)
from orbit.scenarios.browser.browserart.configs import BrowserARTScenarioConfig
from orbit.scenarios.registry import ScenarioPlugin, register_scenario
from orbit.tasks.builder import build_scenario_task

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer
    from inspect_ai.solver import Solver

    from orbit.configs.execution import ExecutionConfig
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.setup import SetupConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario plugin hooks (total over ExperimentConfig)
# ---------------------------------------------------------------------------


def _suppress_browserart_warnings(verbose: bool) -> None:
    if not verbose:
        warnings.filterwarnings("ignore", message=".*Overriding environment.*")
        logging.getLogger(
            "orbit.scenarios.browser.browserart.client"
        ).setLevel(logging.ERROR)


def _resolve_topology(
    config: ExperimentConfig,
) -> tuple[SetupConfig, ExecutionConfig | None]:
    """Resolve (topology_template, execution_template) from the config.

    Inline ``config.setup`` wins (the YAML/​-T factory is authoritative); a
    named ``browserart_condition`` in metadata is the fallback; finally the
    single-agent default.
    """
    meta = config.metadata or {}
    if config.setup.agents:
        execution = config.execution if config.execution.agent_groups else None
        return config.setup, execution
    condition = meta.get("browserart_condition")
    if condition:
        from orbit.scenarios.browser.browserart.condition_presets import get_condition

        cs = get_condition(condition)
        return cs.setup, cs.execution
    return default_topology_template(), None


def _browserart_expand(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Fan the config out into one ExperimentConfig per BrowserART behavior."""
    meta = config.metadata or {}
    _suppress_browserart_warnings(bool(meta.get("browserart_verbose", False)))

    categories = meta.get("browserart_categories")
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]

    task_ids = meta.get("browserart_task_ids")
    if isinstance(task_ids, str):
        task_ids = [int(x.strip()) for x in task_ids.split(",") if x.strip()]
    elif isinstance(task_ids, int):
        task_ids = [task_ids]

    scenario_config = BrowserARTScenarioConfig(
        data_path=meta.get("browserart_data_path"),
        dataset=meta.get("browserart_dataset", "hbb"),
        categories=categories,
        task_ids=task_ids,
        max_behaviors=meta.get("browserart_max_behaviors"),
        seed=meta.get("browserart_seed"),
        classifier_model=meta.get("browserart_classifier_model", "openai/gpt-4.1"),
        max_turns=config.scheduler.max_turns,
        max_time_seconds=config.scheduler.max_time_seconds,
    )

    topology_template, execution_template = _resolve_topology(config)

    configs = build_experiment_configs_from_scenario(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=list(config.attacks) or None,
        defenses=list(config.defenses) or None,
        execution=execution_template,
    )
    if not configs:
        logger.warning(
            "No samples built — check scenario config and available behaviors"
        )
    return configs


def _browserart_solver(config: ExperimentConfig, orchestrator: str) -> Solver:
    """browserart_setup → mas_orchestrator → browserart_teardown."""
    from inspect_ai.solver import chain as solver_chain

    from orbit.scenarios.browser.browserart.setup import (
        browserart_setup,
        browserart_teardown,
    )
    from orbit.solvers.orchestrator import mas_orchestrator

    meta = config.metadata or {}
    return solver_chain(
        browserart_setup(
            headless=meta.get("browserart_headless", True),
            record_video_dir=meta.get("browserart_record_video_dir"),
        ),
        mas_orchestrator(),
        browserart_teardown(),
    )


def _browserart_scorers(config: ExperimentConfig) -> list[Scorer]:
    from orbit.scenarios.browser.browserart.scorer import browserart_scorer
    from orbit.scorers.security_scorer import security_scorer

    meta = config.metadata or {}
    classifier_model = meta.get("browserart_classifier_model", "openai/gpt-4.1")
    return [browserart_scorer(classifier_model), security_scorer()]


BROWSERART_PLUGIN = register_scenario(
    ScenarioPlugin(
        name="browserart",
        expand=_browserart_expand,
        build_setup=lambda config: None,  # setup folded into the solver chain
        build_solver=_browserart_solver,
        build_scorers=_browserart_scorers,
        build_sandbox=lambda config: None,  # persistent browserart-service
        build_task_kwargs=lambda config: {"message_limit": 60},
    )
)


# ---------------------------------------------------------------------------
# @task factory — thin resolver of flat -T params into an ExperimentConfig
# ---------------------------------------------------------------------------


@task
def browserart_safety(
    category: str | None = None,
    max_behaviors: int | None = None,
    seed: int | None = None,
    classifier_model: str = "openai/gpt-4.1",
    data_path: str | None = None,
    max_turns: int = 15,
    max_time: float = 300.0,
    topology_file: str = "default",
    defense_preset: str | None = None,
    attack_preset: str | None = None,
    headless: bool = True,
    record_video_dir: str | None = None,
    condition: str | None = None,
    agents: str | None = None,
    topology: str = "star",
    memory: str = "none",
    instructions: str = "detailed",
    dataset: str = "hbb",
    task_ids: str | None = None,
    verbose: bool = False,
) -> Task:
    """BrowserART multi-agent safety benchmark task.

    Evaluates browser agent safety against harmful behavior instructions
    from the BrowserART dataset. Requires the persistent
    ``browserart-service`` container (see
    ``scripts/browserart_service.sh up``).

    Examples:
        # Default single-agent:
        inspect eval orbit/browserart_safety \\
            -T max_behaviors=1 -T max_turns=5 --model openai/gpt-4o

        # Condition preset:
        inspect eval orbit/browserart_safety \\
            -T condition=star_specialist -T max_behaviors=1 --model openai/gpt-4o

    Args:
        category: Filter behaviors by semantic_category.
        max_behaviors: Limit number of behaviors.
        seed: Random seed for behavior sampling.
        classifier_model: Model for the behavior classifier judge.
        data_path: Override path to hbb.json dataset.
        max_turns: Maximum turns per behavior.
        max_time: Maximum wall-clock seconds per behavior.
        topology_file: Topology template: 'default' or path to YAML file.
        defense_preset: Defense preset name.
        attack_preset: Attack preset name.
        headless: Run browser headlessly inside the service.
        record_video_dir: Directory for browser video recordings inside
            the service container. ``None`` disables recording.
        condition: Condition preset name (e.g. ``"star_specialist"``,
            ``"memory_full"``). When set, overrides topology with the
            condition's setup and routes to the appropriate solver.
        dataset: Dataset variant: ``"hbb"`` (default), ``"hbb_hr"``,
            ``"hbb_benign"``, or ``"hbb_extension"`` (10 multi-step tasks).
        task_ids: Comma-separated list of task IDs to include (e.g.
            ``"227,228,229"``). ``None`` means all tasks in the dataset.
        verbose: Show all warnings (gymnasium registration, browser overlay
            failures, etc.). Default ``False`` suppresses noisy warnings.
    """
    from orbit.configs.metrics import MetricsConfig
    from orbit.configs.scenario import ScenarioConfig
    from orbit.configs.scheduler import SchedulerConfig
    from orbit.configs.setup import SetupConfig

    _suppress_browserart_warnings(verbose)

    # ---- Resolve human-readable params to a condition name ----
    if agents is not None and condition is not None:
        raise ValueError(
            "Cannot specify both --condition and --agents. Use one or the other."
        )
    if agents is not None:
        from orbit.scenarios.browser.browserart.condition_presets import resolve_condition

        condition = resolve_condition(
            agents=agents, topology=topology, memory=memory,
            instructions=instructions,
        )

    # Auto-adjust defaults for multi-step extension tasks.
    if dataset == "hbb_extension":
        if max_turns == 15:  # default wasn't overridden
            max_turns = 50
        if max_time == 300.0:  # default wasn't overridden
            max_time = 600.0

    # ---- Resolve topology template (+ optional Path-C execution) ----
    execution_template: ExecutionConfig | None = None
    if condition is not None:
        from orbit.scenarios.browser.browserart.condition_presets import get_condition

        cs = get_condition(condition)
        topology_template = cs.setup
        execution_template = cs.execution
    elif topology_file == "default":
        topology_template = default_topology_template()
    else:
        import yaml

        with open(topology_file) as f:
            topo_data = yaml.safe_load(f)
        topology_template = SetupConfig(**topo_data)

    # ---- Resolve attack/defense presets into inline configs ----
    attacks = None
    defenses = None
    if attack_preset:
        from orbit.scenarios.browser.browserart.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset, condition=condition)
    if defense_preset:
        from orbit.scenarios.browser.browserart.presets import get_defense_preset

        defenses = get_defense_preset(defense_preset)

    # ---- Parse the dataset-selection params into metadata ----
    categories = [c.strip() for c in category.split(",")] if category else None
    parsed_task_ids = None
    if task_ids is not None:
        if isinstance(task_ids, int):
            parsed_task_ids = [task_ids]
        else:
            parsed_task_ids = [int(x.strip()) for x in str(task_ids).split(",")]

    metadata = {
        "browserart_dataset": dataset,
        "browserart_categories": categories,
        "browserart_task_ids": parsed_task_ids,
        "browserart_max_behaviors": max_behaviors,
        "browserart_seed": seed,
        "browserart_data_path": data_path,
        "browserart_classifier_model": classifier_model,
        "browserart_headless": headless,
        "browserart_record_video_dir": record_video_dir,
        "browserart_verbose": verbose,
    }

    from orbit.configs.experiment import ExperimentConfig

    config_kwargs: dict = {
        "name": "browserart",
        "description": "BrowserART multi-agent safety benchmark",
        "setup": topology_template,
        "scenario": ScenarioConfig(
            name="browserart",
            description="BrowserART web agent safety",
            sandbox=None,
        ),
        "attacks": attacks or [],
        "defenses": defenses or [],
        "scheduler": SchedulerConfig(
            max_turns=max_turns,
            max_time_seconds=max_time,
            halt_on_convergence=False,
        ),
        "metrics": MetricsConfig(),
        "metadata": metadata,
    }
    if execution_template is not None:
        config_kwargs["execution"] = execution_template

    config = ExperimentConfig(**config_kwargs)

    return build_scenario_task(config, BROWSERART_PLUGIN)
