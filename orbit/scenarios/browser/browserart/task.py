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
"""

from __future__ import annotations

import logging
import warnings

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.browser.browserart.config_builder import (
    build_experiment_configs_from_scenario,
    default_topology_template,
)
from orbit.scenarios.browser.browserart.configs import BrowserARTScenarioConfig
from orbit.scenarios.browser.browserart.scorer import browserart_scorer

logger = logging.getLogger(__name__)


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
        topology: Topology template: 'default' or path to YAML file.
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
    # ---- Suppress noisy warnings unless verbose is requested ----
    if not verbose:
        warnings.filterwarnings("ignore", message=".*Overriding environment.*")
        logging.getLogger(
            "orbit.scenarios.browser.browserart.client"
        ).setLevel(logging.ERROR)

    # ---- Resolve human-readable params to condition name ----
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

    # Auto-adjust defaults for multi-step extension tasks
    if dataset == "hbb_extension":
        if max_turns == 15:  # default wasn't overridden
            max_turns = 50
        if max_time == 300.0:  # default wasn't overridden
            max_time = 600.0

    # Build scenario config
    categories = [c.strip() for c in category.split(",")] if category else None
    parsed_task_ids = None
    if task_ids is not None:
        if isinstance(task_ids, int):
            parsed_task_ids = [task_ids]
        else:
            parsed_task_ids = [int(x.strip()) for x in str(task_ids).split(",")]
    scenario_config = BrowserARTScenarioConfig(
        data_path=data_path,
        dataset=dataset,
        categories=categories,
        task_ids=parsed_task_ids,
        max_behaviors=max_behaviors,
        seed=seed,
        classifier_model=classifier_model,
        max_turns=max_turns,
        max_time_seconds=max_time,
    )

    # Load topology template (and optional execution config for mesh
    # conditions that need Path C scheduling).
    execution_template = None
    if condition is not None:
        from orbit.scenarios.browser.browserart.condition_presets import (
            get_condition,
        )

        cs = get_condition(condition)
        topology_template = cs.setup
        execution_template = cs.execution
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
        from orbit.scenarios.browser.browserart.presets import get_attack_preset

        attacks = get_attack_preset(attack_preset, condition=condition)
    if defense_preset:
        from orbit.scenarios.browser.browserart.presets import get_defense_preset

        defenses = get_defense_preset(defense_preset)

    # Build experiment configs
    configs = build_experiment_configs_from_scenario(
        scenario_config=scenario_config,
        topology_template=topology_template,
        attacks=attacks,
        defenses=defenses,
        execution=execution_template,
    )

    samples = [build_sample(config, sample_id=config.name) for config in configs]

    if not samples:
        logger.warning(
            "No samples built — check scenario config and available behaviors"
        )

    # Every condition drives the persistent browserart-service via
    # scenario tools through the same chain: ``browserart_setup`` opens
    # a session and seeds ``BrowserARTState``; ``mas_orchestrator`` runs
    # Path A (star / memory / single) or Path C (mesh with
    # ``agent_groups``); ``browserart_teardown`` releases the session
    # when the loop finishes.
    from inspect_ai.solver import chain as solver_chain

    from orbit.scenarios.browser.browserart.setup import (
        browserart_setup,
        browserart_teardown,
    )
    from orbit.solvers.orchestrator import mas_orchestrator

    task_solver = solver_chain(
        browserart_setup(headless=headless, record_video_dir=record_video_dir),
        mas_orchestrator(),
        browserart_teardown(),
    )

    # Build task
    from orbit.scorers.security_scorer import security_scorer

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        solver=task_solver,
        scorer=[browserart_scorer(classifier_model), security_scorer()],
        message_limit=60,
    )
