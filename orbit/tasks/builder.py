"""
Shared, scenario-agnostic Task builder.

``build_scenario_task`` is the single place that turns one ``ExperimentConfig``
into an Inspect ``Task``. It is total over ``ExperimentConfig``: dataset
expansion, sample building (which serializes the *whole* config into metadata),
setup solver, main solver, scorers and sandbox are all derived from the config
plus the scenario's :class:`~orbit.scenarios.registry.ScenarioPlugin`.

Both entry points converge here, so attack/defense/topology wiring lives in
exactly one consumer per scenario — see ``orbit.scenarios.registry`` for the
rationale. When ``config.scenario.name`` has no registered plugin, the generic
``DEFAULT_PLUGIN`` (the historical ``security_benchmark`` behaviour) is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.registry import ScenarioPlugin, get_scenario

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer

    from orbit.configs.experiment import ExperimentConfig


# ---------------------------------------------------------------------------
# Small shared helpers for scenario plugins
# ---------------------------------------------------------------------------


def split_csv(value: object) -> list[str] | None:
    """Normalise a comma-string or list into a list of strings (or ``None``).

    Scenario metadata may hold a selection filter as a comma-separated string
    (``"a,b"``) or an already-parsed list. Plugins use this to read either.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Generic defaults (extracted verbatim from the original security_benchmark)
# ---------------------------------------------------------------------------


def default_sandbox(config: ExperimentConfig) -> tuple[str, str] | str | None:
    """Derive the Task-level sandbox spec from the scenario config."""
    if config.scenario.sandbox and config.scenario.sandbox.compose_file:
        return ("docker", config.scenario.sandbox.compose_file)
    if config.scenario.sandbox:
        return "docker"
    return None


def default_scorers(config: ExperimentConfig) -> list[Scorer]:
    """The generic scorer list, including DCOP/collusion wiring.

    IMPORTANT: Scorer ordering matters. Scenario-specific scorers populate
    CollusionLog and DCOPState, so they must run BEFORE collusion_scorer and
    dcop_scorer which read from those stores.
    """
    from orbit.scorers.collusion_scorer import collusion_scorer
    from orbit.scorers.security_scorer import security_scorer
    from orbit.validation.integrity_scorer import integrity_scorer

    scorers: list[Scorer] = [security_scorer(), integrity_scorer()]

    # Add scenario-specific scorers FIRST (they populate CollusionLog/DCOPState).
    # DCOP-compatible scenarios (jira_ticket, hospital, meeting_scheduling) get
    # both their domain-specific scorer AND the generic dcop_scorer.
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

    # Add DCOP scorer AFTER scenario scorer (reads DCOPState populated above).
    if scenario_name in ("jira_ticket", "hospital", "meeting_scheduling"):
        from orbit.scenarios.cooperative_allocation.dcop.scorer import dcop_scorer
        scorers.append(dcop_scorer())

    # Add collusion scorer LAST (reads CollusionLog populated by scenario scorer).
    has_collusion = any(a.attack_type == "collusion" for a in config.attacks)
    if has_collusion:
        scorers.append(collusion_scorer())

    return scorers


def _default_task_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    """Task-level knobs for the generic benchmark path."""
    return {
        "max_messages": config.scheduler.max_turns * 10,
        "epochs": config.epochs,
    }


#: The generic, scenario-unaware plugin. Reproduces the historical
#: ``security_benchmark`` Task exactly (single sample, ``mas_environment_setup``,
#: generic scorers, message budget). Used when no scenario plugin matches.
DEFAULT_PLUGIN = ScenarioPlugin(
    name="__default__",
    expand=None,  # [config]
    build_setup=None,  # mas_environment_setup()
    build_solver=None,  # mas_orchestrator (v1/v2)
    build_scorers=default_scorers,
    build_sandbox=default_sandbox,
    build_task_kwargs=_default_task_kwargs,
    name_sample_ids=False,  # ids fall back to "{name}_{variant}"
)


# ---------------------------------------------------------------------------
# The single assembler
# ---------------------------------------------------------------------------


def build_scenario_task(
    config: ExperimentConfig,
    plugin: ScenarioPlugin | None = None,
    *,
    orchestrator: str = "v1",
    variant: str = "default",
) -> Task:
    """Assemble the Inspect ``Task`` for ``config`` using its scenario plugin.

    Args:
        config: The full experiment configuration (post-baseline).
        plugin: The scenario plugin to use. If ``None``, it is looked up by
            ``config.scenario.name``; an unregistered name uses
            :data:`DEFAULT_PLUGIN`.
        orchestrator: ``"v1"`` or ``"v2"`` — selects the orchestrator solver
            when the plugin doesn't override ``build_solver``.
        variant: Baseline/variant label propagated into sample ids/metadata.
    """
    if plugin is None:
        plugin = get_scenario(config.scenario.name) or DEFAULT_PLUGIN

    # 1. Dataset expansion (threads config.setup/attacks/defenses through).
    configs = plugin.expand(config) if plugin.expand else [config]

    # 2. Build one sample per config (serializes the whole config to metadata).
    samples = []
    for cfg in configs:
        if plugin.name_sample_ids:
            samples.append(build_sample(cfg, sample_id=cfg.name, variant=variant))
        else:
            samples.append(build_sample(cfg, variant=variant))

    # 3. Resolve the scenario-specific Task ingredients (defaults otherwise).
    if plugin.build_setup is not None:
        setup = plugin.build_setup(config)
    else:
        from orbit.solvers.setup_solver import mas_environment_setup
        setup = mas_environment_setup()

    if plugin.build_solver is not None:
        solver = plugin.build_solver(config, orchestrator)
    else:
        if orchestrator == "v2":
            from orbit.solvers.orchestrator_v2 import mas_orchestrator_v2
            solver = mas_orchestrator_v2()
        else:
            from orbit.solvers.orchestrator import mas_orchestrator
            solver = mas_orchestrator()

    scorers = plugin.build_scorers(config) if plugin.build_scorers else default_scorers(config)
    sandbox = plugin.build_sandbox(config) if plugin.build_sandbox else default_sandbox(config)
    task_kwargs = plugin.build_task_kwargs(config) if plugin.build_task_kwargs else {}

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        setup=setup,
        solver=solver,
        scorer=scorers,
        sandbox=sandbox,
        **task_kwargs,
    )
