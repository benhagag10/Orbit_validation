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

from collections import Counter
from typing import TYPE_CHECKING, Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.registry import ScenarioPlugin, get_scenario

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer

    from orbit.configs.experiment import ExperimentConfig


class ScenarioExpansionError(ValueError):
    """A scenario plugin's ``expand`` violated the Task-construction contract.

    Raised by the shared builder when dataset expansion silently dropped an
    experiment-level dimension (an attack/defense declared on the config) —
    the issue-#9 bug class, made impossible to do silently.
    """


# ---------------------------------------------------------------------------
# Small shared helpers for scenario plugins
# ---------------------------------------------------------------------------


def _assert_experiment_dimensions_preserved(
    original: ExperimentConfig,
    produced: list[ExperimentConfig],
    plugin_name: str,
) -> None:
    """Fail loud if dataset expansion dropped an experiment-level dimension.

    The contract, stated precisely: ``expand`` owns per-sample construction, so
    a plugin's ``expand()`` MAY ADD attacks/defenses (code_ipi prepends a
    per-task injection) or transform an attack's *properties* (swe_bench
    resolves an ``auto`` side-task payload per group) — but it must PRESERVE
    each declared ``attack_type``/``defense_type``, count included. The one
    thing it must never do is reduce the multiset of declared types: that
    silently collapses a security condition into a benign run — the issue-#9
    bug class.

    The check is a per-type multiset *floor* (``>=`` on the ``attack_type`` /
    ``defense_type`` Counters), keyed on the declared *type*, not on attack
    identity. So a property-only transform that keeps the type passes, and
    additions pass — but a *type-changing* transform is, by design, treated as
    a drop: turning a declared ``injection`` attack into a ``misuse`` one
    leaves the ``injection`` floor unmet and is flagged. Preserve the declared
    type (transform within it) if you need to mutate an attack.

    Enforcing this here, in the single builder every scenario funnels through,
    makes the drop structurally impossible regardless of whether an individual
    plugin author remembered to thread the dimensions through.
    """
    want_attacks = Counter(a.attack_type for a in original.attacks)
    want_defenses = Counter(d.defense_type for d in original.defenses)
    if not want_attacks and not want_defenses:
        return
    for i, cfg in enumerate(produced):
        missing_a = want_attacks - Counter(a.attack_type for a in cfg.attacks)
        missing_d = want_defenses - Counter(d.defense_type for d in cfg.defenses)
        if missing_a or missing_d:
            raise ScenarioExpansionError(
                f"Scenario plugin {plugin_name!r} dropped experiment-level "
                f"dimensions while expanding sample {i} ({cfg.name!r}): "
                f"missing attacks={dict(missing_a)}, defenses={dict(missing_d)}. "
                f"A plugin's expand() must thread config.attacks/config.defenses "
                f"into every produced config (it may add to or transform them, "
                f"but never drop a declared type). See "
                f"orbit.scenarios.registry.ScenarioPlugin.expand."
            )


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


def baseline_keeps_attacks(config: ExperimentConfig) -> bool:
    """Whether ``config``'s baseline mode retains attacks.

    A plugin's ``resolve`` hook materialises attack *preset shorthand* into
    ``config.attacks``. That must NOT happen when a baseline intentionally
    stripped attacks (``no_attack``/``benign``) — otherwise the "control" run
    silently re-acquires the attack and the attack-effect delta is measured
    against a contaminated reference. ``resolve`` hooks gate preset injection on
    this. Keyed on ``baseline_mode`` (set once by ``apply_baseline`` and carried
    through ``model_copy``), so it is correct regardless of whether the baseline
    was applied before or after ``resolve``.
    """
    from orbit.configs.baseline import BaselineMode

    return config.baseline_mode not in (BaselineMode.NO_ATTACK, BaselineMode.BENIGN)


def baseline_keeps_defenses(config: ExperimentConfig) -> bool:
    """Whether ``config``'s baseline mode retains defenses (see
    :func:`baseline_keeps_attacks`)."""
    from orbit.configs.baseline import BaselineMode

    return config.baseline_mode not in (BaselineMode.NO_DEFENSE, BaselineMode.BENIGN)


def resolve_scenario_shorthand(config: ExperimentConfig) -> ExperimentConfig:
    """Apply the scenario plugin's ``resolve`` hook to ``config`` (or a no-op).

    Materialises scenario *shorthand* (a ``*_condition`` topology preset, an
    ``*_attack_preset`` name) into the canonical config, exactly as the shared
    builder does before dataset expansion. Callers that inspect or validate a
    config *before* it reaches the builder — ``orbit validate`` and
    ``orbit run --dry-run`` — use this so a condition-only YAML reports and
    validates against its resolved topology rather than an empty one. Idempotent
    (``resolve`` hooks strip the shorthand they consume)."""
    plugin = get_scenario(config.scenario.name)
    if plugin is not None and plugin.resolve is not None:
        return plugin.resolve(config)
    return config


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

    # 0. Resolve scenario-specific shorthand (preset/condition metadata keys)
    # into the canonical config. Runs on BOTH entry points, so a preset declared
    # in an `orbit run` YAML is resolved exactly as the `-T` factory resolves it
    # — and BEFORE the dimension invariant, so the resolved attacks/defenses are
    # guarded too. Idempotent: a no-op when the config is already resolved.
    if plugin.resolve is not None:
        config = plugin.resolve(config)

    # 1. Dataset expansion (threads config.setup/attacks/defenses through).
    configs = plugin.expand(config) if plugin.expand else [config]

    # 1a. Central guarantee: a plugin's expand() may add attacks/defenses or
    # transform an attack's properties, but must preserve each declared
    # attack_type/defense_type (count included); a type-changing transform is
    # treated as a drop by design (issue #9). Enforced once, here, for every
    # scenario — see the helper's docstring.
    _assert_experiment_dimensions_preserved(config, configs, plugin.name)

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
