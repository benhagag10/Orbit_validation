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

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.registry import ScenarioPlugin, get_scenario
from orbit.validation.validators import ConfigValidator

logger = logging.getLogger(__name__)

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
    stage: str = "expanding",
) -> None:
    """Fail loud if a plugin hook dropped a declared attack/defense dimension.

    The contract, stated precisely: ``resolve`` and ``expand`` own per-sample
    construction, so they MAY ADD attacks/defenses (code_ipi prepends a per-task
    injection) or transform an attack's *properties* (swe_bench resolves an
    ``auto`` side-task payload per group) — but must PRESERVE each declared
    ``attack_type``/``defense_type``, count included. The one thing they must
    never do is reduce the multiset of declared types: that silently collapses a
    security condition into a benign run — the issue-#9 bug class.

    Applied at TWO stages, each keyed on the pre-hook config so neither hook can
    lower the floor the other is checked against:
      * after ``resolve`` — ``resolve`` may materialise preset shorthand into
        MORE attacks/defenses but may not strip a type the config declared;
      * after ``expand`` — keyed on the post-resolve config.

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
                f"dimensions while {stage} sample {i} ({cfg.name!r}): "
                f"missing attacks={dict(missing_a)}, defenses={dict(missing_d)}. "
                f"A plugin's resolve()/expand() must thread "
                f"config.attacks/config.defenses into every produced config (it "
                f"may add to or transform them, but never drop a declared type). "
                f"See orbit.scenarios.registry.ScenarioPlugin."
            )


def _assert_topology_preserved(
    resolved: ExperimentConfig,
    produced: list[ExperimentConfig],
    plugin_name: str,
) -> None:
    """Fail loud if ``expand`` collapsed the topology or dropped shared memory.

    The attack/defense floor does not cover topology, yet "star silently
    degenerates to a single agent" is the worst construct-validity failure this
    project has. For a ``threads_setup`` plugin (the default), a produced config
    may not have FEWER agents than the resolved config, nor turn a declared
    ``memory.shared=True`` into ``False``. This is a floor: adding agents is
    fine (agent expansion happens later in the orchestrator). Scenarios that
    derive topology from a native selector (``threads_setup=False``:
    tau2/agentharm) are exempt — the builder warns instead.
    """
    want_agents = len(resolved.setup.agents)
    want_shared = resolved.setup.memory.shared
    if want_agents == 0 and not want_shared:
        return
    for i, cfg in enumerate(produced):
        if len(cfg.setup.agents) < want_agents:
            raise ScenarioExpansionError(
                f"Scenario plugin {plugin_name!r} collapsed the topology while "
                f"expanding sample {i} ({cfg.name!r}): {want_agents} agent(s) "
                f"declared, {len(cfg.setup.agents)} produced. A threads_setup "
                f"plugin's expand() must thread config.setup into every produced "
                f"config. See orbit.scenarios.registry.ScenarioPlugin.threads_setup."
            )
        if want_shared and not cfg.setup.memory.shared:
            raise ScenarioExpansionError(
                f"Scenario plugin {plugin_name!r} dropped shared memory while "
                f"expanding sample {i} ({cfg.name!r}): config declared "
                f"memory.shared=True but the produced config has shared=False."
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
    """Task-level knobs for the generic benchmark path.

    (``epochs`` is seeded centrally in ``build_scenario_task`` for every
    scenario, so it is not repeated here.)
    """
    return {
        "max_messages": config.scheduler.max_turns * 10,
    }


#: The generic, scenario-unaware plugin. Reproduces the historical
#: ``security_benchmark`` Task exactly (single sample, ``mas_environment_setup``,
#: generic scorers, message budget). Used when no scenario plugin matches.
DEFAULT_PLUGIN = ScenarioPlugin(
    name="__default__",
    expand=None,  # [config]
    build_setup=None,  # mas_environment_setup()
    build_solver=None,  # mas_orchestrator
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
    variant: str = "default",
) -> Task:
    """Assemble the Inspect ``Task`` for ``config`` using its scenario plugin.

    Args:
        config: The full experiment configuration (post-baseline).
        plugin: The scenario plugin to use. If ``None``, it is looked up by
            ``config.scenario.name``; an unregistered name uses
            :data:`DEFAULT_PLUGIN`.
        variant: Baseline/variant label propagated into sample ids/metadata.
    """
    if plugin is None:
        # A known-but-broken scenario raises ScenarioImportError here — an
        # actionable MissingScenarioDependencyError naming the exact
        # `uv sync --extra <name>` command when its optional extra is missing —
        # never a silent generic fallback. Only a genuinely unknown name falls
        # back — loudly, naming the registered scenarios so a typo is visible
        # rather than running as a benign single-sample generic task.
        plugin = get_scenario(config.scenario.name)
        if plugin is None:
            from orbit.scenarios.registry import registered_scenarios

            logger.warning(
                "No scenario plugin registered for scenario.name=%r; using the "
                "generic benchmark path (topology/attacks/defenses are still "
                "threaded, but there is no scenario-specific dataset expansion or "
                "scorers). If this is a typo, the registered scenarios are: %s.",
                config.scenario.name,
                ", ".join(sorted(registered_scenarios())),
            )
            plugin = DEFAULT_PLUGIN

    # 0. Resolve scenario-specific shorthand (preset/condition metadata keys)
    # into the canonical config. Runs on BOTH entry points, so a preset declared
    # in an `orbit run` YAML is resolved exactly as the `-T` factory resolves it.
    # Guarded against dropping a declared type: the floor is snapshotted from the
    # PRE-resolve config, so resolve may only ADD (materialise shorthand), never
    # strip. Idempotent: a no-op when the config is already resolved.
    if plugin.resolve is not None:
        pre_resolve = config
        config = plugin.resolve(config)
        _assert_experiment_dimensions_preserved(
            pre_resolve, [config], plugin.name, stage="resolving"
        )

    # 1. Dataset expansion (threads config.setup/attacks/defenses through).
    configs = plugin.expand(config) if plugin.expand else [config]

    # 1a. Central guarantee: a plugin's expand() may add attacks/defenses or
    # transform an attack's properties, but must preserve each declared
    # attack_type/defense_type (count included); a type-changing transform is
    # treated as a drop by design (issue #9). Enforced once, here, for every
    # scenario — see the helper's docstring.
    _assert_experiment_dimensions_preserved(config, configs, plugin.name)

    # 1b. Topology floor: a threads_setup plugin (the default) may not collapse
    # the agent count or drop shared memory during expansion. Scenarios that
    # derive topology from a native selector (tau2/agentharm) are exempt, but we
    # warn if they were handed an inline setup that will be ignored.
    if plugin.threads_setup:
        _assert_topology_preserved(config, configs, plugin.name)
    elif config.setup.agents:
        logger.warning(
            "Scenario %r derives its topology from its own condition selector; "
            "the inline setup.agents (%s) is IGNORED. Remove it, or select the "
            "topology via the scenario's condition.",
            plugin.name,
            [a.name for a in config.setup.agents],
        )

    # 1c. Surface ConfigValidator problems (unknown target_agent, dangling edge
    # references, ...) centrally. Most plugins override build_setup and so skip
    # `mas_environment_setup`, where ConfigValidator was previously the only
    # gate — an attack whose target_agent doesn't exist would then silently fail
    # to bind (a benign run) with no error. Validated on a representative
    # EXPANDED config (so condition-built rosters are seen, not tau2/agentharm's
    # empty pre-expand setup) with the default identity names — expansion
    # already renamed any template agents (swe_bench's solver -> solver_0), so
    # `expand_templates=True` here would double-expand. LOGGED not raised: the
    # representative sample may not capture per-sample roster variation; the
    # offline `orbit validate` / `--dry-run` (which do expand the template) are
    # the hard gates.
    representative = configs[0] if configs else config
    val_errors = ConfigValidator.validate(representative)
    if val_errors:
        logger.warning(
            "Scenario %r config %r has %d validation issue(s): %s",
            plugin.name,
            representative.name,
            len(val_errors),
            "; ".join(val_errors),
        )

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
        solver = plugin.build_solver(config)
    else:
        from orbit.solvers.orchestrator import mas_orchestrator
        solver = mas_orchestrator()

    scorers = plugin.build_scorers(config) if plugin.build_scorers else default_scorers(config)
    sandbox = plugin.build_sandbox(config) if plugin.build_sandbox else default_sandbox(config)

    # `config.epochs` is honoured for EVERY scenario: seed it as the default and
    # let a plugin's build_task_kwargs override (tau2 pins its own Epochs with
    # pass^k reducers). Without this, a YAML `epochs: N` was silently a no-op for
    # every plugin scenario — only the generic path passed it through.
    task_kwargs: dict[str, Any] = {"epochs": config.epochs}
    if plugin.build_task_kwargs:
        task_kwargs.update(plugin.build_task_kwargs(config))

    return Task(
        dataset=MemoryDataset(samples),
        setup=setup,
        solver=solver,
        scorer=scorers,
        sandbox=sandbox,
        **task_kwargs,
    )
