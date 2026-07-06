"""Declarative YAML-shorthand resolution for scenario plugins (issue #16).

An ``orbit run`` YAML may carry scenario *shorthand* in ``config.metadata``:

    ``{prefix}_condition``       — a named topology/memory condition preset
    ``{prefix}_attack_preset``   — a named attack preset
    ``{prefix}_defense_preset``  — a named defense preset

Before the kit, three scenarios hand-rolled this resolution in per-plugin
``resolve`` hooks and each diverged somewhere (consumed keys not stripped,
conflicts silently swallowed, preset targets tailored differently) — the
issue-#9 shorthand-drop bug class. A scenario now *declares* its shorthand as
a :class:`ShorthandSpec` on its plugin and the shared :func:`resolve_shorthand`
implements the shape once: read key → arbitrate conflicts → materialise →
strip the consumed key → record a ``{prefix}_resolved_*`` breadcrumb.

Conflict policy (issues #31/#32), uniform across every scenario:

* **condition vs inline setup** — the explicit condition WINS. When the inline
  roster's agent names equal the condition's roster the inline block is treated
  as redundant documentation and resolution is silent; a *materially differing*
  roster is an error by default (a stray inline ``setup`` silently relabeling a
  condition sweep is the "mislabeled control" failure at scale), downgradable to
  a warning with ``ORBIT_LENIENT=1``.
* **preset vs inline attacks/defenses** — the inline section WINS and the
  preset is ignored WITH a warning + ``{prefix}_ignored_*_preset`` breadcrumb
  (previously a silent drop).
* **materialised attack targets** — every attack the resolver materialises must
  target an agent that will exist at runtime; a dangling target is an error
  (previously the attack silently failed to bind — a run labeled "attacked"
  that was actually benign).
* **baselines** — preset shorthand is never materialised into a dimension that
  the config's ``baseline_mode`` strips, so a ``no_attack``/``benign`` control
  cannot re-acquire its attack through a preset. Gated here, centrally, AND
  re-stripped after resolution by the shared pipeline (belt and suspenders for
  custom ``resolve`` hooks).

Idempotence: consumed keys are stripped (and the arbitration paths are pure),
so ``resolve`` of an already-resolved config is a no-op — the builder, ``orbit
validate`` and ``--dry-run`` may all resolve the same config.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from orbit.configs.attack import AttackConfig
    from orbit.configs.defense import DefenseConfig
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.setup import SetupConfig

logger = logging.getLogger(__name__)


class ShorthandConflictError(ValueError):
    """A config declared conflicting or unsatisfiable shorthand.

    Raised (unless ``ORBIT_LENIENT=1``) when a ``{prefix}_condition`` and an
    inline ``setup`` declare materially different rosters, or when a
    resolver-materialised attack targets an agent absent from the runtime
    roster. Both are construct-validity failures: the run would execute with a
    topology or threat different from the one its config claims.
    """


def lenient_mode() -> bool:
    """Whether ``ORBIT_LENIENT=1`` downgrades shorthand conflicts to warnings."""
    return os.environ.get("ORBIT_LENIENT", "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class ShorthandSpec:
    """Declares a scenario's YAML shorthand surface.

    Attach to :class:`~orbit.scenarios.registry.ScenarioPlugin` via its
    ``shorthand`` field; the shared builder / ``orbit validate`` / ``--dry-run``
    all resolve it through :func:`resolve_shorthand`. Every callable takes the
    preset/condition *name*; resolution mechanics (key reading, conflict
    arbitration, breadcrumbs, idempotence) live in the shared resolver.
    """

    prefix: str
    """Metadata key prefix: reads ``{prefix}_condition`` /
    ``{prefix}_attack_preset`` / ``{prefix}_defense_preset``."""

    condition: Callable[[str], Any] | None = None
    """Resolve a condition name to its topology — a ``SetupConfig``, or an
    object with ``.setup`` (and optionally ``.execution``) attributes such as
    browserart's ``ConditionSetup``. For ``topology_source=SCENARIO`` plugins
    (tau2/agentharm) leave this ``None``: their expansion consumes the
    condition natively and the resolver only *arbitrates* (see
    ``expected_roster``)."""

    attack_preset: Callable[..., "list[AttackConfig]"] | None = None
    """Resolve an attack-preset name to attack configs. Called with the bare
    name, or with ``(name, condition=<applied condition or None>)`` when
    ``preset_takes_condition`` is set."""

    defense_preset: Callable[[str], "list[DefenseConfig]"] | None = None
    """Resolve a defense-preset name to defense configs."""

    preset_takes_condition: bool = False
    """Pass the applied condition to ``attack_preset`` so the preset can tailor
    its target agent to the resolved topology (osworld/browserart)."""

    default_setup: Callable[[], "SetupConfig"] | None = None
    """The scenario's default topology template — used only to validate
    materialised attack targets when the config declares no topology (the
    scenario's ``expand`` will fall back to this same template)."""

    expected_roster: Callable[["ExperimentConfig", str], set[str] | None] | None = None
    """``topology_source=SCENARIO`` plugins only: the agent names the scenario
    will build for a given condition (and config, e.g. tau2's domain-dependent
    specialists). Used to arbitrate condition-vs-inline-setup; return ``None``
    for "unknown" to fall back to a warning."""

    default_condition: str | None = None
    """``topology_source=SCENARIO`` plugins only: the condition the scenario's
    expansion assumes when no ``{prefix}_condition`` key is declared, so an
    inline setup is arbitrated against what will actually run."""


# ---------------------------------------------------------------------------
# Shared resolver
# ---------------------------------------------------------------------------


def _roster(setup: "SetupConfig") -> set[str]:
    return {agent.name for agent in setup.agents}


def _conflict(message: str) -> None:
    """Raise, or warn under ``ORBIT_LENIENT=1``."""
    if lenient_mode():
        logger.warning("%s (proceeding: ORBIT_LENIENT=1)", message)
    else:
        raise ShorthandConflictError(
            f"{message} Set ORBIT_LENIENT=1 to downgrade this to a warning."
        )


def _arbitrate_scenario_topology(
    config: "ExperimentConfig", spec: ShorthandSpec
) -> None:
    """Condition-vs-inline-setup arbitration for SCENARIO-source plugins.

    tau2/agentharm derive their roster from ``{prefix}_condition`` inside
    ``expand`` and deliberately ignore ``config.setup``; the key is therefore
    NOT consumed here. An inline setup whose roster matches what the condition
    will build is redundant documentation (group validation forces the block to
    exist — issue #32) and passes silently; a materially different roster means
    the run's actual topology is not the one the config describes.
    """
    if not config.setup.agents:
        return
    meta = config.metadata or {}
    condition = meta.get(f"{spec.prefix}_condition", spec.default_condition)
    inline = _roster(config.setup)
    expected: set[str] | None = None
    if condition and spec.expected_roster is not None:
        # The hook's contract: return None for "unknown/unknowable" (expand
        # reports a bad condition name authoritatively). Exceptions propagate —
        # swallowing them here silently degraded arbitration to a warning.
        expected = spec.expected_roster(config, str(condition))
    if expected is not None and inline == expected:
        return  # redundant documentation of the condition's roster — silent
    if expected is None:
        logger.warning(
            "Scenario %r derives its topology from metadata.%s_condition; the "
            "inline setup.agents (%s) is IGNORED at runtime.",
            spec.prefix,
            spec.prefix,
            sorted(inline),
        )
        return
    _conflict(
        f"Scenario {spec.prefix!r} derives its topology from "
        f"metadata.{spec.prefix}_condition={condition!r}, which builds agents "
        f"{sorted(expected)} — but the inline setup.agents declares "
        f"{sorted(inline)}. The inline setup does NOT run; this config "
        f"mislabels the experiment. Remove setup.agents (or mirror the "
        f"condition's roster exactly), or change the condition."
    )


def _resolve_condition(
    config: "ExperimentConfig",
    spec: ShorthandSpec,
    meta: dict,
    updates: dict,
) -> str | None:
    """CONFIG-source condition resolution: the explicit condition wins."""
    key = f"{spec.prefix}_condition"
    condition = meta.get(key)
    if not condition:
        return None
    if spec.condition is None:
        return None  # scenario declares no condition registry; leave the key
    condition = str(condition)
    resolved = spec.condition(condition)  # unknown name raises here, loudly
    setup = getattr(resolved, "setup", resolved)
    execution = getattr(resolved, "execution", None)
    if config.setup.agents:
        inline, target = _roster(config.setup), _roster(setup)
        if inline != target:
            _conflict(
                f"Config declares BOTH an inline setup.agents topology "
                f"({sorted(inline)}) and metadata.{key}={condition!r} "
                f"({sorted(target)}), and the rosters differ. The condition "
                f"wins; declare exactly one topology source."
            )
        # Matching roster: the inline block is redundant documentation of the
        # condition — the condition's canonical setup is materialised silently.
    del meta[key]
    meta[f"{spec.prefix}_resolved_condition"] = condition
    updates["setup"] = setup
    if execution is not None:
        updates["execution"] = execution
    return condition


def _validate_materialized_targets(
    config: "ExperimentConfig",
    spec: ShorthandSpec,
    attacks: "list[AttackConfig]",
    updates: dict,
) -> None:
    """A resolver-materialised attack must target an agent that will exist.

    Uses the post-resolution topology (updates > config.setup > the scenario's
    default template) expanded to runtime names, so template-replicating
    scenarios (swe_bench) validate against ``solver_0``-style names. Issue
    #31.2: previously a preset falling back to an absent target produced a
    "no attack applied" run with exit code 0.
    """
    from orbit.scenarios.registry import runtime_agent_names

    candidate = config.model_copy(
        update={k: v for k, v in updates.items() if k in ("setup", "attacks")}
    )
    names = runtime_agent_names(candidate)
    if not names and spec.default_setup is not None:
        names = runtime_agent_names(
            candidate.model_copy(update={"setup": spec.default_setup()})
        )
    if not names:
        return  # roster genuinely unknowable offline; expand/validators own it
    for attack in attacks:
        targets = []
        if attack.target_agent:
            targets.append(attack.target_agent)
        coalition = (attack.properties or {}).get("coalition_agents")
        if isinstance(coalition, (list, tuple)):
            targets.extend(str(m) for m in coalition)
        missing = [t for t in targets if t not in names]
        if missing:
            _conflict(
                f"Attack preset resolved an attack {attack.name!r} targeting "
                f"agent(s) {missing} that do not exist in the runtime roster "
                f"{sorted(names)}. The attack would silently fail to bind — "
                f"pick a preset/condition combination whose targets exist."
            )


def resolve_shorthand(
    config: "ExperimentConfig", spec: ShorthandSpec, *, scenario_topology: bool
) -> "ExperimentConfig":
    """Resolve ``spec``-declared shorthand in ``config.metadata``.

    ``scenario_topology`` is True for ``topology_source=SCENARIO`` plugins
    (condition arbitrated, not materialised). Returns a new config (frozen
    models); idempotent.
    """
    from orbit.baselines.baselines import baseline_keeps_attacks, baseline_keeps_defenses

    meta = dict(config.metadata or {})
    updates: dict = {}
    applied_condition: str | None = None

    if scenario_topology:
        _arbitrate_scenario_topology(config, spec)
        applied_condition = meta.get(
            f"{spec.prefix}_condition", spec.default_condition
        )
    else:
        applied_condition = _resolve_condition(config, spec, meta, updates)

    for kind, getter, field, keeps in (
        ("attack", spec.attack_preset, "attacks", baseline_keeps_attacks(config)),
        ("defense", spec.defense_preset, "defenses", baseline_keeps_defenses(config)),
    ):
        key = f"{spec.prefix}_{kind}_preset"
        name = meta.get(key)
        if not name or getter is None:
            continue
        name = str(name)
        del meta[key]
        if getattr(config, field):
            logger.warning(
                "Config declares BOTH inline %s and metadata.%s=%r; using the "
                "inline %s and IGNORING the preset. Declare exactly one.",
                field, key, name, field,
            )
            meta[f"{spec.prefix}_ignored_{kind}_preset"] = name
            continue
        if not keeps:
            # A no_attack/benign control must not re-acquire the preset.
            meta[f"{spec.prefix}_baseline_stripped_{kind}_preset"] = name
            continue
        if kind == "attack" and spec.preset_takes_condition:
            items = getter(name, condition=applied_condition)
        else:
            items = getter(name)
        if kind == "attack" and not scenario_topology:
            _validate_materialized_targets(config, spec, items, updates)
        updates[field] = items
        meta[f"{spec.prefix}_resolved_{kind}_preset"] = name

    if meta != (config.metadata or {}):
        updates["metadata"] = meta
    return config.model_copy(update=updates) if updates else config
