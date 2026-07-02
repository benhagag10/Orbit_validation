"""
Scenario plugin registry — the scenario-agnostic Task-construction API.

Every Orbit scenario turns one ``ExperimentConfig`` into an Inspect ``Task``.
The *only* things that genuinely differ between scenarios are six dimensions:

    expand   — how one config fans out into per-sample configs (dataset)
    setup    — the setup solver run before the orchestrator (state init, etc.)
    solver   — the main solver (almost always the shared ``mas_orchestrator``)
    scorers  — the scorer list
    sandbox  — the Task-level sandbox spec
    kwargs   — Task-level knobs (message_limit, message budget)

Everything else — baseline application, building Samples from configs,
serializing the *whole* config into sample metadata, and applying
attacks/defenses/topology at runtime — is identical across scenarios and lives
in exactly one place (``orbit.tasks.builder.build_scenario_task`` and the
``mas_orchestrator`` solver).

A scenario registers a :class:`ScenarioPlugin` declaring only its unique
dimensions; the shared builder fills in the rest from sensible defaults. Both
entry points converge on the same plugin:

    * ``orbit run <yaml>`` → ``security_benchmark(config)`` → ``build_scenario_task``
    * ``inspect eval orbit/<scenario> -T ...`` → the @task factory resolves its
      flat params/presets into an ``ExperimentConfig`` → ``build_scenario_task``

Because there is exactly one consumer per scenario and it sees the entire
``ExperimentConfig``, a config section that is set (topology/attacks/defenses)
can never be silently dropped — the class of bug in issue #9 is impossible by
construction. Adding a new scenario is "register a plugin", not "hand-write
another Task factory".
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from inspect_ai.scorer import Scorer
    from inspect_ai.solver import Solver

    from orbit.configs.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


class ScenarioImportError(ImportError):
    """A registered scenario's module failed to import.

    Raised by :func:`get_scenario` when a *known* scenario (one in ``_LOADERS``)
    cannot be imported. Scenario modules import their heavy optional deps lazily
    (inside ``expand``/dataset builders), so an import-time failure here is a
    real bug — never a merely-missing extra — and must surface instead of
    silently degrading the run to the generic plugin.
    """


@dataclass(frozen=True)
class ScenarioPlugin:
    """Declares the scenario-specific pieces of Task construction.

    Each hook is total over ``ExperimentConfig`` — it receives the whole
    config (with its ``setup``/``attacks``/``defenses``/``metadata``) and
    returns the corresponding Task ingredient. ``None`` means "use the shared
    default". The shared builder (:func:`orbit.tasks.builder.build_scenario_task`)
    composes the ingredients into a single Inspect ``Task``.
    """

    name: str
    """The ``scenario.name`` this plugin handles."""

    resolve: Callable[[ExperimentConfig], ExperimentConfig] | None = None
    """Resolve scenario-specific *shorthand* into the canonical config, BEFORE
    dataset expansion. e.g. a preset name in ``config.metadata``
    (``swe_bench_attack_preset``) -> ``config.attacks``; a named
    ``osworld_condition`` -> ``config.setup``. The shared builder calls this on
    BOTH entry points, so shorthand declared in an ``orbit run`` YAML is resolved
    exactly as the ``inspect eval -T`` factory resolves it — closing the gap
    where the deleted runner dispatch was the only resolver (and making the
    resolved attacks/defenses visible to the dimension-preserving invariant).
    MUST be idempotent: no-op when the config is already resolved (e.g. when
    ``config.attacks`` is already populated). Default: no resolution."""

    expand: Callable[[ExperimentConfig], list[ExperimentConfig]] | None = None
    """Fan one config out into per-sample configs (dataset expansion). The
    expansion MUST thread ``config.setup`` (topology), ``config.attacks`` and
    ``config.defenses`` into every produced config. Default: ``[config]``
    (single sample)."""

    build_setup: Callable[[ExperimentConfig], Solver | None] | None = None
    """Return the Task ``setup=`` solver, or ``None`` for no setup solver.
    If this hook itself is ``None``, the shared default
    (``mas_environment_setup``) is used."""

    build_solver: Callable[[ExperimentConfig, str], Solver] | None = None
    """Return the main solver. Receives ``(config, orchestrator)`` where
    ``orchestrator`` is ``"v1"``/``"v2"``. Default: ``mas_orchestrator`` (v1)
    or ``mas_orchestrator_v2`` (v2). A scenario may wrap the orchestrator
    (e.g. browserart's open/close-session chain) but MUST keep
    ``mas_orchestrator`` so attacks/defenses are applied at runtime."""

    build_scorers: Callable[[ExperimentConfig], list[Scorer]] | None = None
    """Return the scorer list. Default: the generic security/integrity
    (+ DCOP/collusion) scorers."""

    build_sandbox: Callable[[ExperimentConfig], Any] | None = None
    """Return the Task-level ``sandbox=`` spec, or ``None``. If this hook is
    ``None``, the sandbox is derived from ``config.scenario.sandbox``."""

    build_task_kwargs: Callable[[ExperimentConfig], dict[str, Any]] | None = None
    """Return extra keyword arguments for ``Task(...)`` — e.g.
    ``{"message_limit": 60}`` (browserart), ``{"epochs": Epochs(...)}`` (tau2),
    or ``{"max_messages": ..., "epochs": ...}`` (the generic benchmark).
    Default: ``{}``."""

    name_sample_ids: bool = True
    """If True, each produced sample's id is its config name
    (``build_sample(c, sample_id=c.name)``) — the convention for expanding
    scenarios. If False, ids fall back to ``{name}_{variant}`` (the generic
    benchmark convention)."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PLUGINS: dict[str, ScenarioPlugin] = {}

# Lazy loaders: scenario name -> module whose import registers the plugin.
# Imports are deferred so optional scenario dependencies (browsergym, swebench,
# inspect-evals[osworld], scipy, ...) don't have to be installed to use Orbit.
_LOADERS: dict[str, str] = {
    "browserart": "orbit.scenarios.browser.browserart.task",
    "swe_bench": "orbit.scenarios.coding.swe_bench.task",
    "osworld": "orbit.scenarios.desktop.osworld.task",
    "osworld_benchmark": "orbit.scenarios.desktop.osworld.task",
    "redcode_gen": "orbit.scenarios.coding.redcode_gen.task",
    "code_ipi": "orbit.scenarios.coding.code_ipi.task",
    "bigcodebench": "orbit.scenarios.coding.bigcodebench.task",
    "agentharm": "orbit.scenarios.agentharm.task",
    "tau2": "orbit.scenarios.customer_service.tau2.task",
    "converse": "orbit.scenarios.customer_service.converse.task",
    "jira_ticket": "orbit.scenarios.cooperative_allocation.jira_ticket.task",
    "hospital": "orbit.scenarios.cooperative_allocation.hospital.task",
    "meeting_scheduling": "orbit.scenarios.cooperative_allocation.meeting_scheduling.task",
}


def _canonical(name: str) -> str:
    """Map a ``scenario.name`` to its plugin key.

    Most scenarios use a fixed name, but tau2 encodes the domain in the name
    (``tau2_airline``, ``tau2_retail``, ...) so all of those map to ``tau2``.
    """
    if name.startswith("tau2_"):
        return "tau2"
    return name


def register_scenario(plugin: ScenarioPlugin) -> ScenarioPlugin:
    """Register a scenario plugin. Call at scenario-module import time."""
    _PLUGINS[plugin.name] = plugin
    return plugin


def get_scenario(name: str | None) -> ScenarioPlugin | None:
    """Return the registered plugin for ``name``.

    Triggers a lazy import of the owning scenario module the first time a known
    name is requested. Returns ``None`` for an *unknown* name (no registered
    plugin) — the caller falls back to the generic path.

    Raises :class:`ScenarioImportError` if a *known* scenario's module fails to
    import. Scenario modules import optional deps lazily, so an import failure
    is a genuine bug (a missing extra surfaces later, from ``expand``); silently
    returning ``None`` here would run a declared scenario as a benign
    single-sample generic task with no error — the issue-#9 failure class.
    """
    if not name:
        return None
    key = _canonical(name)
    if key not in _PLUGINS and key in _LOADERS:
        try:
            importlib.import_module(_LOADERS[key])
        except Exception as exc:
            logger.error("Scenario plugin %r failed to import: %s", key, exc)
            raise ScenarioImportError(
                f"Scenario {key!r} is registered but its module "
                f"{_LOADERS[key]!r} failed to import ({exc!s}). Optional scenario "
                f"dependencies are imported lazily, so this is a bug in the "
                f"scenario module, not a missing extra."
            ) from exc
    return _PLUGINS.get(key)


def registered_scenarios() -> frozenset[str]:
    """Names of scenarios that have a dedicated plugin (lazy-loadable)."""
    return frozenset(_LOADERS) | frozenset(_PLUGINS)
