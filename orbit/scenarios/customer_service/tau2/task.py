"""
τ²-Bench task entry point.

Registered with Inspect via ``orbit/_registry.py``. Produces one
Inspect ``Sample`` per tau2 task and wires up the tau2 scorer on top
of the generic security scorer.

Supports ``mode='solo'`` (PR 1, Orbit-specific airline-only baseline)
and ``mode='dual_control'`` (PR 2, upstream-faithful user simulator ↔
assistant loop) against the airline, retail, and telecom domains.
"""

from __future__ import annotations

import copy
import logging

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import Solver, TaskState, solver
from inspect_ai.util import store_as

from orbit.dataset.metadata import MASMetadata
from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.customer_service.tau2.config_builder import build_experiment_config
from orbit.scenarios.customer_service.tau2.configs import Tau2ScenarioConfig, Tau2Task
from orbit.scenarios.customer_service.tau2.dataset_builder import (
    filter_tasks,
    load_initial_db,
    load_initial_user_db,
    load_policy,
    load_tau2_tasks,
)
from orbit.scenarios.customer_service.tau2.scorer import (
    _apply_env_function_calls,
    _initialization_calls,
    _load_db_model,
    _load_user_db_model,
    _domain_has_user_db,
)
from orbit.scenarios.customer_service.tau2.scorer import pass_hat_k, tau2_scorer
from orbit.scenarios.customer_service.tau2.state import DomainState
from orbit.scorers.security_scorer import security_scorer
from orbit.solvers.orchestrator import mas_orchestrator
from orbit.solvers.orchestrator_v2 import mas_orchestrator_v2

logger = logging.getLogger(__name__)


def _parse_csv(value: object) -> list[str] | None:
    """Normalize a ``task_ids`` argument into a list of string IDs.

    Accepts ``None``, ``str`` ("1,3,7"), ``int`` (1), or ``list``
    (["1", 3]). Inspect's ``-T`` flag coerces command-line values
    aggressively — ``-T task_ids=1,3`` becomes a Python list, ``-T
    task_ids=1`` becomes an int — so we tolerate all three shapes
    rather than ``str.split``-crashing.
    """
    if value is None or value == "":
        return None
    if isinstance(value, list):
        out = [str(v).strip() for v in value]
        return [v for v in out if v] or None
    if isinstance(value, (int, float)):
        return [str(value)]
    text = str(value)
    return [v.strip() for v in text.split(",") if v.strip()] or None


def _build_setup_solver(
    domain: str,
    initial_db: dict,
    initial_user_db: dict,
    scenario_config: Tau2ScenarioConfig,
) -> Solver:
    """Return a setup solver that seeds ``DomainState`` for each sample.

    Each sample gets an independent deep copy of the vendored initial
    DBs so tool mutations in one sample cannot leak into another. The
    setup solver also extracts the task from ``MASMetadata`` and
    replays the task's ``initial_state.initialization_actions`` against
    the live environments, matching upstream's pre-loop setup. Telecom
    tasks lean heavily on this (``set_user_info`` / ``set_user_location``
    / ``turn_roaming_off`` etc. all fire here).

    In ``cross_domain_handoff`` mode the solver additionally
    materialises every other domain's DB into ``DomainState.dbs`` so
    each per-domain specialist agent's tools can load their own DB.
    Initialization actions still run only against the task's declared
    domain (tasks are single-domain; no cross-domain gold state).
    """

    is_cross_domain = scenario_config.topology == "cross_domain_handoff"

    @solver
    def tau2_setup() -> Solver:
        async def solve(state: TaskState, generate) -> TaskState:
            ds = store_as(DomainState)
            ds.domain = domain

            if is_cross_domain:
                # Seed every domain's agent-side DB so cross_domain
                # specialists can load their own. Initialization
                # actions still only run against the primary domain.
                all_dbs: dict[str, dict] = {}
                all_user_dbs: dict[str, dict] = {}
                # Import here to keep the single-domain path free of
                # the extra dataset_builder calls.
                from orbit.scenarios.customer_service.tau2.dataset_builder import (
                    load_initial_db as _load_all_initial_db,
                    load_initial_user_db as _load_all_initial_user_db,
                )
                for d in ("airline", "retail", "telecom"):
                    per_domain_config = scenario_config.model_copy(
                        update={"domain": d}
                    )
                    if d == domain:
                        all_dbs[d] = copy.deepcopy(initial_db)
                        if _domain_has_user_db(d):
                            all_user_dbs[d] = copy.deepcopy(initial_user_db)
                    else:
                        all_dbs[d] = _load_all_initial_db(per_domain_config)
                        if _domain_has_user_db(d):
                            all_user_dbs[d] = _load_all_initial_user_db(
                                per_domain_config
                            )
                ds.dbs = all_dbs
                ds.user_dbs = all_user_dbs
                # Keep the single-domain surface pointing at the
                # primary domain too so legacy reads still resolve.
                ds.db = all_dbs[domain]
                ds.user_db = all_user_dbs.get(domain, {})
            else:
                ds.db = copy.deepcopy(initial_db)
                if _domain_has_user_db(domain):
                    ds.user_db = copy.deepcopy(initial_user_db)
                else:
                    ds.user_db = {}
                ds.dbs = {}
                ds.user_dbs = {}

            # Pull the per-sample Tau2Task out of metadata and replay
            # initialization_actions against typed shadow DBs, then
            # flush them back into DomainState so the agent's first
            # tool call sees the right state.
            try:
                meta = state.metadata_as(MASMetadata)
            except Exception:  # noqa: BLE001
                return state
            tau2_payload = meta.experiment.metadata.get("tau2_task")
            if not tau2_payload:
                return state
            tau2_task = Tau2Task.model_validate(tau2_payload)
            init_calls = _initialization_calls(tau2_task)
            if not init_calls:
                return state

            try:
                agent_db = _load_db_model(domain, ds.get_db(domain))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "tau2 setup: failed to load agent DB for %s/%s: %s",
                    domain, tau2_task.id, e,
                )
                return state
            user_db = None
            if _domain_has_user_db(domain):
                try:
                    user_db = _load_user_db_model(
                        domain, ds.get_user_db(domain)
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "tau2 setup: failed to load user DB for %s/%s: %s",
                        domain, tau2_task.id, e,
                    )
                    return state

            errors: list[str] = []
            _apply_env_function_calls(
                tau2_task, agent_db, user_db, init_calls, errors
            )
            if errors:
                logger.warning(
                    "tau2 setup: %d init-action error(s) for %s/%s: %s",
                    len(errors), domain, tau2_task.id, errors,
                )
            ds.set_db(domain, agent_db.model_dump(mode="json"))
            if user_db is not None:
                ds.set_user_db(domain, user_db.model_dump(mode="json"))
            return state

        return solve

    return tau2_setup()


_SUPPORTED_DOMAINS = ("airline", "retail", "telecom")
_SUPPORTED_MODES = ("solo", "dual_control")
_SUPPORTED_TOPOLOGIES = (
    "solo",
    "dual_control",
    "supervisor_specialist",
    "tiered_escalation",
    "mesh_committee",
    "dual_control_review",
    "cross_domain_handoff",
)


@task
def tau2(
    domain: str = "airline",
    task_ids: str | None = None,
    max_tasks: int | None = None,
    seed: int | None = None,
    mode: str = "dual_control",
    topology: str = "dual_control",
    judge_model: str = "openai/gpt-4.1",
    max_turns: int = 100,
    max_time_seconds: float = 600.0,
    data_path: str | None = None,
    orchestrator: str = "v1",
) -> Task:
    """τ²-Bench dual-control tool-use benchmark.

    Supports three domains and two execution modes:

    - ``domain='airline'`` — 50 reservation tasks, reward_basis
      ``[DB, COMMUNICATE]``. The only domain where ``mode='solo'`` is
      accepted (as an Orbit-specific baseline; upstream rejects airline
      solo mode).
    - ``domain='retail'`` — 114 order-management tasks, reward_basis
      ``[DB, NL_ASSERTION]`` on 112/114 tasks. Exercises the LLM judge
      ported in PR 2.
    - ``domain='telecom'`` — 20 tasks from ``tasks_small.json`` (the
      regression subset upstream uses for most paper numbers;
      ``tasks.json`` with 2285 tasks is not vendored — point
      ``data_path`` at a local copy to run it). Reward_basis includes
      ``ENV_ASSERTION`` (evaluated against the agent's final shadow
      env) and the user simulator gets a 30-tool device-diagnostics
      tool set.

    Execution modes:

    - ``mode='dual_control'`` (default): round-robin user simulator ↔
      assistant loop with ``peer_messages`` observation and the
      ``check_tau2_stop_sentinels`` halt condition (``###STOP###`` /
      ``###TRANSFER###`` / ``###OUT-OF-SCOPE###``). Upstream-faithful.
    - ``mode='solo'`` (airline only): single-agent baseline with the
      user scenario rendered into the assistant's system prompt as a
      ticket. Orbit-specific.

    The task entry point also runs the NL-assertions LLM judge
    (retail) and evaluates ENV_ASSERTION-based rewards (telecom), and
    every run produces pass^1 / pass^2 / pass^4 via the ``pass_hat_k``
    metric (run with ``--epochs 4`` to populate pass^4).

    Args:
        domain: ``'airline'``, ``'retail'``, or ``'telecom'``.
        task_ids: Comma-separated tau2 task IDs to include
            (e.g. ``'1,2,3'``).
        max_tasks: Maximum number of tasks to evaluate.
        seed: Random seed for deterministic task sampling when
            ``max_tasks`` is set.
        mode: ``'solo'`` (airline only) or ``'dual_control'`` (default).
        judge_model: Model used for NL-assertion grading. Only invoked
            when the task's ``reward_basis`` includes ``NL_ASSERTION``.
        max_turns: Maximum orchestrator turns per task.
        max_time_seconds: Maximum wall-clock time per task.
        data_path: Override for the vendored data directory. Set this
            to a local copy of upstream's full telecom ``tasks.json``
            if you want to run the 2285-task set.
    """
    if domain not in _SUPPORTED_DOMAINS:
        raise ValueError(
            f"Domain '{domain}' is not supported. Expected one of "
            f"{_SUPPORTED_DOMAINS}."
        )
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Mode '{mode}' is not supported. Expected one of "
            f"{_SUPPORTED_MODES}."
        )
    if topology not in _SUPPORTED_TOPOLOGIES:
        raise ValueError(
            f"Topology '{topology}' is not supported. Expected one of "
            f"{_SUPPORTED_TOPOLOGIES}."
        )
    if mode == "solo" and domain != "airline":
        raise ValueError(
            f"Solo mode is an Orbit-specific baseline only wired for "
            f"airline (upstream rejects airline-solo too). Use "
            f"mode='dual_control' for {domain!r}."
        )
    if mode == "solo" and topology != "solo":
        raise ValueError(
            f"mode='solo' only supports topology='solo'. Got "
            f"topology={topology!r}."
        )
    if mode == "dual_control" and topology == "solo":
        raise ValueError(
            "topology='solo' requires mode='solo'. Use "
            "topology='dual_control' for dual-control runs."
        )

    scenario_config = Tau2ScenarioConfig(
        domain=domain,  # type: ignore[arg-type]
        task_ids=_parse_csv(task_ids),
        max_tasks=max_tasks,
        seed=seed,
        mode=mode,  # type: ignore[arg-type]
        topology=topology,  # type: ignore[arg-type]
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
        data_path=data_path,
    )

    tasks = load_tau2_tasks(scenario_config)
    tasks = filter_tasks(tasks, scenario_config)

    policy_text = load_policy(scenario_config)
    initial_db = load_initial_db(scenario_config)
    initial_user_db = load_initial_user_db(scenario_config)

    samples = []
    for t in tasks:
        config = build_experiment_config(
            task=t, scenario_config=scenario_config, policy_text=policy_text
        )
        samples.append(build_sample(config, sample_id=config.name))

    if not samples:
        logger.warning("No tau2 samples built — check filters and domain")

    setup_solver = _build_setup_solver(
        domain=domain,
        initial_db=initial_db,
        initial_user_db=initial_user_db,
        scenario_config=scenario_config,
    )

    # Default to 1 epoch with pass^1 / pass^2 / pass^4 reducers wired
    # in. When the user runs with ``--epochs N`` Inspect overrides the
    # epoch count but preserves the reducer list, so passing
    # ``--epochs 4`` yields directly paper-comparable pass^4 numbers.
    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        setup=setup_solver,
        solver=mas_orchestrator_v2() if orchestrator == "v2" else mas_orchestrator(),
        scorer=[
            tau2_scorer(data_path=data_path, judge_model=judge_model),
            security_scorer(),
        ],
        epochs=Epochs(1, [pass_hat_k(1), pass_hat_k(2), pass_hat_k(4)]),
    )
