"""ConVerse ``@task`` entry point.

Thin adapter: parses CLI / YAML parameters, resolves a topology via
:mod:`.condition_presets` (or a user-supplied YAML), builds one
:class:`ExperimentConfig` per :class:`ConverseSample`, and wires the
scenario's setup solver + scorer onto the resulting Inspect ``Task``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import Solver, TaskState, solver
from inspect_ai.util import store_as

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.customer_service.converse.condition_presets import (
    apply_memory_preset,
    get_condition_setup,
)
from orbit.scenarios.customer_service.converse.config_builder import build_experiment_configs
from orbit.scenarios.customer_service.converse.configs import (
    ConverseAttackMode,
    ConverseEnvironmentState,
    ConverseScenarioConfig,
)
from orbit.scenarios.customer_service.converse.dataset_builder import build_samples
from orbit.scenarios.customer_service.converse.scorer import converse_scorer
from orbit.solvers.runtime_state import MemoryPoisonLog

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_VALID_ATTACK_MODES: tuple[ConverseAttackMode, ...] = (
    "benign",
    "privacy",
    "security",
    "both",
)


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(v.strip() for v in value.split(",") if v.strip())
    return parts or None


@solver
def converse_setup() -> Solver:
    """Populate :class:`ConverseEnvironmentState` from sample metadata.

    Runs once per sample before the orchestrator loop. Reads the
    ``converse_env_state`` metadata payload that ``config_builder`` stashed
    and writes it into the per-sample Store instance so tools can read it.

    Also seeds :class:`MemoryPoisonLog` from ``converse_memory_poison_seed``
    metadata when cross-session runners carry poisoned state forward.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        meta = state.metadata or {}
        payload = meta.get("converse_env_state")
        if not isinstance(payload, dict):
            exp = meta.get("experiment")
            if isinstance(exp, dict):
                inner_meta = exp.get("metadata", {})
                if isinstance(inner_meta, dict):
                    payload = inner_meta.get("converse_env_state")
        if not isinstance(payload, dict):
            logger.warning(
                "converse_setup: no converse_env_state metadata found — "
                "tools will see an empty environment."
            )
        else:
            env = store_as(ConverseEnvironmentState)
            env.persona_id = str(payload.get("persona_id", ""))
            env.domain = str(payload.get("domain", ""))
            env.persona_name = str(payload.get("persona_name", ""))
            env.profile_text = str(payload.get("profile_text", ""))
            env.general_info_text = str(payload.get("general_info_text", ""))
            env.toolkit_sections = dict(payload.get("toolkit_sections", {}) or {})
            env.options_text = str(payload.get("options_text", ""))
            env.task_base_request = str(payload.get("task_base_request", ""))
            env.task_implicit_expectations = dict(
                payload.get("task_implicit_expectations", {}) or {}
            )
            env.task_profile_summary = str(payload.get("task_profile_summary", ""))
            # Reset per-run state in case this sample is being rerun.
            env.actions = []
            env.first_leak_turn = None
            env.cumulative_leak_per_turn = []

        seed_entries = meta.get("converse_memory_poison_seed")
        if not isinstance(seed_entries, list) or not seed_entries:
            exp = meta.get("experiment")
            if isinstance(exp, dict):
                inner_meta = exp.get("metadata", {})
                if isinstance(inner_meta, dict):
                    seed_entries = inner_meta.get("converse_memory_poison_seed")
        if isinstance(seed_entries, list) and seed_entries:
            poison_log = store_as(MemoryPoisonLog)
            for entry in seed_entries:
                if not isinstance(entry, dict):
                    continue
                poison_log.entries.append({
                    "target_memory_group": list(
                        entry.get("target_memory_group") or []
                    ),
                    "payload": str(entry.get("payload") or ""),
                    "mode": entry.get("mode", "append"),
                    "injected_at_turn": int(entry.get("injected_at_turn", 0) or 0),
                    "attack_name": str(
                        entry.get("attack_name", "cross_session_seed")
                    ),
                    "origin": "seeded_from_prior_session",
                    "consumed_by": [],
                })
        return state

    return solve


@task
def converse_safety(
    condition: str = "paper_star",
    topology_file: str | None = None,
    attack_modes: str = "benign",
    domains: str | None = None,
    persona_ids: str | None = None,
    data_categories: str | None = None,
    security_categories: str | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    judge_model: str = "openai/gpt-4.1",
    max_turns: int = 10,
    max_time_seconds: float = 300.0,
    attack_preset: str | None = None,
    defense_preset: str | None = None,
    memory_preset: str | None = None,
    data_path: str | None = None,
    memory_poison_seed_file: str | None = None,
    memory_poison_target_group: str | None = None,
    memory_poison_payload: str | None = None,
    memory_poison_mode: str = "append",
    orchestrator: str = "v1",
) -> Task:
    """ConVerse contextual-safety benchmark task.

    Args:
        condition: Named topology preset from :mod:`condition_presets`
            (``paper_star``, ``benign_pair``, ``guarded_star``, ``single_agent``,
            ``split_planner``, ``dual_external``, ``hierarchical``,
            ``mesh_trio``, ``specialist_trio``).
        topology_file: Path to a user-supplied :class:`SetupConfig` YAML.
            Overrides ``condition`` when set. Must satisfy the ConVerse role
            contract.
        attack_modes: Comma-separated list from ``benign,privacy,security,both``.
        memory_preset: Named memory preset applied as an overlay on the
            resolved topology (``isolated``, ``assistant_environment_shared``,
            ``assistant_cot_leaked``, ``goal_hidden_from_external``).
            Orthogonal to ``condition`` — use any topology × any memory.
        domains: Comma-separated domain filter (``travel,real_estate,insurance``).
        persona_ids: Comma-separated persona id filter.
        data_categories: Privacy-attack filter (``unrelated,related_private,related_useful``).
        security_categories: Security-attack filter (upstream sub-taxonomy keys).
        max_samples: Cap materialized samples per attack mode after filtering.
        seed: Random seed for max_samples subsampling.
        judge_model: LLM judge model for utility and security scoring.
        max_turns: Per-agent turn ceiling (scaled up by agent count inside).
        max_time_seconds: Wall-clock ceiling per sample.
        attack_preset: Optional additional attack preset — not required for
            paper reproduction (per-sample attacks are built automatically).
        defense_preset: Named defense preset from :mod:`presets`.
        data_path: Override the vendored data path.
        memory_poison_seed_file: Optional JSON file containing poisoned memory
            state dumped from a previous session. Used by the cross-session
            runner to carry session-N poisoning forward into session N+1.
            Format: ``{"<persona_id>": [<poison_entry>, ...], ...}``.
        memory_poison_target_group: Comma-separated agent names for a
            per-sample memory-poisoning attack (session-1 runner). When set,
            a ``memory_poisoning`` attack is attached to every sample.
        memory_poison_payload: Payload text for the poisoning attack.
            Required when ``memory_poison_target_group`` is set.
        memory_poison_mode: ``append`` (default), ``prepend`` or ``replace``.
    """
    # --- Parse parameters ------------------------------------------------
    modes_tuple = _parse_csv(attack_modes) or ("benign",)
    for mode in modes_tuple:
        if mode not in _VALID_ATTACK_MODES:
            raise ValueError(
                f"Unknown attack_mode {mode!r}. Valid: {_VALID_ATTACK_MODES}"
            )

    scenario_config = ConverseScenarioConfig(
        data_path=data_path,
        domains=_parse_csv(domains),  # type: ignore[arg-type]
        persona_ids=_parse_csv(persona_ids),
        attack_modes=modes_tuple,  # type: ignore[arg-type]
        data_categories=_parse_csv(data_categories),  # type: ignore[arg-type]
        security_categories=_parse_csv(security_categories),
        max_samples=max_samples,
        seed=seed,
        judge_model=judge_model,
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
    )

    # --- Resolve topology ------------------------------------------------
    if topology_file:
        import yaml

        from orbit.configs.setup import SetupConfig

        with open(topology_file) as fh:
            topo_data = yaml.safe_load(fh)
        setup = SetupConfig(**topo_data)
    else:
        setup = get_condition_setup(condition)

    # --- Optional memory-preset overlay ----------------------------------
    if memory_preset:
        setup = apply_memory_preset(setup, memory_preset)

    # --- Optional preset attacks/defenses --------------------------------
    preset_attacks: list | None = None
    preset_defenses = None

    # Optional inline memory-poisoning attack (used by cross-session runner).
    if memory_poison_target_group:
        if not memory_poison_payload:
            raise ValueError(
                "memory_poison_payload must be set when "
                "memory_poison_target_group is provided"
            )
        if memory_poison_mode not in ("append", "prepend", "replace"):
            raise ValueError(
                f"memory_poison_mode must be append|prepend|replace, "
                f"got {memory_poison_mode!r}"
            )
        from orbit.configs.attack import AttackConfig as _AttackConfig

        # Inspect parses comma-separated -T values into a list automatically;
        # accept both list and raw string forms.
        if isinstance(memory_poison_target_group, str):
            target_list = [
                s.strip()
                for s in memory_poison_target_group.split(",")
                if s.strip()
            ]
        else:
            target_list = [
                str(s).strip()
                for s in memory_poison_target_group
                if str(s).strip()
            ]
        preset_attacks = [
            _AttackConfig(
                name="cross_session_memory_poison",
                attack_type="memory_poisoning",
                payload=memory_poison_payload,
                properties={
                    "target_memory_group": target_list,
                    "poison_payload": memory_poison_payload,
                    "poison_mode": memory_poison_mode,
                },
            )
        ]

    if attack_preset:
        # No default attack preset ships yet — the paper's per-sample attacks
        # are built automatically from sample metadata. This hook stays open
        # for non-paper experiments.
        raise ValueError(
            f"Unknown ConVerse attack preset {attack_preset!r}. "
            "Per-sample paper attacks are applied automatically; additional "
            "presets have not been registered yet."
        )
    if defense_preset:
        from orbit.scenarios.customer_service.converse.presets import get_defense_preset

        preset_defenses = get_defense_preset(defense_preset)

    # --- Build samples + experiment configs ------------------------------
    samples_data = build_samples(scenario_config)

    memory_poison_seeds: dict[str, list[dict]] | None = None
    if memory_poison_seed_file:
        import json
        from pathlib import Path as _Path

        raw = json.loads(_Path(memory_poison_seed_file).read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"memory_poison_seed_file {memory_poison_seed_file!r} must "
                f"be a JSON object mapping persona_id -> list of entries"
            )
        memory_poison_seeds = {
            str(k): list(v or []) for k, v in raw.items()
        }

    configs = build_experiment_configs(
        samples=samples_data,
        setup=setup,
        scenario_config=scenario_config,
        attacks=preset_attacks,
        defenses=preset_defenses,
        memory_poison_seeds=memory_poison_seeds,
    )

    inspect_samples = []
    for cfg in configs:
        inspect_samples.append(build_sample(cfg, sample_id=cfg.name))

    if not inspect_samples:
        logger.warning(
            "ConVerse: no samples materialized. attack_modes=%s, filters=%s",
            modes_tuple,
            {
                "domains": domains,
                "persona_ids": persona_ids,
                "data_categories": data_categories,
                "security_categories": security_categories,
            },
        )

    # --- Assemble Task ---------------------------------------------------
    from orbit.scorers.security_scorer import security_scorer
    from orbit.solvers.orchestrator import mas_orchestrator
    from orbit.solvers.orchestrator_v2 import mas_orchestrator_v2

    return Task(
        dataset=MemoryDataset(inspect_samples),
        setup=converse_setup(),
        solver=mas_orchestrator_v2() if orchestrator == "v2" else mas_orchestrator(),
        scorer=[converse_scorer(judge_model=judge_model), security_scorer()],
    )
