"""AgentHarm task entry point.

Registered via :mod:`orbit._registry` so ``inspect eval orbit/agentharm``
is discoverable from the Inspect CLI.
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from orbit.dataset.sample_factory import build_sample
from orbit.scenarios.agentharm.config_builder import (
    build_experiment_configs_from_scenario,
)
from orbit.scenarios.agentharm.configs import AgentHarmScenarioConfig
from orbit.scenarios.agentharm.scorer import agentharm_scorer

logger = logging.getLogger(__name__)


@task
def agentharm(
    split: str = "harmful",
    partition: str = "test_public",
    condition: str = "single_agent",
    categories: str | None = None,
    behavior_ids: str | None = None,
    detailed_prompt: bool | None = None,
    hint_included: bool | None = None,
    max_samples: int | None = None,
    seed: int = 0,
    judge_model: str = "openai/gpt-4.1",
    n_irrelevant_tools: int = 0,
    tool_mode: str = "filtered",
    prompt_technique: str = "standard",
    system_prompt_variant: str = "default",
    max_turns: int = 10,
    max_time_seconds: float = 300.0,
    data_path: str | None = None,
    orchestrator: str = "v1",
) -> Task:
    """AgentHarm harmful tool-use safety benchmark.

    Orbit port of AgentHarm (Andriushchenko et al., ICLR 2025). Every
    tool is mocked — no sandbox, no network, no filesystem side effects.

    Args:
        split: ``harmful`` (default), ``benign``, or ``chat``.
        partition: ``test_public`` (default, 176 samples) or ``validation``
            (32 samples). The ``chat`` split has 44 / 8 samples.
        condition: Topology preset — see
            :func:`orbit.scenarios.agentharm.condition_presets.list_conditions`.
            Default ``single_agent``.
        categories: Comma-separated harm-category filter
            (e.g. ``"Fraud,Cybercrime"``).
        behavior_ids: Comma-separated behavior-ID filter.
        detailed_prompt: Keep only behaviors with matching ``detailed_prompt``.
        hint_included: Keep only behaviors with matching ``hint_included``.
        max_samples: Cap the number of samples after filtering.
        seed: Seed for deterministic subsampling + irrelevant-tool selection.
        judge_model: Model used for both the refusal judge and any grader
            that invokes ``semantic_judge`` internally.
        n_irrelevant_tools: Pad each agent's tool list with this many
            distractor tools (upstream default: 0).
        tool_mode: ``filtered`` (specialist tool lists are per-sample target
            functions intersected with the specialist's domain, default)
            or ``full_domain`` (each specialist sees their full domain
            tool set regardless of the sample's target functions).
        prompt_technique: Upstream prompt-technique suffix —
            ``standard`` / ``cot`` / ``react`` / ``refusal``.
        system_prompt_variant: ``default`` or ``llama-3.1``.
        max_turns: Maximum agent turns per sample. For mesh conditions that
            use round-robin scheduling, this is the number of full cycles
            across all peers — effective scheduler ticks = ``max_turns ×
            len(peers)``.
        max_time_seconds: Maximum wall-clock time per sample.
        data_path: Override for the pinned ``data/`` directory.
    """
    category_list = (
        tuple(c.strip() for c in categories.split(",") if c.strip())
        if categories else None
    )
    id_list = (
        tuple(b.strip() for b in behavior_ids.split(",") if b.strip())
        if behavior_ids else None
    )

    scenario_config = AgentHarmScenarioConfig(
        data_path=data_path,
        split=split,  # type: ignore[arg-type]
        partition=partition,  # type: ignore[arg-type]
        categories=category_list,
        behavior_ids=id_list,
        detailed_prompt=detailed_prompt,
        hint_included=hint_included,
        max_samples=max_samples,
        seed=seed,
        judge_model=judge_model,
        n_irrelevant_tools=n_irrelevant_tools,
        tool_mode=tool_mode,  # type: ignore[arg-type]
        prompt_technique=prompt_technique,  # type: ignore[arg-type]
        system_prompt_variant=system_prompt_variant,  # type: ignore[arg-type]
        max_turns=max_turns,
        max_time_seconds=max_time_seconds,
    )

    configs = build_experiment_configs_from_scenario(
        scenario_config=scenario_config,
        condition=condition,
    )

    samples = [build_sample(c, sample_id=c.name) for c in configs]
    if not samples:
        logger.warning(
            "AgentHarm produced 0 samples — check split/partition/filters"
        )

    from orbit.scorers.security_scorer import security_scorer
    from orbit.solvers.orchestrator import mas_orchestrator
    from orbit.solvers.orchestrator_v2 import mas_orchestrator_v2

    return Task(
        dataset=MemoryDataset(samples) if samples else MemoryDataset([]),
        solver=mas_orchestrator_v2() if orchestrator == "v2" else mas_orchestrator(),
        scorer=[agentharm_scorer(judge_model), security_scorer()],
    )
