"""AgentHarm condition presets — 12 topology configurations.

Each builder takes an :class:`AgentHarmBehavior` plus scenario-level options
and returns a :class:`SetupConfig`. This diverges from other scenarios
(which typically return a per-sample-invariant SetupConfig template),
because AgentHarm resolves its tool list per sample from the behavior's
``target_functions``.

The 12 conditions:

    single_agent               Single agent, all target tools
    star_batch                 Star: orchestrator + 1 batch executor
    star_step                  Star: orchestrator + 1 step-by-step executor
    star_2_specialist          Star: orchestrator + 2 domain specialists
    star_3_specialist          Star: orchestrator + 3 domain specialists
    star_specialist            Star: orchestrator + 4 domain specialists
    mesh_round_robin           Mesh: 4 peer specialists, round-robin
    mesh_delegation            Mesh: 4 peer specialists, free delegation
    memory_own_reasoning       star_specialist + per-agent CoT memory
    memory_full                star_specialist + shared full memory
    mesh_delegation_cot        mesh_delegation + per-agent CoT memory
    mesh_delegation_full       mesh_delegation + shared full memory
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from orbit.configs.setup import (
    AgentMemoryAccess,
    AgentSpec,
    MemoryConfig,
    SetupConfig,
)
from orbit.configs.topology_presets import (
    mesh_delegation,
    mesh_round_robin,
    single_agent,
    star_orchestrator_workers,
)
from orbit.scenarios.agentharm.configs import (
    AgentHarmBehavior,
    AgentHarmToolMode,
)
from orbit.scenarios.agentharm.prompts import (
    EXECUTOR_BATCH_PROMPT,
    EXECUTOR_STEP_PROMPT,
    MESH_DELEGATION_PROMPT,
    MESH_ROUND_ROBIN_PROMPT,
    ORCH_BATCH_PROMPT,
    ORCH_SPECIALIST_PROMPT,
    ORCH_STEP_PROMPT,
    solo_system_prompt,
    specialist_prompt,
)
from orbit.scenarios.agentharm.tool_domains import (
    TOOL_DOMAINS,
    ToolDomain,
    domain_of,
    tools_in_domain,
)
from orbit.scenarios.agentharm.tool_setup import all_tool_names

_SPECIALTY_DESCRIPTIONS: dict[ToolDomain, tuple[str, str]] = {
    "comms": ("messaging, email, and social media", "email, SMS, WhatsApp, X/Twitter, Instagram, Reddit, forums, contacts"),
    "web": ("web search, browsing, and downloads", "Google/Bing/Onion search, webpage retrieval, file/image/software downloads, VPN"),
    "system": ("terminal, file operations, and media", "terminal, read/edit files, PDF conversion, Stable Diffusion, video/meme generation"),
    "data": ("finance, accounts, and databases", "bank login, PayPal/crypto transfers, purchases, database queries, credential stores"),
}

_DOMAIN_AGENT_NAMES: dict[ToolDomain, str] = {
    "comms": "comms_specialist",
    "web": "web_specialist",
    "system": "system_specialist",
    "data": "data_specialist",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry_key(split: str, tool_name: str) -> str:
    """Build the ``_TOOL_REGISTRY`` key for an AgentHarm tool."""
    return f"agentharm_{split}_{tool_name}"


def _behavior_tool_keys(
    behavior: AgentHarmBehavior,
    split: str,
) -> list[str]:
    """Registry keys for a behavior's ``target_functions`` in the given split."""
    return [_registry_key(split, fn) for fn in behavior.target_functions]


def _specialist_tool_keys(
    behavior: AgentHarmBehavior,
    *,
    split: str,
    domain: ToolDomain,
    tool_mode: AgentHarmToolMode,
    n_irrelevant: int,
    seed: int,
) -> list[str]:
    """Compute the tool name list for a specialist in ``domain``."""
    if split == "chat":
        return []

    if tool_mode == "full_domain":
        names = tools_in_domain(all_tool_names(split), domain)
        return [_registry_key(split, n) for n in names]

    # filtered: target functions in this domain + in-domain padding
    in_domain = [
        fn for fn in behavior.target_functions if domain_of(fn) == domain
    ]
    keys = [_registry_key(split, fn) for fn in in_domain]

    if n_irrelevant > 0:
        import random

        available = [
            n for n in tools_in_domain(all_tool_names(split), domain)
            if n not in in_domain
        ]
        rng = random.Random(f"{seed}:{behavior.id}:{domain}")
        k = min(n_irrelevant, len(available))
        for extra in rng.sample(available, k):
            keys.append(_registry_key(split, extra))

    return keys


def _specialist_worker_specs(
    behavior: AgentHarmBehavior,
    *,
    split: str,
    tool_mode: AgentHarmToolMode,
    n_irrelevant: int,
    seed: int,
    domains: tuple[ToolDomain, ...] = TOOL_DOMAINS,
) -> list[dict[str, Any]]:
    """Build ``worker_specs`` dicts for the requested specialist domains."""
    specs: list[dict[str, Any]] = []
    for dom in domains:
        specialty, tools_hint = _SPECIALTY_DESCRIPTIONS[dom]
        specs.append({
            "name": _DOMAIN_AGENT_NAMES[dom],
            "role": "specialist",
            "specialty": specialty,
            "system_prompt": specialist_prompt(specialty, tools_hint),
            "tools": _specialist_tool_keys(
                behavior,
                split=split,
                domain=dom,
                tool_mode=tool_mode,
                n_irrelevant=n_irrelevant,
                seed=seed,
            ),
        })
    return specs


def _merged_specialist_worker_specs(
    behavior: AgentHarmBehavior,
    *,
    split: str,
    tool_mode: AgentHarmToolMode,
    n_irrelevant: int,
    seed: int,
    groupings: list[tuple[ToolDomain, ...]],
) -> list[dict[str, Any]]:
    """Build specialist workers where each owns multiple domains."""
    specs: list[dict[str, Any]] = []
    for group in groupings:
        name = "_".join(_DOMAIN_AGENT_NAMES[d].removesuffix("_specialist") for d in group)
        name = f"{name}_specialist"
        specialties = ", ".join(_SPECIALTY_DESCRIPTIONS[d][0] for d in group)
        tools_hints = "; ".join(_SPECIALTY_DESCRIPTIONS[d][1] for d in group)
        tool_keys: list[str] = []
        for d in group:
            tool_keys.extend(
                _specialist_tool_keys(
                    behavior,
                    split=split,
                    domain=d,
                    tool_mode=tool_mode,
                    n_irrelevant=n_irrelevant,
                    seed=seed,
                )
            )
        specs.append({
            "name": name,
            "role": "specialist",
            "specialty": specialties,
            "system_prompt": specialist_prompt(specialties, tools_hints),
            "tools": tool_keys,
        })
    return specs


# ---------------------------------------------------------------------------
# Per-condition builders
# ---------------------------------------------------------------------------


def _build_single_agent(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    split = opts["split"]
    tools = _behavior_tool_keys(behavior, split) if split != "chat" else []
    return single_agent(
        name="agentharm_agent",
        role="executor",
        system_prompt=solo_system_prompt(
            system_prompt_variant=opts.get("system_prompt_variant", "default"),
            prompt_technique=opts.get("prompt_technique", "standard"),
        ),
        tools=tools,
    )


def _build_star_batch(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    split = opts["split"]
    tools = _behavior_tool_keys(behavior, split) if split != "chat" else []
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_BATCH_PROMPT,
        worker_specs=[{
            "name": "executor",
            "role": "executor",
            "system_prompt": EXECUTOR_BATCH_PROMPT,
            "tools": tools,
        }],
        properties={"condition_type": "star_batch", "execution_style": "batch"},
    )


def _build_star_step(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    split = opts["split"]
    tools = _behavior_tool_keys(behavior, split) if split != "chat" else []
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_STEP_PROMPT,
        worker_specs=[{
            "name": "executor",
            "role": "executor",
            "system_prompt": EXECUTOR_STEP_PROMPT,
            "tools": tools,
        }],
        properties={"condition_type": "star_step", "execution_style": "step"},
    )


def _build_star_specialist(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    workers = _specialist_worker_specs(behavior, **_specialist_opts(opts))
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_PROMPT.format(
            specialist_list=_format_specialist_list(workers),
        ),
        worker_specs=workers,
        properties={"condition_type": "star_specialist"},
    )


def _build_star_2_specialist(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    workers = _merged_specialist_worker_specs(
        behavior,
        **_specialist_opts(opts),
        groupings=[("comms", "web"), ("system", "data")],
    )
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_PROMPT.format(
            specialist_list=_format_specialist_list(workers),
        ),
        worker_specs=workers,
        properties={"condition_type": "star_2_specialist"},
    )


def _build_star_3_specialist(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    workers = _merged_specialist_worker_specs(
        behavior,
        **_specialist_opts(opts),
        groupings=[("comms",), ("web", "system"), ("data",)],
    )
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_PROMPT.format(
            specialist_list=_format_specialist_list(workers),
        ),
        worker_specs=workers,
        properties={"condition_type": "star_3_specialist"},
    )


def _build_mesh_round_robin(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    specs = _mesh_specs(
        behavior, MESH_ROUND_ROBIN_PROMPT, **_specialist_opts(opts),
    )
    return mesh_round_robin(
        agent_specs=specs,
        properties={"condition_type": "mesh_round_robin"},
    )


def _build_mesh_delegation(
    behavior: AgentHarmBehavior, **opts: Any,
) -> SetupConfig:
    specs = _mesh_specs(
        behavior, MESH_DELEGATION_PROMPT, **_specialist_opts(opts),
    )
    return mesh_delegation(
        agent_specs=specs,
        properties={"condition_type": "mesh_delegation"},
    )


def _mesh_specs(
    behavior: AgentHarmBehavior,
    base_prompt: str,
    *,
    split: str,
    tool_mode: AgentHarmToolMode,
    n_irrelevant: int,
    seed: int,
) -> list[dict[str, Any]]:
    specs = _specialist_worker_specs(
        behavior,
        split=split,
        tool_mode=tool_mode,
        n_irrelevant=n_irrelevant,
        seed=seed,
    )
    for s in specs:
        s["system_prompt"] = (
            base_prompt + "\n\n" + s["system_prompt"]
        )
    return specs


def _build_memory_condition(
    behavior: AgentHarmBehavior,
    *,
    level: str,
    base_builder: Callable[..., SetupConfig],
    condition_name: str,
    **opts: Any,
) -> SetupConfig:
    base = base_builder(behavior, **opts)
    memory = _memory_for_level(level, [a.name for a in base.agents])
    return base.model_copy(update={
        "memory": memory,
        "properties": {**base.properties, "condition_type": condition_name},
    })


def _memory_for_level(
    level: str,
    agent_names: list[str],
) -> MemoryConfig:
    if level == "own_reasoning":
        return MemoryConfig(
            shared=False,
            agent_memory_access=[
                AgentMemoryAccess(
                    agent_name=name,
                    own_action_history=True,
                    own_cot=True,
                )
                for name in agent_names
            ],
        )
    if level == "full":
        return MemoryConfig(
            shared=True,
            shared_groups=[agent_names],
            agent_memory_access=[
                AgentMemoryAccess(
                    agent_name=name,
                    own_action_history=True,
                    own_cot=True,
                    shared_action_history=True,
                    shared_cot=True,
                    nl_instructions_accumulated=True,
                )
                for name in agent_names
            ],
        )
    raise ValueError(f"Unknown memory level: {level!r}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _specialist_opts(opts: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": opts["split"],
        "tool_mode": opts.get("tool_mode", "filtered"),
        "n_irrelevant": opts.get("n_irrelevant", 0),
        "seed": opts.get("seed", 0),
    }


def _format_specialist_list(worker_specs: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {w['name']}: {w.get('specialty', w.get('role', ''))}"
        for w in worker_specs
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_CONDITION_BUILDERS: dict[str, Callable[..., SetupConfig]] = {
    "single_agent": _build_single_agent,
    "star_batch": _build_star_batch,
    "star_step": _build_star_step,
    "star_2_specialist": _build_star_2_specialist,
    "star_3_specialist": _build_star_3_specialist,
    "star_specialist": _build_star_specialist,
    "mesh_round_robin": _build_mesh_round_robin,
    "mesh_delegation": _build_mesh_delegation,
    "memory_own_reasoning": lambda b, **o: _build_memory_condition(
        b, level="own_reasoning", base_builder=_build_star_specialist,
        condition_name="memory_own_reasoning", **o,
    ),
    "memory_full": lambda b, **o: _build_memory_condition(
        b, level="full", base_builder=_build_star_specialist,
        condition_name="memory_full", **o,
    ),
    "mesh_delegation_cot": lambda b, **o: _build_memory_condition(
        b, level="own_reasoning", base_builder=_build_mesh_delegation,
        condition_name="mesh_delegation_cot", **o,
    ),
    "mesh_delegation_full": lambda b, **o: _build_memory_condition(
        b, level="full", base_builder=_build_mesh_delegation,
        condition_name="mesh_delegation_full", **o,
    ),
}


def get_condition_setup(
    condition: str,
    behavior: AgentHarmBehavior,
    *,
    split: str,
    tool_mode: AgentHarmToolMode = "filtered",
    n_irrelevant: int = 0,
    seed: int = 0,
    system_prompt_variant: str = "default",
    prompt_technique: str = "standard",
) -> SetupConfig:
    """Build the ``SetupConfig`` for ``condition`` on a single behavior."""
    if condition not in _CONDITION_BUILDERS:
        raise ValueError(
            f"Unknown condition: {condition!r}. "
            f"Available: {sorted(_CONDITION_BUILDERS)}"
        )
    return _CONDITION_BUILDERS[condition](
        behavior,
        split=split,
        tool_mode=tool_mode,
        n_irrelevant=n_irrelevant,
        seed=seed,
        system_prompt_variant=system_prompt_variant,
        prompt_technique=prompt_technique,
    )


def list_conditions() -> list[str]:
    """Return the sorted list of valid condition names."""
    return sorted(_CONDITION_BUILDERS)
