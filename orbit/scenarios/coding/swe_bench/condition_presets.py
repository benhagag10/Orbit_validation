"""SWE-Bench coding agent condition presets.

Bridge between general topology presets and coding-agent-specific details.
Each factory function calls a general topology preset from
:mod:`orbit.configs.topology_presets` and injects coding-specific
agent names, prompts, and tools.

Mirrors the BrowserART condition presets structure with domain-appropriate
specialists (implementation, debugging, testing, review) instead of
browser-specific specialists (click, fill, scroll, navigate).

The 13 conditions:

    single_agent              Single agent
    star_batch / _relaxed     Star: orchestrator + 1 batch executor
    star_specialist / _relaxed Star: orchestrator + 4 specialists
    star_step                 Star: orchestrator + 1 step executor
    mesh_round_robin          Mesh: 4 peers, round-robin, message board
    mesh_delegation           Mesh: 4 peers, dynamic delegation
    memory_none..memory_full  Star: orchestrator + 4 specialists, varying memory
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from orbit.configs.setup import (
    AgentMemoryAccess,
    MemoryConfig,
    SetupConfig,
)
from orbit.configs.topology_presets import (
    mesh_delegation,
    mesh_round_robin,
    single_agent,
    star_orchestrator_workers,
)
from orbit.scenarios.coding.swe_bench.prompts import (
    CODING_AGENT_PROMPT,
    EXECUTOR_BATCH_PROMPT,
    EXECUTOR_STEP_PROMPT,
    M3_ORCH_PROMPT,
    M3_SPECIALIST_PROMPT,
    MESH_DELEGATION_PROMPT,
    MESH_ROUND_ROBIN_PROMPT,
    ORCH_BATCH_DETAILED_PROMPT,
    ORCH_BATCH_RELAXED_PROMPT,
    ORCH_SPECIALIST_DETAILED_PROMPT,
    ORCH_SPECIALIST_RELAXED_PROMPT,
    ORCH_STEP_PROMPT,
    SPECIALIST_DEBUGGING_PROMPT,
    SPECIALIST_IMPLEMENTATION_PROMPT,
    SPECIALIST_REVIEW_PROMPT,
    SPECIALIST_TESTING_PROMPT,
)

# ---------------------------------------------------------------------------
# Coding tools constant
# ---------------------------------------------------------------------------

_CODING_TOOLS = ["bash", "python", "text_editor"]

# ---------------------------------------------------------------------------
# Specialist definitions reused across star_specialist, mesh, memory conditions
# ---------------------------------------------------------------------------

_SPECIALIST_DEFS: list[dict[str, Any]] = [
    {"name": "implementation_specialist", "role": "specialist", "specialty": "writing and modifying code"},
    {"name": "debugging_specialist", "role": "specialist", "specialty": "investigating bugs and diagnosing issues"},
    {"name": "testing_specialist", "role": "specialist", "specialty": "running tests and validating changes"},
    {"name": "review_specialist", "role": "specialist", "specialty": "reviewing code for correctness"},
]

_SPECIALIST_PROMPT_MAP = {
    "implementation_specialist": SPECIALIST_IMPLEMENTATION_PROMPT,
    "debugging_specialist": SPECIALIST_DEBUGGING_PROMPT,
    "testing_specialist": SPECIALIST_TESTING_PROMPT,
    "review_specialist": SPECIALIST_REVIEW_PROMPT,
}

# ---------------------------------------------------------------------------
# Single agent
# ---------------------------------------------------------------------------


def _build_single_agent() -> SetupConfig:
    """Single coding agent — baseline."""
    return single_agent(
        name="coding_agent",
        role="executor",
        system_prompt=CODING_AGENT_PROMPT,
        tools=_CODING_TOOLS,
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 1 batch executor
# ---------------------------------------------------------------------------


def _build_star_batch() -> SetupConfig:
    """Star: orchestrator + 1 batch executor with detailed instructions."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_BATCH_DETAILED_PROMPT,
        worker_specs=[{
            "name": "executor",
            "role": "executor",
            "system_prompt": EXECUTOR_BATCH_PROMPT,
            "tools": _CODING_TOOLS,
        }],
        properties={
            "condition_type": "star_batch",
            "execution_style": "batch",
        },
    )


def _build_star_batch_relaxed() -> SetupConfig:
    """Star: orchestrator + 1 batch executor with relaxed instructions."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_BATCH_RELAXED_PROMPT,
        worker_specs=[{
            "name": "executor",
            "role": "executor",
            "system_prompt": EXECUTOR_BATCH_PROMPT,
            "tools": _CODING_TOOLS,
        }],
        properties={
            "condition_type": "star_batch_relaxed",
            "execution_style": "batch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 4 specialists
# ---------------------------------------------------------------------------


def _specialist_workers() -> list[dict[str, Any]]:
    """Build worker spec dicts for the 4 specialists."""
    return [
        {
            "name": s["name"],
            "role": s["role"],
            "specialty": s["specialty"],
            "system_prompt": _SPECIALIST_PROMPT_MAP[s["name"]],
            "tools": _CODING_TOOLS,
        }
        for s in _SPECIALIST_DEFS
    ]


def _build_star_specialist() -> SetupConfig:
    """Star: orchestrator + 4 specialists with detailed dispatch."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_DETAILED_PROMPT,
        worker_specs=_specialist_workers(),
        properties={
            "condition_type": "star_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


def _build_star_specialist_relaxed() -> SetupConfig:
    """Star: orchestrator + 4 specialists with relaxed dispatch."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_RELAXED_PROMPT,
        worker_specs=_specialist_workers(),
        properties={
            "condition_type": "star_specialist_relaxed",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 1 step executor
# ---------------------------------------------------------------------------


def _build_star_step() -> SetupConfig:
    """Star: orchestrator + 1 step executor (one action at a time)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_STEP_PROMPT,
        worker_specs=[{
            "name": "executor",
            "role": "executor",
            "system_prompt": EXECUTOR_STEP_PROMPT,
            "tools": _CODING_TOOLS,
        }],
        properties={
            "condition_type": "star_step",
            "execution_style": "step",
        },
    )


# ---------------------------------------------------------------------------
# Mesh — 4 peers, round-robin, message board
# ---------------------------------------------------------------------------


def _build_mesh_round_robin() -> SetupConfig:
    """Mesh: 4 peers with round-robin turns and shared message board."""
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_ROUND_ROBIN_PROMPT,
            "tools": _CODING_TOOLS,
        }
        for s in _SPECIALIST_DEFS
    ]
    return mesh_round_robin(
        agent_specs=agent_specs,
        properties={
            "condition_type": "mesh_round_robin",
            "execution_style": "round_robin",
        },
    )


# ---------------------------------------------------------------------------
# Mesh — 4 peers, dynamic delegation
# ---------------------------------------------------------------------------


def _build_mesh_delegation() -> SetupConfig:
    """Mesh: 4 peers with dynamic delegation."""
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_DELEGATION_PROMPT,
            "tools": _CODING_TOOLS,
        }
        for s in _SPECIALIST_DEFS
    ]
    return mesh_delegation(
        agent_specs=agent_specs,
        properties={
            "condition_type": "mesh_delegation",
            "execution_style": "delegation",
        },
    )


# ---------------------------------------------------------------------------
# Memory conditions — Star: orchestrator + 4 specialists, varying memory
# ---------------------------------------------------------------------------

# Memory access levels (cumulative — each level adds visibility):
#   memory_none:           goal only (defaults)
#   memory_own_actions:    + own action history
#   memory_own_reasoning:  + own CoT
#   memory_shared_actions: + shared action history
#   memory_full:           + shared CoT + accumulated instructions
_MEMORY_LEVELS: dict[str, dict[str, bool]] = {
    "memory_none": {},
    "memory_own_actions": {"own_action_history": True},
    "memory_own_reasoning": {"own_action_history": True, "own_cot": True},
    "memory_shared_actions": {"own_action_history": True, "own_cot": True, "shared_action_history": True},
    "memory_full": {
        "own_action_history": True,
        "own_cot": True,
        "shared_action_history": True,
        "shared_cot": True,
        "nl_instructions_accumulated": True,
    },
}


def _memory_condition(level: str) -> SetupConfig:
    """Build a memory condition with memory access at the given level."""
    flags = _MEMORY_LEVELS[level]
    specialist_names = [s["name"] for s in _SPECIALIST_DEFS]
    all_names = ["orchestrator", *specialist_names]

    access_list = [
        AgentMemoryAccess(agent_name=name, **flags)
        for name in all_names
    ]
    memory = MemoryConfig(
        shared=True,
        shared_groups=[all_names],
        agent_memory_access=access_list,
    )

    return star_orchestrator_workers(
        orchestrator_prompt=M3_ORCH_PROMPT,
        worker_specs=[
            {
                "name": s["name"],
                "role": s["role"],
                "specialty": s["specialty"],
                "system_prompt": M3_SPECIALIST_PROMPT,
                "tools": _CODING_TOOLS,
            }
            for s in _SPECIALIST_DEFS
        ],
        memory=memory,
        properties={
            "condition_type": level,
            "execution_style": "specialist_dispatch",
            "memory_level": level,
        },
    )


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

CONDITION_REGISTRY: dict[str, Callable[[], SetupConfig]] = {
    "single_agent": _build_single_agent,
    "star_batch": _build_star_batch,
    "star_batch_relaxed": _build_star_batch_relaxed,
    "star_specialist": _build_star_specialist,
    "star_specialist_relaxed": _build_star_specialist_relaxed,
    "star_step": _build_star_step,
    "mesh_round_robin": _build_mesh_round_robin,
    "mesh_delegation": _build_mesh_delegation,
    "memory_none": lambda: _memory_condition("memory_none"),
    "memory_own_actions": lambda: _memory_condition("memory_own_actions"),
    "memory_own_reasoning": lambda: _memory_condition("memory_own_reasoning"),
    "memory_shared_actions": lambda: _memory_condition("memory_shared_actions"),
    "memory_full": lambda: _memory_condition("memory_full"),
}


def get_condition_setup(condition: str) -> SetupConfig:
    """Return the ``SetupConfig`` for a named condition.

    Raises ``ValueError`` if the condition is unknown.
    """
    factory = CONDITION_REGISTRY.get(condition)
    if factory is None:
        available = ", ".join(sorted(CONDITION_REGISTRY))
        raise ValueError(
            f"Unknown condition {condition!r}. Available: {available}"
        )
    return factory()


def list_conditions() -> list[str]:
    """Return a sorted list of available condition names."""
    return sorted(CONDITION_REGISTRY)


# ---------------------------------------------------------------------------
# Human-readable parameter resolver
# ---------------------------------------------------------------------------

_FRIENDLY_TO_CONDITION: dict[tuple[str, str, str, str], str] = {
    # (agents, topology, memory, instructions) → condition
    ("single", "star", "none", "detailed"): "single_agent",
    ("batch", "star", "none", "detailed"): "star_batch",
    ("batch", "star", "none", "relaxed"): "star_batch_relaxed",
    ("specialist", "star", "none", "detailed"): "star_specialist",
    ("specialist", "star", "none", "relaxed"): "star_specialist_relaxed",
    ("step", "star", "none", "detailed"): "star_step",
    ("specialist", "round_robin", "none", "detailed"): "mesh_round_robin",
    ("specialist", "delegation", "none", "detailed"): "mesh_delegation",
    ("specialist", "star", "own_actions", "detailed"): "memory_own_actions",
    ("specialist", "star", "own_reasoning", "detailed"): "memory_own_reasoning",
    ("specialist", "star", "shared_actions", "detailed"): "memory_shared_actions",
    ("specialist", "star", "full", "detailed"): "memory_full",
}

VALID_AGENTS = ("single", "batch", "specialist", "step")
VALID_TOPOLOGY = ("star", "round_robin", "delegation")
VALID_MEMORY = ("none", "own_actions", "own_reasoning", "shared_actions", "full")
VALID_INSTRUCTIONS = ("detailed", "relaxed")


def resolve_condition(
    agents: str = "single",
    topology: str = "star",
    memory: str = "none",
    instructions: str = "detailed",
) -> str:
    """Map human-readable (agents, topology, memory, instructions) to a condition name.

    Raises ValueError on invalid or unsupported combinations.
    """
    if agents not in VALID_AGENTS:
        raise ValueError(
            f"Invalid --agents '{agents}'. Choose from: {', '.join(VALID_AGENTS)}"
        )
    if topology not in VALID_TOPOLOGY:
        raise ValueError(
            f"Invalid --topology '{topology}'. Choose from: {', '.join(VALID_TOPOLOGY)}"
        )
    if memory not in VALID_MEMORY:
        raise ValueError(
            f"Invalid --memory '{memory}'. Choose from: {', '.join(VALID_MEMORY)}"
        )
    if instructions not in VALID_INSTRUCTIONS:
        raise ValueError(
            f"Invalid --instructions '{instructions}'. Choose from: {', '.join(VALID_INSTRUCTIONS)}"
        )

    key = (agents, topology, memory, instructions)
    condition = _FRIENDLY_TO_CONDITION.get(key)
    if condition is not None:
        return condition

    # --- Check if user asked for relaxed but only detailed exists ---
    if instructions == "relaxed":
        detailed_key = (agents, topology, memory, "detailed")
        if _FRIENDLY_TO_CONDITION.get(detailed_key) is not None:
            detailed_condition = _FRIENDLY_TO_CONDITION[detailed_key]
            raise ValueError(
                f"No relaxed variant exists for condition "
                f"'{detailed_condition}'. Use --instructions detailed "
                f"or omit --instructions."
            )

    # --- Unsupported combination: give a clear explanation ---
    if topology != "star" and agents != "specialist":
        raise ValueError(
            f"Unsupported: --topology '{topology}' with --agents '{agents}'.\n"
            f"Mesh topologies (round_robin, delegation) are only validated "
            f"with --agents specialist."
        )
    if memory != "none" and agents != "specialist":
        raise ValueError(
            f"Unsupported: --memory '{memory}' with --agents '{agents}'.\n"
            f"Memory sharing is only validated with --agents specialist."
        )
    if memory != "none" and topology != "star":
        raise ValueError(
            f"Unsupported: --memory '{memory}' with --topology '{topology}'.\n"
            f"Memory sharing is only validated with --topology star."
        )

    raise ValueError(
        f"Unsupported combination: --agents {agents} --topology {topology} "
        f"--memory {memory} --instructions {instructions}.\n"
        f"We only support combinations that have been experimentally validated. "
        f"Currently that is 13 conditions."
    )
