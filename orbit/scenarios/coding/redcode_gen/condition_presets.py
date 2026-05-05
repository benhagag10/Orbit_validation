"""RedCode-Gen condition presets for ICML experiments.

Bridge between general topology presets and RedCode-Gen-specific details.
Each factory function calls a general topology preset from
:mod:`orbit.configs.topology_presets` and injects coding-domain
agent names, prompts, and tools.

Specialist decomposition for code generation:
- **design_specialist**: requirement analysis and algorithm design
- **code_specialist**: Python implementation
- **review_specialist**: code review for correctness
- **test_specialist**: validation and testing

The 13 ICML conditions:

    single_agent               Single coding agent
    star_batch / _relaxed      Star: orchestrator + 1 batch executor
    star_step                  Star: orchestrator + 1 step executor
    star_2_specialist          Star: orchestrator + 2 combined specialists
    star_3_specialist          Star: orchestrator + 3 specialists
    star_specialist            Star: orchestrator + 4 specialists
    mesh_round_robin           Mesh: 4 peers, round-robin
    mesh_delegation            Mesh: 4 peers, dynamic delegation
    memory_own_reasoning       Star + 4 specialists + CoT memory
    memory_full                Star + 4 specialists + full shared memory
    mesh_delegation_cot        Mesh + 4 peers + CoT memory
    mesh_delegation_full       Mesh + 4 peers + full shared memory
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
from orbit.scenarios.coding.redcode_gen.prompts import (
    DEFAULT_CODER_SYSTEM_PROMPT,
    EXECUTOR_BATCH_PROMPT,
    EXECUTOR_STEP_PROMPT,
    MESH_DELEGATION_PROMPT,
    MESH_ROUND_ROBIN_PROMPT,
    ORCH_2_SPECIALIST_PROMPT,
    ORCH_3_SPECIALIST_PROMPT,
    ORCH_BATCH_DETAILED_PROMPT,
    ORCH_BATCH_RELAXED_PROMPT,
    ORCH_SPECIALIST_PROMPT,
    ORCH_STEP_PROMPT,
    SPECIALIST_CODE_PROMPT,
    SPECIALIST_DESIGN_CODE_PROMPT,
    SPECIALIST_DESIGN_PROMPT,
    SPECIALIST_REVIEW_PROMPT,
    SPECIALIST_REVIEW_TEST_PROMPT,
    SPECIALIST_TEST_PROMPT,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_CODER_TOOLS = ["python"]

# Per-specialist tool decomposition (matches OSWorld's focused-tool pattern):
# - design_specialist: think only (pure reasoning, no execution)
# - code_specialist: python + text_editor (write and test code)
# - review_specialist: python + think (run code to verify, reason)
# - test_specialist: python + bash (run tests, validate)
_SPECIALIST_TOOLS_MAP: dict[str, list[str]] = {
    "design_specialist": ["think"],
    "code_specialist": ["python", "text_editor"],
    "review_specialist": ["python", "think"],
    "test_specialist": ["python", "bash"],
}

# Combined specialist tool maps (union of component roles)
_2_SPECIALIST_TOOLS_MAP: dict[str, list[str]] = {
    "design_code_specialist": ["python", "text_editor", "think"],
    "review_test_specialist": ["python", "bash", "think"],
}

_3_SPECIALIST_TOOLS_MAP: dict[str, list[str]] = {
    "design_code_specialist": ["python", "text_editor", "think"],
    "review_specialist": ["python", "think"],
    "test_specialist": ["python", "bash"],
}

# ---------------------------------------------------------------------------
# Specialist definitions (4 specialists)
# ---------------------------------------------------------------------------

_SPECIALIST_DEFS: list[dict[str, Any]] = [
    {"name": "design_specialist", "role": "specialist", "specialty": "requirement analysis and algorithm design"},
    {"name": "code_specialist", "role": "specialist", "specialty": "Python code implementation"},
    {"name": "review_specialist", "role": "specialist", "specialty": "code review for correctness and completeness"},
    {"name": "test_specialist", "role": "specialist", "specialty": "implementation validation and testing"},
]

_SPECIALIST_PROMPT_MAP: dict[str, str] = {
    "design_specialist": SPECIALIST_DESIGN_PROMPT,
    "code_specialist": SPECIALIST_CODE_PROMPT,
    "review_specialist": SPECIALIST_REVIEW_PROMPT,
    "test_specialist": SPECIALIST_TEST_PROMPT,
}

# ---------------------------------------------------------------------------
# Single agent
# ---------------------------------------------------------------------------


def _build_single_agent() -> SetupConfig:
    """Single coding agent — baseline."""
    return single_agent(
        name="coder",
        role="coder",
        system_prompt=DEFAULT_CODER_SYSTEM_PROMPT,
        tools=_CODER_TOOLS,
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
            "tools": _CODER_TOOLS,
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
            "tools": _CODER_TOOLS,
        }],
        properties={
            "condition_type": "star_batch_relaxed",
            "execution_style": "batch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 1 step executor
# ---------------------------------------------------------------------------


def _build_star_step() -> SetupConfig:
    """Star: orchestrator + 1 step executor (one instruction at a time)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_STEP_PROMPT,
        worker_specs=[{
            "name": "executor",
            "role": "executor",
            "system_prompt": EXECUTOR_STEP_PROMPT,
            "tools": _CODER_TOOLS,
        }],
        properties={
            "condition_type": "star_step",
            "execution_style": "step",
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
            "tools": _SPECIALIST_TOOLS_MAP.get(s["name"], _CODER_TOOLS),
        }
        for s in _SPECIALIST_DEFS
    ]


def _build_star_specialist() -> SetupConfig:
    """Star: orchestrator + 4 specialists with detailed dispatch."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_PROMPT,
        worker_specs=_specialist_workers(),
        properties={
            "condition_type": "star_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 2 combined specialists
# ---------------------------------------------------------------------------


def _build_star_2_specialist() -> SetupConfig:
    """Star: orchestrator + 2 combined specialists (design+code, review+test)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_2_SPECIALIST_PROMPT,
        worker_specs=[
            {
                "name": "design_code_specialist",
                "role": "specialist",
                "specialty": "requirement analysis, algorithm design, and Python implementation",
                "system_prompt": SPECIALIST_DESIGN_CODE_PROMPT,
                "tools": _2_SPECIALIST_TOOLS_MAP["design_code_specialist"],
            },
            {
                "name": "review_test_specialist",
                "role": "specialist",
                "specialty": "code review, correctness validation, and testing",
                "system_prompt": SPECIALIST_REVIEW_TEST_PROMPT,
                "tools": _2_SPECIALIST_TOOLS_MAP["review_test_specialist"],
            },
        ],
        properties={
            "condition_type": "star_2_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 3 specialists
# ---------------------------------------------------------------------------


def _build_star_3_specialist() -> SetupConfig:
    """Star: orchestrator + 3 specialists (design+code combined, review, test)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_3_SPECIALIST_PROMPT,
        worker_specs=[
            {
                "name": "design_code_specialist",
                "role": "specialist",
                "specialty": "requirement analysis, algorithm design, and Python implementation",
                "system_prompt": SPECIALIST_DESIGN_CODE_PROMPT,
                "tools": _3_SPECIALIST_TOOLS_MAP["design_code_specialist"],
            },
            {
                "name": "review_specialist",
                "role": "specialist",
                "specialty": "code review for correctness and completeness",
                "system_prompt": SPECIALIST_REVIEW_PROMPT,
                "tools": _3_SPECIALIST_TOOLS_MAP["review_specialist"],
            },
            {
                "name": "test_specialist",
                "role": "specialist",
                "specialty": "implementation validation and testing",
                "system_prompt": SPECIALIST_TEST_PROMPT,
                "tools": _3_SPECIALIST_TOOLS_MAP["test_specialist"],
            },
        ],
        properties={
            "condition_type": "star_3_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Mesh — 4 peers, round-robin
# ---------------------------------------------------------------------------


def _build_mesh_round_robin() -> SetupConfig:
    """Mesh: 4 specialist peers with round-robin turns."""
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_ROUND_ROBIN_PROMPT.format(
                specialty=s["specialty"],
            ),
            "tools": _SPECIALIST_TOOLS_MAP.get(s["name"], _CODER_TOOLS),
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


def _mesh_delegation_peer_list() -> str:
    """Build the peer list string for mesh delegation prompts."""
    lines = []
    for s in _SPECIALIST_DEFS:
        lines.append(f"- ``{s['name']}``: {s['specialty']}")
    return "\n".join(lines)


def _build_mesh_delegation() -> SetupConfig:
    """Mesh: 4 specialist peers with dynamic delegation."""
    peer_list = _mesh_delegation_peer_list()
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_DELEGATION_PROMPT.format(
                specialty=s["specialty"],
                peer_list=peer_list,
            ),
            "tools": _SPECIALIST_TOOLS_MAP.get(s["name"], _CODER_TOOLS),
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
# Memory levels
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Star — specialists with varying memory
# ---------------------------------------------------------------------------


def _memory_condition(level: str) -> SetupConfig:
    """Build a specialist memory condition."""
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
        orchestrator_prompt=ORCH_SPECIALIST_PROMPT,
        worker_specs=_specialist_workers(),
        memory=memory,
        properties={
            "condition_type": level,
            "execution_style": "specialist_dispatch",
            "memory_level": level,
        },
    )


# ---------------------------------------------------------------------------
# Mesh — delegation with varying memory
# ---------------------------------------------------------------------------


def _build_mesh_delegation_memory(level: str, *, condition_name: str) -> SetupConfig:
    """Mesh: 4 specialist peers with delegation and memory."""
    flags = _MEMORY_LEVELS[level]
    specialist_names = [s["name"] for s in _SPECIALIST_DEFS]

    access_list = [
        AgentMemoryAccess(agent_name=name, **flags)
        for name in specialist_names
    ]
    memory = MemoryConfig(
        shared=True,
        shared_groups=[specialist_names],
        agent_memory_access=access_list,
    )

    peer_list = _mesh_delegation_peer_list()
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_DELEGATION_PROMPT.format(
                specialty=s["specialty"],
                peer_list=peer_list,
            ),
            "tools": _SPECIALIST_TOOLS_MAP.get(s["name"], _CODER_TOOLS),
        }
        for s in _SPECIALIST_DEFS
    ]
    return mesh_delegation(
        agent_specs=agent_specs,
        memory=memory,
        properties={
            "condition_type": condition_name,
            "execution_style": "delegation",
            "memory_level": level,
        },
    )


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

CONDITION_REGISTRY: dict[str, Callable[[], SetupConfig]] = {
    # Table 1: Role Distribution
    "single_agent": _build_single_agent,
    "star_batch": _build_star_batch,
    "star_batch_relaxed": _build_star_batch_relaxed,
    "star_step": _build_star_step,
    "star_2_specialist": _build_star_2_specialist,
    "star_3_specialist": _build_star_3_specialist,
    "star_specialist": _build_star_specialist,
    # Table 2: Communication Topology
    "mesh_round_robin": _build_mesh_round_robin,
    "mesh_delegation": _build_mesh_delegation,
    # Table 3: Memory & State Visibility
    "memory_own_reasoning": lambda: _memory_condition("memory_own_reasoning"),
    "memory_full": lambda: _memory_condition("memory_full"),
    "mesh_delegation_cot": lambda: _build_mesh_delegation_memory("memory_own_reasoning", condition_name="mesh_delegation_cot"),
    "mesh_delegation_full": lambda: _build_mesh_delegation_memory("memory_full", condition_name="mesh_delegation_full"),
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
