"""OSWorld desktop-agent condition presets.

Bridge between general topology presets and desktop-agent-specific details.
Each factory function calls a general topology preset from
:mod:`orbit.configs.topology_presets` and injects OSWorld-specific
agent names, prompts, and tools.

Two specialist flavours:
- **Tool-action specialists** (default): click, type, key, scroll — each
  specialist handles one ``computer`` tool action type.
- **Application specialists**: email, browser, document, code — each
  specialist handles one desktop application.

The 26 conditions:

    single_agent                    Single agent
    star_batch / _relaxed           Star: orchestrator + 1 batch executor
    star_tool_specialist / _relaxed Star: orchestrator + 4 tool-action specialists
    star_specialist / _relaxed      Star: orchestrator + 4 app specialists
    star_2_specialist               Star: orchestrator + 2 combined app specialists
    star_3_specialist               Star: orchestrator + 3 app specialists
    star_step                       Star: orchestrator + 1 step executor
    mesh_tool_round_robin           Mesh: 4 tool-action peers, round-robin
    mesh_round_robin                Mesh: 4 app peers, round-robin
    mesh_tool_delegation            Mesh: 4 tool-action peers, dynamic delegation
    mesh_delegation                 Mesh: 4 app peers, dynamic delegation
    mesh_delegation_cot             Mesh: 4 app peers, delegation + CoT memory
    mesh_delegation_full            Mesh: 4 app peers, delegation + full memory
    tool_memory_none..full          Star: tool-action specialists, varying memory
    memory_none..memory_full        Star: app specialists, varying memory
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
from orbit.scenarios.desktop.osworld.prompts import (
    COMPUTER_USE_AGENT_PROMPT,
    EXECUTOR_BATCH_PROMPT,
    EXECUTOR_STEP_PROMPT,
    MESH_DELEGATION_PROMPT,
    MESH_ROUND_ROBIN_PROMPT,
    ORCH_BATCH_DETAILED_PROMPT,
    ORCH_BATCH_RELAXED_PROMPT,
    ORCH_SPECIALIST_DETAILED_PROMPT,
    ORCH_SPECIALIST_RELAXED_PROMPT,
    ORCH_STEP_PROMPT,
    ORCH_TOOL_SPECIALIST_DETAILED_PROMPT,
    ORCH_TOOL_SPECIALIST_RELAXED_PROMPT,
    SPECIALIST_BROWSER_PROMPT,
    SPECIALIST_CLICK_PROMPT,
    SPECIALIST_CLICK_SCROLL_PROMPT,
    SPECIALIST_CODE_PROMPT,
    SPECIALIST_DOCUMENT_CODE_PROMPT,
    SPECIALIST_DOCUMENT_PROMPT,
    SPECIALIST_EMAIL_BROWSER_PROMPT,
    SPECIALIST_EMAIL_PROMPT,
    SPECIALIST_KEY_PROMPT,
    SPECIALIST_SCROLL_PROMPT,
    SPECIALIST_TYPE_KEY_PROMPT,
    SPECIALIST_TYPE_PROMPT,
    ORCH_TOOL_2_SPECIALIST_PROMPT,
    ORCH_TOOL_3_SPECIALIST_PROMPT,
)

# ---------------------------------------------------------------------------
# Computer-use tools constant
# ---------------------------------------------------------------------------

_COMPUTER_TOOLS = ["computer", "submit"]

# ---------------------------------------------------------------------------
# Focused computer action tools for tool-action specialists
# ---------------------------------------------------------------------------

_CLICK_SPECIALIST_TOOLS = ["computer_click", "computer_screenshot", "submit"]
_TYPE_SPECIALIST_TOOLS = ["computer_type", "computer_screenshot", "submit"]
_KEY_SPECIALIST_TOOLS = ["computer_key", "computer_screenshot", "submit"]
_SCROLL_SPECIALIST_TOOLS = ["computer_scroll", "computer_screenshot", "submit"]

_CLICK_SCROLL_SPECIALIST_TOOLS = [
    "computer_click", "computer_scroll", "computer_screenshot", "submit",
]
_TYPE_KEY_SPECIALIST_TOOLS = [
    "computer_type", "computer_key", "computer_screenshot", "submit",
]

_TOOL_SPECIALIST_TOOLS_MAP: dict[str, list[str]] = {
    "click_specialist": _CLICK_SPECIALIST_TOOLS,
    "type_specialist": _TYPE_SPECIALIST_TOOLS,
    "key_specialist": _KEY_SPECIALIST_TOOLS,
    "scroll_specialist": _SCROLL_SPECIALIST_TOOLS,
}

# ---------------------------------------------------------------------------
# Tool-action specialist definitions (default)
# ---------------------------------------------------------------------------

_TOOL_SPECIALIST_DEFS: list[dict[str, Any]] = [
    {"name": "click_specialist", "role": "specialist", "specialty": "mouse clicking (left_click, right_click, double_click, mouse_move, drag)"},
    {"name": "type_specialist", "role": "specialist", "specialty": "text input (type action)"},
    {"name": "key_specialist", "role": "specialist", "specialty": "keyboard shortcuts (key, hold_key actions)"},
    {"name": "scroll_specialist", "role": "specialist", "specialty": "scrolling (scroll action)"},
]

_TOOL_SPECIALIST_PROMPT_MAP = {
    "click_specialist": SPECIALIST_CLICK_PROMPT,
    "type_specialist": SPECIALIST_TYPE_PROMPT,
    "key_specialist": SPECIALIST_KEY_PROMPT,
    "scroll_specialist": SPECIALIST_SCROLL_PROMPT,
}

# ---------------------------------------------------------------------------
# Application specialist definitions (legacy, --agents app_specialist)
# ---------------------------------------------------------------------------

_SPECIALIST_DEFS: list[dict[str, Any]] = [
    {"name": "email_specialist", "role": "specialist", "specialty": "email clients (Thunderbird)"},
    {"name": "browser_specialist", "role": "specialist", "specialty": "web browsers (Chrome)"},
    {"name": "document_specialist", "role": "specialist", "specialty": "document suites (LibreOffice, GIMP)"},
    {"name": "code_specialist", "role": "specialist", "specialty": "code editors and terminal (VS Code, OS)"},
]

_SPECIALIST_PROMPT_MAP = {
    "email_specialist": SPECIALIST_EMAIL_PROMPT,
    "browser_specialist": SPECIALIST_BROWSER_PROMPT,
    "document_specialist": SPECIALIST_DOCUMENT_PROMPT,
    "code_specialist": SPECIALIST_CODE_PROMPT,
}

# ---------------------------------------------------------------------------
# Single agent
# ---------------------------------------------------------------------------


def _build_single_agent() -> SetupConfig:
    """Single computer-use agent — baseline."""
    return single_agent(
        name="computer_use_agent",
        role="executor",
        system_prompt=COMPUTER_USE_AGENT_PROMPT,
        tools=_COMPUTER_TOOLS,
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
            "tools": _COMPUTER_TOOLS,
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
            "tools": _COMPUTER_TOOLS,
        }],
        properties={
            "condition_type": "star_batch_relaxed",
            "execution_style": "batch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 4 tool-action specialists (default)
# ---------------------------------------------------------------------------


def _tool_specialist_workers() -> list[dict[str, Any]]:
    """Build worker spec dicts for the 4 tool-action specialists."""
    return [
        {
            "name": s["name"],
            "role": s["role"],
            "specialty": s["specialty"],
            "system_prompt": _TOOL_SPECIALIST_PROMPT_MAP[s["name"]],
            "tools": _TOOL_SPECIALIST_TOOLS_MAP[s["name"]],
        }
        for s in _TOOL_SPECIALIST_DEFS
    ]


def _build_star_tool_specialist() -> SetupConfig:
    """Star: orchestrator + 4 tool-action specialists with detailed dispatch."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_TOOL_SPECIALIST_DETAILED_PROMPT,
        worker_specs=_tool_specialist_workers(),
        properties={
            "condition_type": "star_tool_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


def _build_star_tool_specialist_relaxed() -> SetupConfig:
    """Star: orchestrator + 4 tool-action specialists with relaxed dispatch."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_TOOL_SPECIALIST_RELAXED_PROMPT,
        worker_specs=_tool_specialist_workers(),
        properties={
            "condition_type": "star_tool_specialist_relaxed",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 2 combined tool-action specialists
# ---------------------------------------------------------------------------


def _build_star_tool_2_specialist() -> SetupConfig:
    """Star: orchestrator + 2 combined tool-action specialists (type+key, click+scroll)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_TOOL_2_SPECIALIST_PROMPT,
        worker_specs=[
            {
                "name": "type_key_specialist",
                "role": "specialist",
                "specialty": "text input (typing) and keyboard shortcuts (key combinations)",
                "system_prompt": SPECIALIST_TYPE_KEY_PROMPT,
                "tools": _TYPE_KEY_SPECIALIST_TOOLS,
            },
            {
                "name": "click_scroll_specialist",
                "role": "specialist",
                "specialty": "mouse actions (click, right_click, double_click, drag) and scrolling",
                "system_prompt": SPECIALIST_CLICK_SCROLL_PROMPT,
                "tools": _CLICK_SCROLL_SPECIALIST_TOOLS,
            },
        ],
        properties={
            "condition_type": "star_tool_2_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 3 tool-action specialists
# ---------------------------------------------------------------------------


def _build_star_tool_3_specialist() -> SetupConfig:
    """Star: orchestrator + 3 tool-action specialists (click+scroll combined, key, type)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_TOOL_3_SPECIALIST_PROMPT,
        worker_specs=[
            {
                "name": "click_scroll_specialist",
                "role": "specialist",
                "specialty": "mouse actions (click, right_click, double_click, drag) and scrolling",
                "system_prompt": SPECIALIST_CLICK_SCROLL_PROMPT,
                "tools": _CLICK_SCROLL_SPECIALIST_TOOLS,
            },
            {
                "name": "key_specialist",
                "role": "specialist",
                "specialty": "keyboard shortcuts (key combinations like ctrl+s, alt+Tab, Return, Escape)",
                "system_prompt": SPECIALIST_KEY_PROMPT,
                "tools": _KEY_SPECIALIST_TOOLS,
            },
            {
                "name": "type_specialist",
                "role": "specialist",
                "specialty": "text input (typing text into fields and applications)",
                "system_prompt": SPECIALIST_TYPE_PROMPT,
                "tools": _TYPE_SPECIALIST_TOOLS,
            },
        ],
        properties={
            "condition_type": "star_tool_3_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 4 application specialists (legacy)
# ---------------------------------------------------------------------------


def _specialist_workers() -> list[dict[str, Any]]:
    """Build worker spec dicts for the 4 application specialists."""
    return [
        {
            "name": s["name"],
            "role": s["role"],
            "specialty": s["specialty"],
            "system_prompt": _SPECIALIST_PROMPT_MAP[s["name"]],
            "tools": _COMPUTER_TOOLS,
        }
        for s in _SPECIALIST_DEFS
    ]


def _build_star_specialist() -> SetupConfig:
    """Star: orchestrator + 4 app specialists with detailed dispatch."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_DETAILED_PROMPT,
        worker_specs=_specialist_workers(),
        properties={
            "condition_type": "star_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


def _build_star_specialist_relaxed() -> SetupConfig:
    """Star: orchestrator + 4 app specialists with relaxed dispatch."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_RELAXED_PROMPT,
        worker_specs=_specialist_workers(),
        properties={
            "condition_type": "star_specialist_relaxed",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 2 combined application specialists
# ---------------------------------------------------------------------------


def _build_star_2_specialist() -> SetupConfig:
    """Star: orchestrator + 2 combined app specialists (email+browser, document+code)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_DETAILED_PROMPT,
        worker_specs=[
            {
                "name": "email_browser_specialist",
                "role": "specialist",
                "specialty": "email clients (Thunderbird) and web browsers (Chrome)",
                "system_prompt": SPECIALIST_EMAIL_BROWSER_PROMPT,
                "tools": _COMPUTER_TOOLS,
            },
            {
                "name": "document_code_specialist",
                "role": "specialist",
                "specialty": "document suites (LibreOffice, GIMP) and code editors/terminal (VS Code, OS)",
                "system_prompt": SPECIALIST_DOCUMENT_CODE_PROMPT,
                "tools": _COMPUTER_TOOLS,
            },
        ],
        properties={
            "condition_type": "star_2_specialist",
            "execution_style": "specialist_dispatch",
        },
    )


# ---------------------------------------------------------------------------
# Star — Orchestrator + 3 application specialists
# ---------------------------------------------------------------------------


def _build_star_3_specialist() -> SetupConfig:
    """Star: orchestrator + 3 app specialists (email+browser combined, document, code)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_DETAILED_PROMPT,
        worker_specs=[
            {
                "name": "email_browser_specialist",
                "role": "specialist",
                "specialty": "email clients (Thunderbird) and web browsers (Chrome)",
                "system_prompt": SPECIALIST_EMAIL_BROWSER_PROMPT,
                "tools": _COMPUTER_TOOLS,
            },
            {
                "name": "document_specialist",
                "role": "specialist",
                "specialty": "document suites (LibreOffice, GIMP)",
                "system_prompt": SPECIALIST_DOCUMENT_PROMPT,
                "tools": _COMPUTER_TOOLS,
            },
            {
                "name": "code_specialist",
                "role": "specialist",
                "specialty": "code editors and terminal (VS Code, OS)",
                "system_prompt": SPECIALIST_CODE_PROMPT,
                "tools": _COMPUTER_TOOLS,
            },
        ],
        properties={
            "condition_type": "star_3_specialist",
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
            "tools": _COMPUTER_TOOLS,
        }],
        properties={
            "condition_type": "star_step",
            "execution_style": "step",
        },
    )


# ---------------------------------------------------------------------------
# Mesh — 4 tool-action peers, round-robin (default)
# ---------------------------------------------------------------------------


def _build_mesh_tool_round_robin() -> SetupConfig:
    """Mesh: 4 tool-action peers with round-robin turns and shared message board."""
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_ROUND_ROBIN_PROMPT,
            "tools": _TOOL_SPECIALIST_TOOLS_MAP[s["name"]],
        }
        for s in _TOOL_SPECIALIST_DEFS
    ]
    return mesh_round_robin(
        agent_specs=agent_specs,
        properties={
            "condition_type": "mesh_tool_round_robin",
            "execution_style": "round_robin",
        },
    )


# ---------------------------------------------------------------------------
# Mesh — 4 tool-action peers, dynamic delegation (default)
# ---------------------------------------------------------------------------


def _build_mesh_tool_delegation() -> SetupConfig:
    """Mesh: 4 tool-action peers with dynamic delegation."""
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_DELEGATION_PROMPT,
            "tools": _TOOL_SPECIALIST_TOOLS_MAP[s["name"]],
        }
        for s in _TOOL_SPECIALIST_DEFS
    ]
    return mesh_delegation(
        agent_specs=agent_specs,
        properties={
            "condition_type": "mesh_tool_delegation",
            "execution_style": "delegation",
        },
    )


def _build_mesh_tool_delegation_memory(level: str, *, condition_name: str) -> SetupConfig:
    """Mesh: 4 tool-action peers with dynamic delegation and memory at the given level."""
    flags = _MEMORY_LEVELS[level]
    specialist_names = [s["name"] for s in _TOOL_SPECIALIST_DEFS]

    access_list = [
        AgentMemoryAccess(agent_name=name, **flags)
        for name in specialist_names
    ]
    memory = MemoryConfig(
        shared=True,
        shared_groups=[specialist_names],
        agent_memory_access=access_list,
    )

    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_DELEGATION_PROMPT,
            "tools": _TOOL_SPECIALIST_TOOLS_MAP[s["name"]],
        }
        for s in _TOOL_SPECIALIST_DEFS
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
# Mesh — 4 app peers, round-robin (legacy)
# ---------------------------------------------------------------------------


def _build_mesh_round_robin() -> SetupConfig:
    """Mesh: 4 app peers with round-robin turns and shared message board."""
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_ROUND_ROBIN_PROMPT,
            "tools": _COMPUTER_TOOLS,
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
# Mesh — 4 app peers, dynamic delegation (legacy)
# ---------------------------------------------------------------------------


def _build_mesh_delegation() -> SetupConfig:
    """Mesh: 4 app peers with dynamic delegation."""
    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_DELEGATION_PROMPT,
            "tools": _COMPUTER_TOOLS,
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
# Mesh — 4 app peers, dynamic delegation with memory
# ---------------------------------------------------------------------------


def _build_mesh_delegation_memory(level: str, *, condition_name: str) -> SetupConfig:
    """Mesh: 4 app peers with dynamic delegation and memory at the given level."""
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

    agent_specs = [
        {
            "name": s["name"],
            "role": "peer",
            "specialty": s["specialty"],
            "system_prompt": MESH_DELEGATION_PROMPT,
            "tools": _COMPUTER_TOOLS,
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
# Memory conditions — Star: specialists with varying memory
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


def _tool_memory_condition(level: str) -> SetupConfig:
    """Build a tool-action specialist memory condition."""
    # Strip "tool_" prefix to look up the base memory level
    base_level = level.removeprefix("tool_")
    flags = _MEMORY_LEVELS[base_level]
    specialist_names = [s["name"] for s in _TOOL_SPECIALIST_DEFS]
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

    # Use the same orchestrator and specialist prompts as star_tool_specialist
    # so the ONLY variable is the MemoryConfig.
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_TOOL_SPECIALIST_DETAILED_PROMPT,
        worker_specs=[
            {
                "name": s["name"],
                "role": s["role"],
                "specialty": s["specialty"],
                "system_prompt": _TOOL_SPECIALIST_PROMPT_MAP[s["name"]],
                "tools": _TOOL_SPECIALIST_TOOLS_MAP[s["name"]],
            }
            for s in _TOOL_SPECIALIST_DEFS
        ],
        memory=memory,
        properties={
            "condition_type": level,
            "execution_style": "specialist_dispatch",
            "memory_level": base_level,
        },
    )


def _memory_condition(level: str) -> SetupConfig:
    """Build an app-specialist memory condition with memory access at the given level."""
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

    # Use the same orchestrator and specialist prompts as star_specialist
    # so the ONLY variable is the MemoryConfig.
    return star_orchestrator_workers(
        orchestrator_prompt=ORCH_SPECIALIST_DETAILED_PROMPT,
        worker_specs=[
            {
                "name": s["name"],
                "role": s["role"],
                "specialty": s["specialty"],
                "system_prompt": _SPECIALIST_PROMPT_MAP[s["name"]],
                "tools": _COMPUTER_TOOLS,
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
    # Non-specialist conditions
    "single_agent": _build_single_agent,
    "star_batch": _build_star_batch,
    "star_batch_relaxed": _build_star_batch_relaxed,
    "star_step": _build_star_step,
    # Tool-action specialists (default)
    "star_tool_specialist": _build_star_tool_specialist,
    "star_tool_specialist_relaxed": _build_star_tool_specialist_relaxed,
    "star_tool_2_specialist": _build_star_tool_2_specialist,
    "star_tool_3_specialist": _build_star_tool_3_specialist,
    "mesh_tool_round_robin": _build_mesh_tool_round_robin,
    "mesh_tool_delegation": _build_mesh_tool_delegation,
    "mesh_tool_delegation_cot": lambda: _build_mesh_tool_delegation_memory("memory_own_reasoning", condition_name="mesh_tool_delegation_cot"),
    "mesh_tool_delegation_full": lambda: _build_mesh_tool_delegation_memory("memory_full", condition_name="mesh_tool_delegation_full"),
    "tool_memory_none": lambda: _tool_memory_condition("tool_memory_none"),
    "tool_memory_own_actions": lambda: _tool_memory_condition("tool_memory_own_actions"),
    "tool_memory_own_reasoning": lambda: _tool_memory_condition("tool_memory_own_reasoning"),
    "tool_memory_shared_actions": lambda: _tool_memory_condition("tool_memory_shared_actions"),
    "tool_memory_full": lambda: _tool_memory_condition("tool_memory_full"),
    # Application specialists (legacy)
    "star_specialist": _build_star_specialist,
    "star_specialist_relaxed": _build_star_specialist_relaxed,
    "star_2_specialist": _build_star_2_specialist,
    "star_3_specialist": _build_star_3_specialist,
    "mesh_round_robin": _build_mesh_round_robin,
    "mesh_delegation": _build_mesh_delegation,
    "mesh_delegation_cot": lambda: _build_mesh_delegation_memory("memory_own_reasoning", condition_name="mesh_delegation_cot"),
    "mesh_delegation_full": lambda: _build_mesh_delegation_memory("memory_full", condition_name="mesh_delegation_full"),
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
    ("single", "star", "none", "relaxed"): "single_agent",        # no relaxed variant
    ("batch", "star", "none", "detailed"): "star_batch",
    ("batch", "star", "none", "relaxed"): "star_batch_relaxed",
    ("step", "star", "none", "detailed"): "star_step",
    ("step", "star", "none", "relaxed"): "star_step",             # no relaxed variant
    # Tool-action specialists (default for --agents specialist)
    ("specialist", "star", "none", "detailed"): "star_tool_specialist",
    ("specialist", "star", "none", "relaxed"): "star_tool_specialist_relaxed",
    ("specialist", "round_robin", "none", "detailed"): "mesh_tool_round_robin",
    ("specialist", "round_robin", "none", "relaxed"): "mesh_tool_round_robin",  # no relaxed variant
    ("specialist", "delegation", "none", "detailed"): "mesh_tool_delegation",
    ("specialist", "delegation", "none", "relaxed"): "mesh_tool_delegation",   # no relaxed variant
    ("specialist", "star", "own_actions", "detailed"): "tool_memory_own_actions",
    ("specialist", "star", "own_reasoning", "detailed"): "tool_memory_own_reasoning",
    ("specialist", "star", "shared_actions", "detailed"): "tool_memory_shared_actions",
    ("specialist", "star", "full", "detailed"): "tool_memory_full",
    # Application specialists (--agents app_specialist)
    ("app_specialist", "star", "none", "detailed"): "star_specialist",
    ("app_specialist", "star", "none", "relaxed"): "star_specialist_relaxed",
    ("app_specialist", "round_robin", "none", "detailed"): "mesh_round_robin",
    ("app_specialist", "round_robin", "none", "relaxed"): "mesh_round_robin",  # no relaxed variant
    ("app_specialist", "delegation", "none", "detailed"): "mesh_delegation",
    ("app_specialist", "delegation", "none", "relaxed"): "mesh_delegation",   # no relaxed variant
    ("app_specialist", "star", "own_actions", "detailed"): "memory_own_actions",
    ("app_specialist", "star", "own_reasoning", "detailed"): "memory_own_reasoning",
    ("app_specialist", "star", "shared_actions", "detailed"): "memory_shared_actions",
    ("app_specialist", "star", "full", "detailed"): "memory_full",
}

VALID_AGENTS = ("single", "batch", "specialist", "app_specialist", "step")
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

    # --- Unsupported combination: give a clear explanation ---
    if topology != "star" and agents not in ("specialist", "app_specialist"):
        raise ValueError(
            f"Unsupported: --topology '{topology}' with --agents '{agents}'.\n"
            f"Mesh topologies (round_robin, delegation) are only validated "
            f"with --agents specialist or app_specialist."
        )
    if memory != "none" and agents not in ("specialist", "app_specialist"):
        raise ValueError(
            f"Unsupported: --memory '{memory}' with --agents '{agents}'.\n"
            f"Memory sharing is only validated with --agents specialist or app_specialist."
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
        f"Currently that is 26 conditions."
    )
