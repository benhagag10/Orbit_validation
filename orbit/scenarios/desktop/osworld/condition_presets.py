"""OSWorld / OS-Harm condition presets — thin binding over the shared YAML-backed loader.

Conditions ship as runnable configs in the general orbit syntax under
``orbit/scenarios/conditions/osworld/*.yaml``. This module binds the shared loader
(:mod:`orbit.scenarios.conditions`) to this scenario; there is no per-scenario registry to
maintain. Edit or add a condition by editing/adding its YAML.

``resolve_condition`` maps the friendly ``--agents/--topology/--memory/--instructions`` flags to a
condition name (OSWorld has both tool-action and application specialist families) and is kept here
as the only scenario-specific data.
"""

from __future__ import annotations

from typing import Callable

from orbit.configs.setup import SetupConfig
from orbit.scenarios import conditions as _shared

_SCENARIO = "osworld"


def get_condition_setup(condition: str) -> SetupConfig:
    """Return the SetupConfig for a named condition (loaded from its shipped YAML)."""
    return _shared.get_condition_setup(_SCENARIO, condition)


def list_conditions() -> list[str]:
    """Return a sorted list of available condition names."""
    return _shared.list_conditions(_SCENARIO)


# Back-compat: name -> zero-arg factory that rebuilds the SetupConfig from its YAML.
CONDITION_REGISTRY: dict[str, Callable[[], SetupConfig]] = {
    name: (lambda n=name: get_condition_setup(n)) for name in list_conditions()
}


# ---------------------------------------------------------------------------
# Human-readable parameter resolver (friendly flags -> condition name)
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
