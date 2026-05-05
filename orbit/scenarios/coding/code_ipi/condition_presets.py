"""CodeIPI topology condition presets.

Maps general topology presets to coding-agent-specific agent names,
prompts, and tools. Supports single-agent, star (orchestrator + workers),
and mesh topologies.
"""

from __future__ import annotations

from typing import Any, Callable

from orbit.configs.setup import SetupConfig
from orbit.configs.topology_presets import (
    mesh_delegation,
    mesh_round_robin,
    single_agent,
    star_orchestrator_workers,
)
from orbit.scenarios.coding.code_ipi.prompts import (
    CODE_FIXER_PROMPT,
    CODE_READER_PROMPT,
    CODE_TESTER_PROMPT,
    CODING_AGENT_PROMPT,
    MESH_PEER_PROMPT,
    ORCHESTRATOR_PROMPT,
    SINGLE_EXECUTOR_PROMPT,
)

_CODING_TOOLS = ["bash", "python"]


def single_coder() -> SetupConfig:
    """Single coding agent — baseline condition."""
    return single_agent(
        name="coder",
        role="executor",
        system_prompt=CODING_AGENT_PROMPT,
        tools=_CODING_TOOLS,
    )


def star_single_executor() -> SetupConfig:
    """Star: orchestrator + 1 executor."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        worker_specs=[
            {
                "name": "executor",
                "role": "executor",
                "system_prompt": SINGLE_EXECUTOR_PROMPT,
                "tools": _CODING_TOOLS,
            },
        ],
        properties={"condition_type": "star_single_executor"},
    )


def star_specialists() -> SetupConfig:
    """Star: orchestrator + 3 specialists (reader, fixer, tester)."""
    return star_orchestrator_workers(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        worker_specs=[
            {
                "name": "code_reader",
                "role": "specialist",
                "system_prompt": CODE_READER_PROMPT,
                "tools": _CODING_TOOLS,
            },
            {
                "name": "code_fixer",
                "role": "specialist",
                "system_prompt": CODE_FIXER_PROMPT,
                "tools": _CODING_TOOLS,
            },
            {
                "name": "code_tester",
                "role": "specialist",
                "system_prompt": CODE_TESTER_PROMPT,
                "tools": _CODING_TOOLS,
            },
        ],
        properties={"condition_type": "star_specialists"},
    )


def mesh_peers() -> SetupConfig:
    """Mesh: 3 peer agents with round-robin turns."""
    return mesh_round_robin(
        agent_specs=[
            {
                "name": f"coder_{i}",
                "role": "peer",
                "system_prompt": MESH_PEER_PROMPT,
                "tools": _CODING_TOOLS,
            }
            for i in range(3)
        ],
        properties={"condition_type": "mesh_peers"},
    )


def mesh_delegation_peers() -> SetupConfig:
    """Mesh: 3 peer agents with free delegation."""
    return mesh_delegation(
        agent_specs=[
            {
                "name": f"coder_{i}",
                "role": "peer",
                "system_prompt": MESH_PEER_PROMPT,
                "tools": _CODING_TOOLS,
            }
            for i in range(3)
        ],
        properties={"condition_type": "mesh_delegation"},
    )


CONDITION_REGISTRY: dict[str, Callable[[], SetupConfig]] = {
    "single_agent": single_coder,
    "star_single_executor": star_single_executor,
    "star_specialists": star_specialists,
    "mesh_round_robin": mesh_peers,
    "mesh_delegation": mesh_delegation_peers,
}


def get_condition_setup(condition: str) -> SetupConfig:
    """Return the SetupConfig for a named condition.

    Raises ValueError if the condition is unknown.
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
