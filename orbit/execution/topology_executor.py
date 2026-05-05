"""TopologyExecutor — single root agent with sub-agents wired as tools.

All agents share one TaskState. The root agent runs and delegates to
sub-agents via handoff/as_tool edges defined in the topology config.

This is the backward-compatible execution path (formerly Path A).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from inspect_ai.agent import run

from orbit.agents.topologies import (
    build_topology,
    build_topology_agents,
    find_root_agents,
    get_direct_run_sequence,
)
from orbit.execution.agent_scheduler import AgentTurnResult
from orbit.execution.executor import TurnExecutor

if TYPE_CHECKING:
    from collections.abc import Callable

    from inspect_ai.agent import Agent, AgentState
    from inspect_ai.model import Model
    from inspect_ai.solver import TaskState

    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.setup import SetupConfig

logger = logging.getLogger(__name__)


class TopologyExecutor(TurnExecutor):
    """Executes a single-topology agent graph on shared TaskState."""

    def __init__(
        self,
        setup: SetupConfig,
        model_wrapper: Callable[[str, Model], Model] | None = None,
        tool_wrappers: dict[str, object] | None = None,
    ) -> None:
        self._setup = setup
        self._agents = build_topology_agents(
            setup, model_wrapper=model_wrapper, tool_wrappers=tool_wrappers,
        )
        self._root: Agent | None = None
        self._direct_seq: list[str] = []
        self._root_names: list[str] = []
        self._completed = False
        self._invoked: list[str] = []

    @property
    def agents(self) -> dict[str, Agent]:
        return self._agents

    def initialize(self, state: TaskState, config: ExperimentConfig) -> None:
        self._root = build_topology(self._setup, self._agents)
        self._direct_seq = get_direct_run_sequence(self._setup)
        self._root_names = find_root_agents(self._setup)

    async def run(self, state: TaskState, turn: int) -> list[AgentTurnResult]:
        msg_before = len(state.messages)
        configured_names = {s.name for s in self._setup.agents}
        last_agent = self._root_names[0] if self._root_names else "root"

        if self._direct_seq:
            for name in self._direct_seq:
                if name in self._agents:
                    result = await run(self._agents[name], state)
                    state.messages[:] = result.messages
                    state.output = result.output
                    last_agent = name
                    if name not in self._invoked:
                        self._invoked.append(name)
        else:
            result = await run(self._root, state)
            state.messages[:] = result.messages
            state.output = result.output
            root_name = self._root_names[0] if self._root_names else "root"
            if root_name not in self._invoked:
                self._invoked.append(root_name)
            for msg in state.messages[min(msg_before, len(state.messages)):]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.function in configured_names and tc.function not in self._invoked:
                            self._invoked.append(tc.function)
                            last_agent = tc.function

        if state.output and state.output.completion:
            self._completed = True

        msg_after = len(state.messages)
        return [AgentTurnResult(
            agent_name=last_agent,
            messages_before=min(msg_before, msg_after),
            messages_after=msg_after,
            submitted=self._completed,
        )]

    def is_complete(self) -> bool:
        return self._completed

    def get_agent_state(self, name: str) -> AgentState | None:
        return None

    def inject_messages(self, messages: list, target: str | None) -> None:
        pass  # messages already in shared TaskState

    @property
    def invoked_agents(self) -> list[str]:
        return self._invoked
