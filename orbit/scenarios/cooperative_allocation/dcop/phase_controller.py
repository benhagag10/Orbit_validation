"""
Two-phase execution controller for DCOP scenarios.

Implements Terrarium/Colosseum's planning+execution phase model:
    - Planning phase: agents communicate via blackboards, action tools blocked
    - Execution phase: agents commit actions, blackboard posting optional

The phase controller wraps (does NOT replace) the existing AgentScheduler.
The orchestrator calls phase_controller.update_state(turn) at the start of
each turn callback. Action tools check DCOPState.current_phase and return
an error during planning if enforce_phase_separation is True.

No changes to AgentScheduler are needed.
"""

from __future__ import annotations

from inspect_ai.util import store_as

from orbit.scenarios.cooperative_allocation.dcop.models import DCOPExecutionConfig
from orbit.scenarios.cooperative_allocation.dcop.state import DCOPState


class DCOPPhaseController:
    """Controls planning/execution phase transitions.

    Each "round" consists of:
        - planning_turns turns of planning phase
        - execution_turns turns of execution phase

    After total_rounds rounds, stays in execution phase.
    """

    def __init__(self, config: DCOPExecutionConfig) -> None:
        self.config = config
        self._turns_per_round = config.planning_turns + config.execution_turns
        self._total_turns = self._turns_per_round * config.total_rounds

    def get_phase(self, turn: int) -> tuple[str, int]:
        """Determine phase and round for a given turn number.

        Args:
            turn: Current turn (0-indexed).

        Returns:
            (phase, round_number) where phase is "planning" or "execution".
        """
        if turn >= self._total_turns:
            # Past all rounds -- stay in execution
            return "execution", self.config.total_rounds - 1

        current_round = turn // self._turns_per_round
        turn_in_round = turn % self._turns_per_round

        if turn_in_round < self.config.planning_turns:
            return "planning", current_round
        return "execution", current_round

    def update_state(self, turn: int) -> None:
        """Update DCOPState with current phase and round.

        Called by the orchestrator at the start of each execute_turn callback.

        Args:
            turn: Current turn (0-indexed).
        """
        dcop_state = store_as(DCOPState)
        phase, round_num = self.get_phase(turn)
        dcop_state.current_phase = phase  # type: ignore[assignment]
        dcop_state.current_round = round_num

    def is_action_allowed(self, turn: int) -> bool:
        """Check if action tools should be allowed on this turn.

        Args:
            turn: Current turn.

        Returns:
            True if actions are allowed (execution phase or enforcement off).
        """
        if not self.config.enforce_phase_separation:
            return True
        phase, _ = self.get_phase(turn)
        return phase == "execution"

    @property
    def enforce_phase_separation(self) -> bool:
        """Whether phase separation is enforced."""
        return self.config.enforce_phase_separation
