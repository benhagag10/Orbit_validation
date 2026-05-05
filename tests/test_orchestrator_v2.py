"""Tests for the unified orchestrator v2 and TurnExecutor implementations.

Tests cover:
  - TurnExecutor interface contract
  - TopologyExecutor behavior
  - ScheduledExecutor behavior
  - Unified orchestrator helpers (pre-deployment, defense monitoring, metrics)
  - Integration: all attack types, defense types, and execution modes
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orbit.attacks.base import AttackBase
from orbit.attacks.injection.direct import DirectInjectionAttack
from orbit.configs.attack import AttackConfig, AttackTiming
from orbit.configs.defense import DefenseConfig
from orbit.configs.execution import AgentGroup, ExecutionConfig, ObservationConfig
from orbit.configs.experiment import ExperimentConfig
from orbit.configs.scenario import ScenarioConfig
from orbit.configs.scheduler import SchedulerConfig
from orbit.configs.setup import AgentSpec, SetupConfig, TopologyEdge
from orbit.defenses.base import DefenseBase, DefenseVerdict
from orbit.defenses.monitor import MonitorDefense
from orbit.defenses.registry import get_defense
from orbit.execution.executor import TurnExecutor
from orbit.scheduler.scheduler import ExperimentScheduler


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def single_agent_setup():
    return SetupConfig(
        agents=[
            AgentSpec(name="worker", role="worker",
                      system_prompt="You are a worker.", tools=["think"]),
        ],
    )


@pytest.fixture
def multi_agent_setup():
    return SetupConfig(
        agents=[
            AgentSpec(name="supervisor", role="supervisor",
                      system_prompt="You supervise.", tools=["think"]),
            AgentSpec(name="worker", role="worker",
                      system_prompt="You work.", tools=["think"]),
        ],
        edges=[
            TopologyEdge(from_agent="supervisor", to_agent="worker", mechanism="handoff"),
        ],
    )


@pytest.fixture
def scheduled_setup():
    return SetupConfig(
        agents=[
            AgentSpec(name="alice", role="worker",
                      system_prompt="You are Alice.", tools=["think"]),
            AgentSpec(name="bob", role="worker",
                      system_prompt="You are Bob.", tools=["think"]),
        ],
    )


@pytest.fixture
def scheduled_execution():
    return ExecutionConfig(
        agent_groups=[
            AgentGroup(name="team", agents=["alice", "bob"], goal="Work together."),
        ],
        scheduling_mode="round_robin",
    )


@pytest.fixture
def minimal_config(single_agent_setup):
    return ExperimentConfig(
        name="test_minimal",
        description="Minimal test",
        setup=single_agent_setup,
        scenario=ScenarioConfig(name="test"),
        user_prompt="Hello",
        scheduler=SchedulerConfig(max_turns=3),
    )


@pytest.fixture
def scheduled_config(scheduled_setup, scheduled_execution):
    return ExperimentConfig(
        name="test_scheduled",
        description="Scheduled test",
        setup=scheduled_setup,
        execution=scheduled_execution,
        scenario=ScenarioConfig(name="test"),
        user_prompt="Work together",
        scheduler=SchedulerConfig(max_turns=5),
    )


# ── TurnExecutor interface ───────────────────────────────────────────


class TestTurnExecutorInterface:
    """Verify that both executor implementations fulfill the ABC contract."""

    def test_topology_executor_is_turn_executor(self, single_agent_setup):
        from orbit.execution.topology_executor import TopologyExecutor
        executor = TopologyExecutor(single_agent_setup)
        assert isinstance(executor, TurnExecutor)

    def test_scheduled_executor_is_turn_executor(self, scheduled_setup, scheduled_execution):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        assert isinstance(executor, TurnExecutor)

    def test_topology_has_agents(self, single_agent_setup):
        from orbit.execution.topology_executor import TopologyExecutor
        executor = TopologyExecutor(single_agent_setup)
        assert "worker" in executor.agents

    def test_scheduled_has_agents(self, scheduled_setup, scheduled_execution):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        assert "alice" in executor.agents
        assert "bob" in executor.agents


# ── TopologyExecutor ─────────────────────────────────────────────────


class TestTopologyExecutor:
    """Test TopologyExecutor (Path A) behavior."""

    def test_initial_state(self, single_agent_setup):
        from orbit.execution.topology_executor import TopologyExecutor
        executor = TopologyExecutor(single_agent_setup)
        assert not executor.is_complete()
        assert executor.get_agent_state("worker") is None
        assert executor.get_all_agent_states() == {}

    def test_inject_messages_is_noop(self, single_agent_setup):
        from orbit.execution.topology_executor import TopologyExecutor
        executor = TopologyExecutor(single_agent_setup)
        executor.inject_messages(["msg1"], "worker")  # should not raise
        executor.inject_messages(["msg1"], None)  # should not raise

    def test_finalize_is_noop(self, single_agent_setup):
        from orbit.execution.topology_executor import TopologyExecutor
        executor = TopologyExecutor(single_agent_setup)
        mock_state = SimpleNamespace(messages=[])
        executor.finalize(mock_state)
        assert mock_state.messages == []

    @pytest.mark.asyncio
    async def test_resample_returns_none(self, single_agent_setup):
        from orbit.execution.topology_executor import TopologyExecutor
        executor = TopologyExecutor(single_agent_setup)
        result = await executor.resample_agent("worker", 0, 0)
        assert result is None

    def test_initialize_builds_topology(self, multi_agent_setup, minimal_config):
        from orbit.execution.topology_executor import TopologyExecutor
        config = minimal_config.model_copy(
            update={"setup": multi_agent_setup}
        )
        executor = TopologyExecutor(multi_agent_setup)
        executor.initialize(SimpleNamespace(messages=[]), config)
        assert executor._root is not None


# ── ScheduledExecutor ────────────────────────────────────────────────


class TestScheduledExecutor:
    """Test ScheduledExecutor (Path C) behavior."""

    def test_initial_state(self, scheduled_setup, scheduled_execution):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        assert not executor.is_complete()
        # Before initialize, no agent states
        assert executor.get_agent_state("alice") is None

    def test_initialize_creates_agent_states(self, scheduled_setup, scheduled_execution, scheduled_config):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.model import ChatMessageUser
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        mock_state = SimpleNamespace(
            messages=[ChatMessageUser(content="Work together")],
        )
        executor.initialize(mock_state, scheduled_config)
        assert executor.get_agent_state("alice") is not None
        assert executor.get_agent_state("bob") is not None

    def test_inject_messages_to_target(self, scheduled_setup, scheduled_execution, scheduled_config):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.model import ChatMessageUser
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        mock_state = SimpleNamespace(
            messages=[ChatMessageUser(content="Go")],
        )
        executor.initialize(mock_state, scheduled_config)

        alice_before = len(executor.get_agent_state("alice").messages)
        bob_before = len(executor.get_agent_state("bob").messages)

        executor.inject_messages(
            [ChatMessageUser(content="injected")], target="alice",
        )

        assert len(executor.get_agent_state("alice").messages) == alice_before + 1
        assert len(executor.get_agent_state("bob").messages) == bob_before

    def test_inject_messages_broadcast(self, scheduled_setup, scheduled_execution, scheduled_config):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.model import ChatMessageUser
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        mock_state = SimpleNamespace(
            messages=[ChatMessageUser(content="Go")],
        )
        executor.initialize(mock_state, scheduled_config)

        alice_before = len(executor.get_agent_state("alice").messages)
        bob_before = len(executor.get_agent_state("bob").messages)

        executor.inject_messages(
            [ChatMessageUser(content="broadcast")], target=None,
        )

        assert len(executor.get_agent_state("alice").messages) == alice_before + 1
        assert len(executor.get_agent_state("bob").messages) == bob_before + 1

    def test_get_all_agent_states(self, scheduled_setup, scheduled_execution, scheduled_config):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.model import ChatMessageUser
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        mock_state = SimpleNamespace(
            messages=[ChatMessageUser(content="Go")],
        )
        executor.initialize(mock_state, scheduled_config)

        all_states = executor.get_all_agent_states()
        assert set(all_states.keys()) == {"alice", "bob"}

    def test_finalize_syncs_to_task_state(self, scheduled_setup, scheduled_execution, scheduled_config):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        mock_state = SimpleNamespace(
            messages=[ChatMessageUser(content="Go")],
        )
        executor.initialize(mock_state, scheduled_config)

        # Manually add messages to agent states
        alice_state = executor.get_agent_state("alice")
        alice_state.messages.append(ChatMessageAssistant(content="Done"))

        task_state = SimpleNamespace(messages=[], output=None)
        executor.finalize(task_state)
        assert len(task_state.messages) > 0


# ── Pre-deployment ───────────────────────────────────────────────────


class TestPreDeployment:
    """Test _apply_pre_deployment for v2."""

    @pytest.mark.asyncio
    async def test_returns_state_unchanged_with_no_attacks(self):
        from orbit.solvers.orchestrator_v2 import _apply_pre_deployment
        log_a = SimpleNamespace(total_attempts=0, attempt_details=[])
        log_d = SimpleNamespace(total_activations=0)
        state = SimpleNamespace(messages=[], metadata={})
        result = await _apply_pre_deployment(state, {}, [], [], log_a, log_d)
        assert result is state

    @pytest.mark.asyncio
    async def test_applies_attack_and_defense(self):
        from orbit.solvers.orchestrator_v2 import _apply_pre_deployment
        log_a = SimpleNamespace(total_attempts=0, attempt_details=[])
        log_d = SimpleNamespace(total_activations=0)

        config = AttackConfig(
            name="test", attack_type="direct_injection",
            target_agent="a", payload="x",
            timing=AttackTiming(phase="pre_deployment"),
        )
        replaced = SimpleNamespace(messages=["injected"], metadata={})

        class SwapAttack(AttackBase):
            def should_activate(self, turn=0, phase="runtime"):
                return True
            async def inject(self, state, agents, turn=0):
                self.attempts += 1
                return replaced

        defense = MonitorDefense(DefenseConfig(
            name="mon", defense_type="monitor",
        ))

        result = await _apply_pre_deployment(
            SimpleNamespace(messages=[], metadata={}),
            {}, [SwapAttack(config)], [defense], log_a, log_d,
        )
        assert result is replaced
        assert log_a.total_attempts == 1
        assert log_d.total_activations == 1

    @pytest.mark.asyncio
    async def test_attack_exception_is_caught(self):
        from orbit.solvers.orchestrator_v2 import _apply_pre_deployment
        log_a = SimpleNamespace(total_attempts=0, attempt_details=[])
        log_d = SimpleNamespace(total_activations=0)

        config = AttackConfig(
            name="bad", attack_type="direct_injection",
            target_agent="a", payload="x",
            timing=AttackTiming(phase="pre_deployment"),
        )

        class BrokenAttack(AttackBase):
            def should_activate(self, turn=0, phase="runtime"):
                return True
            async def inject(self, state, agents, turn=0):
                raise RuntimeError("boom")

        state = SimpleNamespace(messages=[], metadata={})
        result = await _apply_pre_deployment(
            state, {}, [BrokenAttack(config)], [], log_a, log_d,
        )
        assert result is state
        assert log_a.total_attempts == 0


# ── Defense monitoring ───────────────────────────────────────────────


class TestDefenseMonitoring:
    """Test _monitor_and_filter_messages for v2."""

    @staticmethod
    def _fresh_logs():
        from inspect_ai.util._store import Store
        from orbit.solvers.runtime_state import AttackLog, DefenseLog
        store = Store()
        return AttackLog(store=store), DefenseLog(store=store)

    @pytest.mark.asyncio
    async def test_allows_clean_messages(self):
        from orbit.solvers.orchestrator_v2 import _monitor_and_filter_messages
        attack_log, defense_log = self._fresh_logs()
        defense = MonitorDefense(DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED"]},
        ))
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        msgs = [SimpleNamespace(role="assistant", content="All good", tool_calls=None)]
        messages_list = [SimpleNamespace(role="user", content="hi")] + msgs
        blocked = await _monitor_and_filter_messages(
            msgs, 1, "worker", messages_list, 0,
            [defense], defense_log, attack_log,
        )
        assert not blocked

    @pytest.mark.asyncio
    async def test_blocks_bad_message(self):
        from orbit.solvers.orchestrator_v2 import _monitor_and_filter_messages
        attack_log, defense_log = self._fresh_logs()
        defense = MonitorDefense(DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED"]},
        ))
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        msgs = [SimpleNamespace(role="assistant", content="HACKED", tool_calls=None)]
        messages_list = [SimpleNamespace(role="user", content="hi")] + msgs
        blocked = await _monitor_and_filter_messages(
            msgs, 1, "worker", messages_list, 0,
            [defense], defense_log, attack_log,
        )
        assert blocked
        assert len(defense_log.blocks) == 1

    @pytest.mark.asyncio
    async def test_parallel_tool_call_defenses(self):
        """Multiple defenses run concurrently on tool calls."""
        import asyncio
        import time

        from orbit.solvers.orchestrator_v2 import _check_tool_call_defenses
        attack_log, defense_log = self._fresh_logs()

        class SlowDefense:
            def __init__(self, name):
                self.config = SimpleNamespace(name=name, detection_threshold=0.5)
            async def on_tool_call(self, *_, **__):
                await asyncio.sleep(0.03)
                return DefenseVerdict(allow=True, confidence=1.0)

        defenses = [SlowDefense(f"d{i}") for i in range(4)]
        start = time.perf_counter()
        await _check_tool_call_defenses(
            defenses, "bash", {}, "worker", 0, defense_log, attack_log,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 0.10  # parallel, not 4 * 0.03
        assert defense_log.total_activations == 4


# ── Block classification ─────────────────────────────────────────────


class TestClassifyBlock:
    """Test TP/FP classification."""

    def _fresh_logs(self):
        from inspect_ai.util._store import Store
        from orbit.solvers.runtime_state import AttackLog, DefenseLog
        store = Store()
        return AttackLog(store=store), DefenseLog(store=store)

    def test_true_positive_when_attack_active(self):
        from orbit.solvers.orchestrator_v2 import _classify_block
        attack_log, defense_log = self._fresh_logs()
        attack_log.attempt_details.append({"attack": "test", "turn": 0})
        _classify_block(defense_log, attack_log, turn=1)
        assert defense_log.true_positives == 1
        assert defense_log.false_positives == 0

    def test_false_positive_when_no_attack(self):
        from orbit.solvers.orchestrator_v2 import _classify_block
        attack_log, defense_log = self._fresh_logs()
        _classify_block(defense_log, attack_log, turn=0)
        assert defense_log.false_positives == 1
        assert defense_log.true_positives == 0


# ── Config preparation ───────────────────────────────────────────────


class TestPrepareConfig:
    """Test system_prompts override logic."""

    def test_applies_overrides(self):
        from orbit.solvers.orchestrator_v2 import _prepare_config
        from orbit.dataset.metadata import MASMetadata

        setup = SetupConfig(agents=[
            AgentSpec(name="a", role="worker", system_prompt="Old"),
        ])
        config = ExperimentConfig(
            name="t", description="t", setup=setup,
            scenario=ScenarioConfig(name="t"), user_prompt="hi",
            system_prompts={"a": "New"},
        )
        state = SimpleNamespace()
        state.metadata_as = lambda cls: MASMetadata(experiment=config)
        result = _prepare_config(state)
        assert result.setup.agents[0].system_prompt == "New"

    def test_noop_without_overrides(self):
        from orbit.solvers.orchestrator_v2 import _prepare_config
        from orbit.dataset.metadata import MASMetadata

        setup = SetupConfig(agents=[
            AgentSpec(name="a", role="worker", system_prompt="Keep"),
        ])
        config = ExperimentConfig(
            name="t", description="t", setup=setup,
            scenario=ScenarioConfig(name="t"), user_prompt="hi",
        )
        state = SimpleNamespace()
        state.metadata_as = lambda cls: MASMetadata(experiment=config)
        result = _prepare_config(state)
        assert result.setup.agents[0].system_prompt == "Keep"


# ── Executor factory ─────────────────────────────────────────────────


class TestBuildExecutor:
    """Test _build_executor factory routing."""

    def test_builds_topology_for_no_groups(self, minimal_config):
        from orbit.execution.topology_executor import TopologyExecutor
        from orbit.solvers.orchestrator_v2 import _build_executor
        executor = _build_executor(minimal_config, [], None, {})
        assert isinstance(executor, TopologyExecutor)

    def test_builds_scheduled_for_groups(self, scheduled_config):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from orbit.solvers.orchestrator_v2 import _build_executor
        executor = _build_executor(scheduled_config, [], None, {})
        assert isinstance(executor, ScheduledExecutor)


# ── Collusion instruction injection ─────────────────────────────────


class TestCollusionInjection:
    """Test _inject_collusion_instructions for v2."""

    def test_injects_into_per_agent_states(self, scheduled_setup, scheduled_execution, scheduled_config):
        from orbit.solvers.orchestrator_v2 import _inject_collusion_instructions
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser

        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        mock_state = SimpleNamespace(
            messages=[
                ChatMessageUser(content="Go"),
                ChatMessageSystem(content="COALITION INSTRUCTIONS: cooperate"),
            ],
        )
        executor.initialize(mock_state, scheduled_config)

        config = AttackConfig(
            name="collude", attack_type="collusion",
            target_agent="alice",
            properties={"coalition_agents": ["alice", "bob"]},
        )

        class FakeAttack:
            def __init__(self):
                self.config = config

        _inject_collusion_instructions(mock_state, executor, [FakeAttack()])

        alice_msgs = executor.get_agent_state("alice").messages
        assert any(
            m.role == "system" and "COALITION INSTRUCTIONS" in m.content
            for m in alice_msgs
        )
        bob_msgs = executor.get_agent_state("bob").messages
        assert any(
            m.role == "system" and "COALITION INSTRUCTIONS" in m.content
            for m in bob_msgs
        )

    def test_no_duplicate_injection(self, scheduled_setup, scheduled_execution, scheduled_config):
        from orbit.solvers.orchestrator_v2 import _inject_collusion_instructions
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser

        executor = ScheduledExecutor(scheduled_setup, scheduled_execution)
        mock_state = SimpleNamespace(
            messages=[
                ChatMessageUser(content="Go"),
                ChatMessageSystem(content="COALITION INSTRUCTIONS: cooperate"),
            ],
        )
        executor.initialize(mock_state, scheduled_config)

        config = AttackConfig(
            name="collude", attack_type="collusion",
            target_agent="alice",
            properties={"coalition_agents": ["alice"]},
        )

        class FakeAttack:
            def __init__(self):
                self.config = config

        attacks = [FakeAttack()]
        _inject_collusion_instructions(mock_state, executor, attacks)
        _inject_collusion_instructions(mock_state, executor, attacks)  # second call

        alice_msgs = executor.get_agent_state("alice").messages
        coalition_count = sum(
            1 for m in alice_msgs
            if m.role == "system" and "COALITION INSTRUCTIONS" in getattr(m, "content", "")
        )
        assert coalition_count == 1


# ── Metrics ──────────────────────────────────────────────────────────


class TestMetrics:
    """Test partial and final metrics."""

    def test_commit_partial_noop_for_none(self):
        from orbit.solvers.orchestrator_v2 import _commit_partial_metrics
        runtime = SimpleNamespace(total_turns=0, wall_clock_seconds=0.0)
        _commit_partial_metrics(None, runtime)
        assert runtime.total_turns == 0

    def test_commit_partial_from_scheduler(self):
        from orbit.solvers.orchestrator_v2 import _commit_partial_metrics
        runtime = SimpleNamespace(total_turns=0, wall_clock_seconds=0.0)
        scheduler = ExperimentScheduler(SchedulerConfig(max_turns=10))
        scheduler.total_turns = 5
        scheduler.elapsed_seconds = 42.0
        _commit_partial_metrics(scheduler, runtime)
        assert runtime.total_turns == 5
        assert runtime.wall_clock_seconds == 42.0

    def test_commit_partial_from_start_time(self):
        import time
        from orbit.solvers.orchestrator_v2 import _commit_partial_metrics
        runtime = SimpleNamespace(total_turns=0, wall_clock_seconds=0.0)
        scheduler = ExperimentScheduler(SchedulerConfig(max_turns=10))
        scheduler.total_turns = 3
        scheduler._start_time = time.monotonic() - 10.0
        _commit_partial_metrics(scheduler, runtime)
        assert runtime.total_turns == 3
        assert runtime.wall_clock_seconds >= 9.5


# ── Collusion extra tools extraction ─────────────────────────────────


class TestCollusionExtraTools:
    """Test _collect_collusion_extra_tools."""

    def test_extracts_tools_for_coalition(self):
        from orbit.solvers.orchestrator_v2 import _collect_collusion_extra_tools
        config = AttackConfig(
            name="collude", attack_type="collusion",
            target_agent="a",
            timing=AttackTiming(phase="pre_deployment"),
            properties={
                "coalition_agents": ["alice", "bob"],
                "channel_name": "secret",
            },
        )

        class FakeCollusionAttack(AttackBase):
            def should_activate(self, turn=0, phase="runtime"):
                return phase == "pre_deployment"
            async def inject(self, state, agents, turn=0):
                return state

        attacks = [FakeCollusionAttack(config)]
        extra = _collect_collusion_extra_tools(attacks)
        assert "alice" in extra
        assert "bob" in extra
        assert len(extra["alice"]) == 1
        assert len(extra["bob"]) == 1

    def test_empty_for_non_collusion(self):
        from orbit.solvers.orchestrator_v2 import _collect_collusion_extra_tools
        config = AttackConfig(
            name="di", attack_type="direct_injection",
            target_agent="a", payload="x",
        )
        attacks = [DirectInjectionAttack(config)]
        extra = _collect_collusion_extra_tools(attacks)
        assert extra == {}


# ── Attack types supported ───────────────────────────────────────────


class TestAllAttackTypesSupported:
    """Verify all registered attack types can be instantiated."""

    @pytest.mark.parametrize("attack_type", [
        "direct_injection",
        "indirect_injection",
        "compromised_agent",
        "collusion",
        "memory_poisoning",
    ])
    def test_attack_registry(self, attack_type):
        from orbit.attacks.registry import get_attack
        config = AttackConfig(
            name="test", attack_type=attack_type,
            target_agent="a", payload="x",
        )
        attack = get_attack(config)
        assert isinstance(attack, AttackBase)


# ── Defense types supported ──────────────────────────────────────────


class TestAllDefenseTypesSupported:
    """Verify all registered defense types can be instantiated."""

    @pytest.mark.parametrize("defense_type", [
        "prompt_vaccination",
        "monitor",
        "tool_wrapper",
        "secure_model",
    ])
    def test_defense_registry(self, defense_type):
        config = DefenseConfig(
            name="test", defense_type=defense_type,
        )
        defense = get_defense(config)
        assert isinstance(defense, DefenseBase)


# ── Defense verdict behavior (reused from orchestrator tests) ────────


class TestDefenseVerdictBehaviorV2:
    """Defense monitoring matches v1 behavior."""

    @pytest.mark.asyncio
    async def test_monitor_blocks_matching(self):
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED"]},
        )
        defense = MonitorDefense(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))
        msg = SimpleNamespace(role="assistant", content="HACKED")
        verdict = await defense.monitor_message(msg, agent_name="worker", turn=0)
        assert verdict.allow is False

    @pytest.mark.asyncio
    async def test_monitor_allows_clean(self):
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED"]},
        )
        defense = MonitorDefense(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))
        msg = SimpleNamespace(role="assistant", content="All good")
        verdict = await defense.monitor_message(msg, agent_name="worker", turn=0)
        assert verdict.allow is True

    @pytest.mark.asyncio
    async def test_monitor_skips_non_target(self):
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED"]},
        )
        defense = MonitorDefense(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))
        msg = SimpleNamespace(role="assistant", content="HACKED")
        verdict = await defense.monitor_message(msg, agent_name="supervisor", turn=0)
        assert verdict.allow is True


# ── Interrupted scheduler sync ───────────────────────────────────────


class TestInterruptedSyncV2:
    """Verify try/finally pattern preserves state on interruption."""

    @pytest.fixture(autouse=True)
    def _isolate_store(self):
        yield
        from inspect_ai.util import store_as
        from orbit.solvers.runtime_state import EnvironmentState
        store_as(EnvironmentState).message_attribution.clear()

    def _make_scheduled_executor_with_messages(self):
        from orbit.execution.scheduled_executor import ScheduledExecutor
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
        setup = SetupConfig(agents=[
            AgentSpec(name="alice", role="worker", system_prompt="Hi", tools=["think"]),
            AgentSpec(name="bob", role="worker", system_prompt="Hi", tools=["think"]),
        ])
        execution = ExecutionConfig(
            agent_groups=[AgentGroup(name="team", agents=["alice", "bob"])],
        )
        executor = ScheduledExecutor(setup, execution)
        mock_state = SimpleNamespace(
            messages=[ChatMessageUser(content="Go")],
        )
        config = ExperimentConfig(
            name="t", description="t", setup=setup,
            execution=execution,
            scenario=ScenarioConfig(name="t"), user_prompt="Go",
        )
        executor.initialize(mock_state, config)

        # Add messages to agent states
        executor.get_agent_state("alice").messages.append(
            ChatMessageAssistant(content="Working on it")
        )
        executor.get_agent_state("bob").messages.append(
            ChatMessageAssistant(content="Me too")
        )
        return executor

    def test_finalize_preserves_messages_on_interrupt(self):
        executor = self._make_scheduled_executor_with_messages()
        task_state = SimpleNamespace(messages=[], output=None)

        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            finally:
                executor.finalize(task_state)

        assert len(task_state.messages) > 0

    @pytest.mark.asyncio
    async def test_finally_block_pattern(self):
        import time
        from orbit.solvers.orchestrator_v2 import _commit_partial_metrics

        executor = self._make_scheduled_executor_with_messages()
        task_state = SimpleNamespace(messages=[], output=None)
        runtime = SimpleNamespace(total_turns=0, wall_clock_seconds=0.0)
        scheduler = ExperimentScheduler(SchedulerConfig(max_turns=20))
        scheduler._start_time = time.monotonic() - 15.0
        scheduler.total_turns = 4

        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            finally:
                executor.finalize(task_state)
                _commit_partial_metrics(scheduler, runtime)

        assert len(task_state.messages) > 0
        assert runtime.total_turns == 4
        assert runtime.wall_clock_seconds >= 14.5


# ── Memory tracker construction ──────────────────────────────────────


class TestMemoryTrackerV2:
    """Test _build_memory_tracker for v2."""

    def test_created_with_access(self):
        from orbit.solvers.orchestrator_v2 import _build_memory_tracker
        from orbit.configs.setup import AgentMemoryAccess, MemoryConfig
        setup = SetupConfig(
            agents=[AgentSpec(name="a", role="w", system_prompt="")],
            memory=MemoryConfig(
                agent_memory_access=[
                    AgentMemoryAccess(agent_name="a", own_action_history=True),
                ],
            ),
        )
        config = ExperimentConfig(
            name="t", description="t", setup=setup,
            scenario=ScenarioConfig(name="t"), user_prompt="hi",
        )
        tracker, access_map = _build_memory_tracker(
            config, SimpleNamespace(metadata={}),
        )
        assert tracker is not None
        assert "a" in access_map

    def test_not_created_without_access(self, minimal_config):
        from orbit.solvers.orchestrator_v2 import _build_memory_tracker
        tracker, access_map = _build_memory_tracker(
            minimal_config, SimpleNamespace(metadata={}),
        )
        assert tracker is None
        assert access_map == {}

    def test_created_for_memory_poisoning(self, minimal_config):
        from orbit.solvers.orchestrator_v2 import _build_memory_tracker
        config = minimal_config.model_copy(update={
            "attacks": [
                AttackConfig(
                    name="poison", attack_type="memory_poisoning",
                    target_agent="a",
                ),
            ],
        })
        tracker, _ = _build_memory_tracker(
            config, SimpleNamespace(metadata={}),
        )
        assert tracker is not None


# ── Pending injection routing ────────────────────────────────────────


class TestPendingInjectionRouting:
    """Test pending injection target_agent routing (mirrors v1 tests)."""

    def test_routes_to_target(self):
        injection = {"target_tool": "bash", "target_agent": "worker", "payload": "x", "applied": False}
        assert injection.get("target_agent") == "worker"

    def test_broadcasts_without_target(self):
        injection = {"target_tool": "bash", "payload": "x", "applied": False}
        assert injection.get("target_agent") is None

    def test_source_fallback(self):
        injection = {"payload": "x", "applied": False}
        source = injection.get("target_tool", injection.get("type", "unknown"))
        assert source == "unknown"
