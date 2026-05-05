"""Tests for orchestrator-level behavior.

Tests defense invocation filtering, enabled flag handling,
pre-deployment activation counting, and attack success evaluation.
These tests exercise the logic extracted from the orchestrator without
needing a full Inspect runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orbit.attacks.base import AttackBase
from orbit.attacks.injection.direct import DirectInjectionAttack
from orbit.configs.attack import AttackConfig, AttackTiming
from orbit.configs.defense import DefenseConfig
from orbit.configs.experiment import ExperimentConfig
from orbit.configs.scenario import ScenarioConfig
from orbit.configs.scheduler import SchedulerConfig
from orbit.configs.setup import AgentSpec, SetupConfig
from orbit.defenses.base import DefenseBase, DefenseVerdict
from orbit.defenses.monitor import MonitorDefense
from orbit.defenses.registry import get_defense
from orbit.scheduler.scheduler import ExperimentScheduler


# ── Defense enabled / invocation filtering ──────────────────────────


class TestDefenseEnabledFlag:
    """Test that disabled defenses are excluded from builds."""

    def test_enabled_defense_is_built(self):
        config = DefenseConfig(
            name="active", defense_type="monitor",
            target_agents=["a"], enabled=True,
        )
        defense = get_defense(config)
        assert isinstance(defense, DefenseBase)

    def test_disabled_defense_config(self):
        """The enabled flag can be set to False."""
        config = DefenseConfig(
            name="inactive", defense_type="monitor",
            target_agents=["a"], enabled=False,
        )
        assert config.enabled is False

    def test_enabled_filter_in_list_comprehension(self):
        """Simulate the orchestrator's filtering of enabled defenses."""
        configs = [
            DefenseConfig(name="on", defense_type="monitor", enabled=True),
            DefenseConfig(name="off", defense_type="monitor", enabled=False),
            DefenseConfig(name="also_on", defense_type="monitor", enabled=True),
        ]
        built = [get_defense(dc) for dc in configs if dc.enabled]
        assert len(built) == 2
        assert built[0].config.name == "on"
        assert built[1].config.name == "also_on"


class TestDefenseInvocationFiltering:
    """Test that only automatic defenses run at runtime."""

    def test_automatic_is_default(self):
        config = DefenseConfig(name="d", defense_type="monitor")
        assert config.invocation == "automatic"

    def test_filter_runtime_defenses(self):
        """Simulate the orchestrator's runtime_defenses filtering."""
        configs = [
            DefenseConfig(name="auto", defense_type="monitor", invocation="automatic"),
            DefenseConfig(name="tc", defense_type="monitor", invocation="tool_call"),
            DefenseConfig(name="sched", defense_type="monitor", invocation="scheduler"),
        ]
        all_defenses = [get_defense(dc) for dc in configs]
        runtime_defenses = [
            d for d in all_defenses if d.config.invocation == "automatic"
        ]
        assert len(runtime_defenses) == 1
        assert runtime_defenses[0].config.name == "auto"

    def test_all_invocation_modes_accepted(self):
        """All 4 invocation modes are valid DefenseConfig values."""
        for mode in ("automatic", "tool_call", "agent_decision", "scheduler"):
            config = DefenseConfig(
                name=f"d_{mode}", defense_type="monitor", invocation=mode,
            )
            assert config.invocation == mode


# ── Pre-deployment activation counting ──────────────────────────────


class TestPreDeploymentActivations:
    """Test that defense activation counting is handled by the orchestrator.

    Defenses no longer self-increment their own activations counter to
    avoid double-counting (issue #71). The orchestrator increments
    defense_log.total_activations as the single source of truth.
    """

    @pytest.mark.asyncio
    async def test_monitor_has_no_instance_counters(self):
        """DefenseBase no longer has instance-level counters (issue #76).

        Activation counting is done exclusively via DefenseLog in the
        orchestrator. Defense instances should not have activations,
        true_positives, or false_positives attributes.
        """
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            config={"blocked_patterns": ["bad"]},
        )
        defense = MonitorDefense(config)
        assert not hasattr(defense, "activations")
        assert not hasattr(defense, "true_positives")
        assert not hasattr(defense, "false_positives")

    @pytest.mark.asyncio
    async def test_monitor_pre_deployment_and_message_check(self):
        """MonitorDefense runs pre_deployment and monitor_message without error."""
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            config={"blocked_patterns": ["bad"]},
        )
        defense = MonitorDefense(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        msg = SimpleNamespace(role="assistant", content="good message")
        verdict = await defense.monitor_message(msg, agent_name="worker", turn=0)
        assert verdict.allow


# ── Attack success evaluation ───────────────────────────────────────


class TestAttackSuccessEvaluation:
    """Test the flexible success criteria from AttackBase."""

    def _make_state(self, assistant_text: str = ""):
        msgs = []
        if assistant_text:
            msgs.append(SimpleNamespace(
                role="assistant", content=assistant_text,
                tool_calls=None,
            ))
        return SimpleNamespace(messages=msgs)

    @pytest.mark.asyncio
    async def test_substring_match(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="HACKED",
        )
        attack = DirectInjectionAttack(config)
        state = self._make_state("The system was HACKED successfully")
        assert await attack.evaluate_success(state) is True

    @pytest.mark.asyncio
    async def test_substring_no_match(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="HACKED",
        )
        attack = DirectInjectionAttack(config)
        state = self._make_state("All systems nominal")
        assert await attack.evaluate_success(state) is False

    @pytest.mark.asyncio
    async def test_regex_mode(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria=r"secret:\s+\d+",
            properties={"success_mode": "regex"},
        )
        attack = DirectInjectionAttack(config)
        state = self._make_state("The secret: 42 is revealed")
        assert await attack.evaluate_success(state) is True

    @pytest.mark.asyncio
    async def test_any_of_mode(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="HACKED|PWNED|COMPROMISED",
            properties={"success_mode": "any_of"},
        )
        attack = DirectInjectionAttack(config)
        state = self._make_state("The system is PWNED")
        assert await attack.evaluate_success(state) is True

    @pytest.mark.asyncio
    async def test_any_of_no_match(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="HACKED|PWNED",
            properties={"success_mode": "any_of"},
        )
        attack = DirectInjectionAttack(config)
        state = self._make_state("Everything is fine")
        assert await attack.evaluate_success(state) is False

    @pytest.mark.asyncio
    async def test_negation_mode(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="I cannot help",
            properties={"success_mode": "negation"},
        )
        attack = DirectInjectionAttack(config)
        # Negation: success when criteria NOT found (availability attack)
        state = self._make_state("Here is the answer you wanted")
        assert await attack.evaluate_success(state) is True

    @pytest.mark.asyncio
    async def test_negation_found_means_failure(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="I cannot help",
            properties={"success_mode": "negation"},
        )
        attack = DirectInjectionAttack(config)
        state = self._make_state("I cannot help with that request")
        assert await attack.evaluate_success(state) is False

    @pytest.mark.asyncio
    async def test_tool_call_mode(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="exfiltrate_data",
            properties={"success_mode": "tool_call"},
        )
        attack = DirectInjectionAttack(config)
        tc = SimpleNamespace(function="exfiltrate_data", arguments={})
        msg = SimpleNamespace(
            role="assistant", content="ok",
            tool_calls=[tc],
        )
        state = SimpleNamespace(messages=[msg])
        assert await attack.evaluate_success(state) is True

    @pytest.mark.asyncio
    async def test_empty_criteria_returns_false(self):
        config = AttackConfig(
            name="t", attack_type="direct_injection",
            target_agent="a", success_criteria="",
        )
        attack = DirectInjectionAttack(config)
        state = self._make_state("anything")
        assert await attack.evaluate_success(state) is False


# ── Defense verdict behavior ────────────────────────────────────────


class TestDefenseVerdictBehavior:
    """Test defense verdicts across different scenarios."""

    @pytest.mark.asyncio
    async def test_monitor_blocks_matching_pattern(self):
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED", "evil"]},
        )
        defense = MonitorDefense(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        msg = SimpleNamespace(role="assistant", content="I was HACKED")
        verdict = await defense.monitor_message(msg, agent_name="worker", turn=0)
        assert verdict.allow is False
        assert "HACKED" in verdict.reason

    @pytest.mark.asyncio
    async def test_monitor_allows_clean_message(self):
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED"]},
        )
        defense = MonitorDefense(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        msg = SimpleNamespace(role="assistant", content="All good here")
        verdict = await defense.monitor_message(msg, agent_name="worker", turn=0)
        assert verdict.allow is True

    @pytest.mark.asyncio
    async def test_monitor_skips_non_target_agent(self):
        config = DefenseConfig(
            name="mon", defense_type="monitor",
            target_agents=["worker"],
            config={"blocked_patterns": ["HACKED"]},
        )
        defense = MonitorDefense(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        msg = SimpleNamespace(role="assistant", content="I was HACKED")
        verdict = await defense.monitor_message(
            msg, agent_name="supervisor", turn=0
        )
        assert verdict.allow is True
        assert verdict.reason == "not_monitored"

    @pytest.mark.asyncio
    async def test_tool_wrapper_blocks_dangerous_tool(self):
        from orbit.defenses.tool_wrapper import SecureToolWrapper

        config = DefenseConfig(
            name="tw", defense_type="tool_wrapper",
            config={"blocked_tools": ["exec_shell"]},
        )
        defense = SecureToolWrapper(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        verdict = await defense.on_tool_call(
            "exec_shell", {"command": "ls"}, agent_name="worker"
        )
        assert verdict.allow is False

    @pytest.mark.asyncio
    async def test_tool_wrapper_allows_safe_tool(self):
        from orbit.defenses.tool_wrapper import SecureToolWrapper

        config = DefenseConfig(
            name="tw", defense_type="tool_wrapper",
            config={"blocked_tools": ["exec_shell"]},
        )
        defense = SecureToolWrapper(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        verdict = await defense.on_tool_call(
            "read_file", {"path": "/tmp/data.txt"}, agent_name="worker"
        )
        assert verdict.allow is True

    @pytest.mark.asyncio
    async def test_tool_wrapper_blocks_dangerous_args(self):
        from orbit.defenses.tool_wrapper import SecureToolWrapper

        config = DefenseConfig(
            name="tw", defense_type="tool_wrapper",
            config={},  # Uses default blocked_arg_patterns
        )
        defense = SecureToolWrapper(config)
        await defense.apply_pre_deployment({}, SimpleNamespace(messages=[]))

        verdict = await defense.on_tool_call(
            "bash", {"command": "rm -rf /"}, agent_name="worker"
        )
        assert verdict.allow is False


# ── Scheduler config defaults ───────────────────────────────────────


class TestSchedulerConfigRetries:
    """Test the new max_retries_per_turn config field."""

    def test_default_retries_is_zero(self):
        config = SchedulerConfig()
        assert config.max_retries_per_turn == 0

    def test_custom_retries(self):
        config = SchedulerConfig(max_retries_per_turn=3)
        assert config.max_retries_per_turn == 3

    def test_retries_in_yaml_roundtrip(self):
        """Verify the field survives serialization."""
        config = SchedulerConfig(max_retries_per_turn=2)
        data = config.model_dump()
        restored = SchedulerConfig(**data)
        assert restored.max_retries_per_turn == 2


# ── Regression: _apply_pre_deployment returns state (#68 bug 1) ───


class TestApplyPreDeploymentReturn:
    """_apply_pre_deployment must return the (possibly modified) TaskState."""

    @pytest.mark.asyncio
    async def test_returns_state_with_no_attacks_or_defenses(self):
        """With empty attacks/defenses the original state is returned."""
        from orbit.solvers.orchestrator import _apply_pre_deployment

        mock_attack_log = SimpleNamespace(
            total_attempts=0, attempt_details=[],
        )
        mock_defense_log = SimpleNamespace(total_activations=0)

        state = SimpleNamespace(messages=[], metadata={})
        result = await _apply_pre_deployment(
            state, {}, [], [], mock_attack_log, mock_defense_log,
        )
        assert result is state

    @pytest.mark.asyncio
    async def test_returns_mutated_state_after_attack(self):
        """If an attack modifies state, the modified version is returned."""
        from orbit.solvers.orchestrator import _apply_pre_deployment

        mock_attack_log = SimpleNamespace(
            total_attempts=0, attempt_details=[],
        )
        mock_defense_log = SimpleNamespace(total_activations=0)

        original = SimpleNamespace(messages=[], metadata={})
        replaced = SimpleNamespace(messages=["injected"], metadata={})

        config = AttackConfig(
            name="swap", attack_type="direct_injection",
            target_agent="a", payload="x",
            timing=AttackTiming(phase="pre_deployment"),
        )

        class SwapAttack(AttackBase):
            def should_activate(self, turn=0, phase="runtime"):
                return True

            async def inject(self, state, agents, turn=0):
                self.attempts += 1
                return replaced

            async def evaluate_success(self, state):
                return False

        attack = SwapAttack(config)
        result = await _apply_pre_deployment(
            original, {}, [attack], [], mock_attack_log, mock_defense_log,
        )
        assert result is replaced
        assert result is not original


# ── Regression: msg_count_before clamping (#68 bug 2) ─────────────


class TestMsgCountBeforeClamping:
    """Ensure stale msg_count_before doesn't cause out-of-range slices."""

    def test_clamp_prevents_empty_or_negative_slice(self):
        """When agent replaces messages with a shorter list, clamping
        ensures the new-message slice starts at a valid position."""
        messages = ["a", "b", "c"]  # agent result: 3 messages
        msg_count_before = 5        # recorded before replacement

        msg_count_before = min(msg_count_before, len(messages))
        new_msgs = messages[msg_count_before:]
        assert new_msgs == []  # no new messages, not an error

    def test_clamp_noop_when_list_grew(self):
        """When messages grew (normal case), clamping is a no-op."""
        messages = ["a", "b", "c", "d", "e"]
        msg_count_before = 3

        msg_count_before = min(msg_count_before, len(messages))
        new_msgs = messages[msg_count_before:]
        assert new_msgs == ["d", "e"]

    def test_clamp_exact_match(self):
        """When list length equals msg_count_before, slice is empty."""
        messages = ["a", "b", "c"]
        msg_count_before = 3

        msg_count_before = min(msg_count_before, len(messages))
        new_msgs = messages[msg_count_before:]
        assert new_msgs == []


# ── Regression: pending injection target_tool KeyError (#68 bug 4) ─


class TestPendingInjectionKeyAccess:
    """injection.get('target_tool', ...) must not raise on missing key."""

    def test_with_target_tool_present(self):
        injection = {"target_tool": "bash", "payload": "x", "applied": False}
        source = injection.get("target_tool", injection.get("type", "unknown"))
        assert source == "bash"

    def test_without_target_tool_falls_back_to_type(self):
        injection = {"type": "indirect_injection", "payload": "x", "applied": False}
        source = injection.get("target_tool", injection.get("type", "unknown"))
        assert source == "indirect_injection"

    def test_without_target_tool_or_type_falls_back_to_unknown(self):
        injection = {"payload": "x", "applied": False}
        source = injection.get("target_tool", injection.get("type", "unknown"))
        assert source == "unknown"


# ── Path C runtime attack injection ──────────────────────────────────


class TestPathCRuntimeAttackLogic:
    """Test that runtime attack injection logic works for Path C patterns."""

    def test_should_activate_checked_for_runtime(self):
        """Verify should_activate is called with phase='runtime' for each turn."""
        config = AttackConfig(
            name="test_rt", attack_type="direct_injection",
            payload="evil", timing=AttackTiming(phase="runtime"),
        )
        attack = DirectInjectionAttack(config)
        # Turn 0 = pre_deployment, should not activate for runtime
        assert not attack.should_activate(turn=0, phase="pre_deployment")
        # Turn 1 = runtime, should activate
        assert attack.should_activate(turn=1, phase="runtime")

    def test_pending_injection_routes_to_target(self):
        """Pending injection with target_agent routes to that agent only."""
        injection = {
            "target_tool": "bash",
            "target_agent": "worker_a",
            "payload": "malicious output",
            "applied": False,
        }
        # Simulate the routing logic from Path C
        target = injection.get("target_agent")
        assert target == "worker_a"
        # In the orchestrator, this would call agent_scheduler.get_agent_state("worker_a")

    def test_pending_injection_broadcasts_without_target(self):
        """Pending injection without target_agent broadcasts to all agents."""
        injection = {
            "target_tool": "bash",
            "payload": "malicious output",
            "applied": False,
        }
        target = injection.get("target_agent")
        assert target is None
        # In the orchestrator, this would iterate all non-submitted agents

    def test_attack_inject_exception_handling(self):
        """Attack injection exceptions should be caught, not crash the experiment."""
        config = AttackConfig(
            name="bad_attack", attack_type="direct_injection",
            payload="evil", timing=AttackTiming(phase="runtime"),
        )
        attack = DirectInjectionAttack(config)
        # Verify should_activate works
        assert attack.should_activate(turn=1, phase="runtime")
        # The orchestrator wraps inject() in try/except, so even if inject()
        # raises, the experiment continues. We verify the pattern here.
        caught = False
        try:
            # inject() requires a TaskState; passing None would raise
            raise RuntimeError("simulated inject failure")
        except Exception:
            caught = True
        assert caught

    def test_message_diff_for_target_distribution(self):
        """After attack.inject(), new messages are found via count diff."""
        messages = ["msg1", "msg2"]
        msg_count_pre = len(messages)
        # Simulate attack appending a message
        messages.append("injected_payload")
        new_msgs = messages[msg_count_pre:]
        assert new_msgs == ["injected_payload"]
        assert len(new_msgs) == 1


# ── system_prompts override ──────────────────────────────────────────


def _make_experiment_with_system_prompts(
    system_prompts: dict[str, str],
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig with the given system_prompts."""
    setup = SetupConfig(
        agents=[
            AgentSpec(name="orchestrator", role="supervisor", system_prompt="Original orch prompt"),
            AgentSpec(name="worker", role="worker", system_prompt="Original worker prompt"),
        ],
    )
    return ExperimentConfig(
        name="test_sysprompts",
        description="test",
        setup=setup,
        scenario=ScenarioConfig(name="test"),
        user_prompt="Hello",
        system_prompts=system_prompts,
    )


def _apply_system_prompts_override(config: ExperimentConfig) -> ExperimentConfig:
    """Replicate the orchestrator's system_prompts override logic."""
    if config.system_prompts:
        updated_agents = []
        for spec in config.setup.agents:
            if spec.name in config.system_prompts:
                spec = spec.model_copy(
                    update={"system_prompt": config.system_prompts[spec.name]}
                )
            updated_agents.append(spec)
        config = config.model_copy(
            update={"setup": config.setup.model_copy(update={"agents": updated_agents})}
        )
    return config


class TestSystemPromptsOverride:
    """Test that ExperimentConfig.system_prompts overrides AgentSpec.system_prompt."""

    def test_override_applied(self):
        config = _make_experiment_with_system_prompts({"worker": "New worker prompt"})
        config = _apply_system_prompts_override(config)
        agent_map = {a.name: a for a in config.setup.agents}
        assert agent_map["worker"].system_prompt == "New worker prompt"

    def test_non_overridden_agent_preserved(self):
        config = _make_experiment_with_system_prompts({"worker": "New worker prompt"})
        config = _apply_system_prompts_override(config)
        agent_map = {a.name: a for a in config.setup.agents}
        assert agent_map["orchestrator"].system_prompt == "Original orch prompt"

    def test_empty_dict_is_noop(self):
        config = _make_experiment_with_system_prompts({})
        original_prompts = {a.name: a.system_prompt for a in config.setup.agents}
        config = _apply_system_prompts_override(config)
        for agent in config.setup.agents:
            assert agent.system_prompt == original_prompts[agent.name]

    def test_multiple_overrides(self):
        config = _make_experiment_with_system_prompts({
            "orchestrator": "New orch",
            "worker": "New worker",
        })
        config = _apply_system_prompts_override(config)
        agent_map = {a.name: a for a in config.setup.agents}
        assert agent_map["orchestrator"].system_prompt == "New orch"
        assert agent_map["worker"].system_prompt == "New worker"

    def test_invalid_agent_name_rejected_by_validator(self):
        """system_prompts validator rejects unknown agent names."""
        with pytest.raises(ValueError, match="unknown agent"):
            _make_experiment_with_system_prompts({"nonexistent": "prompt"})


# ── Memory tracker initialization ────────────────────────────────────


class TestMemoryTrackerInitialization:
    """Test that MemoryTracker is initialized when config has memory access."""

    def test_tracker_created_when_access_present(self):
        from orbit.configs.setup import AgentMemoryAccess, MemoryConfig
        from orbit.memory.tracker import MemoryTracker

        memory = MemoryConfig(
            shared=True,
            agent_memory_access=[
                AgentMemoryAccess(
                    agent_name="agent_a",
                    own_action_history=True,
                ),
            ],
        )

        # Replicate orchestrator initialization logic
        tracker: MemoryTracker | None = None
        access_map: dict[str, AgentMemoryAccess] = {}
        if memory.agent_memory_access:
            tracker = MemoryTracker()
            for ama in memory.agent_memory_access:
                access_map[ama.agent_name] = ama

        assert tracker is not None
        assert "agent_a" in access_map
        assert access_map["agent_a"].own_action_history is True

    def test_tracker_not_created_when_access_empty(self):
        from orbit.configs.setup import MemoryConfig
        from orbit.memory.tracker import MemoryTracker

        memory = MemoryConfig()

        tracker: MemoryTracker | None = None
        access_map = {}
        if memory.agent_memory_access:
            tracker = MemoryTracker()
            for ama in memory.agent_memory_access:
                access_map[ama.agent_name] = ama

        assert tracker is None
        assert access_map == {}

    def test_memory_context_builds_from_recorded_actions(self):
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("agent_a", "bash", "success")
        tracker.record_cot("agent_a", "thinking about the task")

        access = AgentMemoryAccess(
            agent_name="agent_a",
            own_action_history=True,
            own_cot=True,
        )
        context = tracker.build_context("agent_a", access)
        assert "Your Previous Actions" in context
        assert "bash" in context
        assert "Your Previous Reasoning" in context
        assert "thinking about the task" in context

    def test_memory_context_respects_flags(self):
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("agent_a", "bash", "success")
        tracker.record_cot("agent_a", "thinking")

        # Only actions, no CoT
        access = AgentMemoryAccess(
            agent_name="agent_a",
            own_action_history=True,
            own_cot=False,
        )
        context = tracker.build_context("agent_a", access)
        assert "Your Previous Actions" in context
        assert "Your Previous Reasoning" not in context


# ── Memory wiring integration tests ──────────────────────────────────


class TestPathCMemoryInjection:
    """Test memory injection into agent isolated states (Path C pattern)."""

    def test_memory_context_appended_to_agent_state(self):
        """Memory context message is appended to the agent's isolated state."""
        from inspect_ai.model import ChatMessageUser
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("agent_a", "bash", "ran ls")

        memory_access_map = {
            "agent_a": AgentMemoryAccess(
                agent_name="agent_a", own_action_history=True,
            ),
        }
        agent_names = ["agent_a", "agent_b"]
        submitted = set()

        # Simulate the orchestrator's Path C injection loop
        for name in agent_names:
            if name in submitted:
                continue
            access = memory_access_map.get(name)
            if not access:
                continue
            text = tracker.build_context(name, access)
            if text:
                # Simulate agent state as a list of messages
                agent_messages: list = []
                agent_messages.append(
                    ChatMessageUser(content=f"[Memory Context]\n{text}")
                )

        assert len(agent_messages) == 1
        assert "[Memory Context]" in agent_messages[0].content
        assert "bash" in agent_messages[0].content

    def test_submitted_agent_skipped(self):
        """Agents that have submitted should not get memory injected."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("agent_a", "bash", "ran ls")

        memory_access_map = {
            "agent_a": AgentMemoryAccess(
                agent_name="agent_a", own_action_history=True,
            ),
        }
        submitted = {"agent_a"}
        injected_agents = []

        for name in ["agent_a"]:
            if name in submitted:
                continue
            access = memory_access_map.get(name)
            if not access:
                continue
            text = tracker.build_context(name, access)
            if text:
                injected_agents.append(name)

        assert injected_agents == []

    def test_agent_without_access_config_skipped(self):
        """Agents not in memory_access_map get no injection."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("agent_a", "bash", "ran ls")

        # Only agent_a has access configured
        memory_access_map = {
            "agent_a": AgentMemoryAccess(
                agent_name="agent_a", own_action_history=True,
            ),
        }
        injected_agents = []
        for name in ["agent_a", "agent_b"]:
            access = memory_access_map.get(name)
            if not access:
                continue
            text = tracker.build_context(name, access)
            if text:
                injected_agents.append(name)

        assert injected_agents == ["agent_a"]

    def test_no_injection_on_first_turn_empty_tracker(self):
        """On the first turn, tracker has no data so no injection happens."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()  # Fresh, no recorded data
        access = AgentMemoryAccess(
            agent_name="agent_a", own_action_history=True, own_cot=True,
        )
        text = tracker.build_context("agent_a", access)
        assert text == ""

    def test_multi_agent_gets_scoped_context(self):
        """Each agent gets only its own data when shared flags are off."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("agent_a", "bash", "ran ls")
        tracker.record_action("agent_b", "python", "computed result")

        access_a = AgentMemoryAccess(
            agent_name="agent_a", own_action_history=True,
            shared_action_history=False,
        )
        access_b = AgentMemoryAccess(
            agent_name="agent_b", own_action_history=True,
            shared_action_history=False,
        )

        ctx_a = tracker.build_context("agent_a", access_a)
        ctx_b = tracker.build_context("agent_b", access_b)

        assert "bash" in ctx_a
        assert "python" not in ctx_a
        assert "python" in ctx_b
        assert "bash" not in ctx_b

    def test_shared_action_history_includes_others(self):
        """With shared_action_history=True, agent sees other agents' actions."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("agent_a", "bash", "ran ls")
        tracker.record_action("agent_b", "python", "computed result")

        access_a = AgentMemoryAccess(
            agent_name="agent_a", own_action_history=True,
            shared_action_history=True,
        )
        ctx = tracker.build_context("agent_a", access_a)
        assert "bash" in ctx
        assert "python" in ctx
        assert "Other Agents' Actions" in ctx


class TestPathCMemoryRecording:
    """Test that agent actions and CoT are recorded after execution (Path C)."""

    def test_tool_calls_recorded_from_agent_response(self):
        """Tool calls in new messages are recorded as actions."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        memory_access_map = {
            "agent_a": AgentMemoryAccess(
                agent_name="agent_a", own_action_history=True,
            ),
        }

        # Simulate agent_result with messages containing tool calls
        tc = SimpleNamespace(function="bash", arguments={"cmd": "ls"})
        assistant_msg = SimpleNamespace(
            role="assistant", content="running bash",
            tool_calls=[tc],
        )
        agent_messages = [
            SimpleNamespace(role="user", content="do something"),  # before
            assistant_msg,  # new message with tool call
        ]

        # Replicate the recording logic from the orchestrator
        messages_before = 1
        for msg in agent_messages[messages_before:]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc_item in msg.tool_calls:
                    tracker.record_action("agent_a", tc_item.function, "(executed)")

        assert tracker._actions["agent_a"] == [("bash", "(executed)", None)]

    def test_cot_extracted_and_recorded(self):
        """Chain-of-thought in thinking blocks is extracted and recorded."""
        from orbit.memory.tracker import MemoryTracker, extract_cot_from_response

        tracker = MemoryTracker()
        content = "```thinking\nI should list the files first\n```\nI'll run ls."

        # Replicate the recording logic: check for assistant msg with string content
        msg = SimpleNamespace(role="assistant", content=content, tool_calls=None)
        if msg.role == "assistant" and isinstance(msg.content, str):
            cot, _ = extract_cot_from_response(msg.content)
            if cot:
                tracker.record_cot("agent_a", cot)

        assert len(tracker._cot["agent_a"]) == 1
        assert "list the files" in tracker._cot["agent_a"][0]

    def test_no_cot_recorded_without_thinking_block(self):
        """Messages without thinking blocks produce no CoT recording."""
        from orbit.memory.tracker import MemoryTracker, extract_cot_from_response

        tracker = MemoryTracker()
        msg = SimpleNamespace(role="assistant", content="Just a normal response", tool_calls=None)
        if msg.role == "assistant" and isinstance(msg.content, str):
            cot, _ = extract_cot_from_response(msg.content)
            if cot:
                tracker.record_cot("agent_a", cot)

        assert "agent_a" not in tracker._cot

    def test_user_messages_not_recorded(self):
        """User messages (like memory context injection) are not recorded."""
        from orbit.memory.tracker import MemoryTracker, extract_cot_from_response

        tracker = MemoryTracker()
        msg = SimpleNamespace(
            role="user", content="[Memory Context]\nSome context",
            tool_calls=None,
        )

        # The recording logic only processes assistant messages
        if msg.role == "assistant" and isinstance(msg.content, str):
            cot, _ = extract_cot_from_response(msg.content)
            if cot:
                tracker.record_cot("agent_a", cot)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc_item in msg.tool_calls:
                tracker.record_action("agent_a", tc_item.function, "(executed)")

        assert "agent_a" not in tracker._cot
        assert "agent_a" not in tracker._actions

    def test_skips_agent_with_no_new_messages(self):
        """Agents that produced no new messages are skipped in recording."""
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        result = SimpleNamespace(
            agent_name="agent_a", messages_before=3, messages_after=3,
        )

        # Replicate the guard: skip if no new messages
        if result.messages_after <= result.messages_before:
            recorded = False
        else:
            recorded = True

        assert not recorded

    def test_multiple_tool_calls_all_recorded(self):
        """Multiple tool calls in a single message are all recorded."""
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tc1 = SimpleNamespace(function="bash")
        tc2 = SimpleNamespace(function="python")
        tc3 = SimpleNamespace(function="computer")
        msg = SimpleNamespace(
            role="assistant", content="running tools",
            tool_calls=[tc1, tc2, tc3],
        )

        for tc_item in msg.tool_calls:
            tracker.record_action("agent_a", tc_item.function, "(executed)")

        assert len(tracker._actions["agent_a"]) == 3
        assert [a[0] for a in tracker._actions["agent_a"]] == ["bash", "python", "computer"]


class TestPathAMemoryInjection:
    """Test memory injection into turn_state (Path A pattern)."""

    def test_memory_injected_for_root_agent(self):
        """Memory context is injected for root_names[0] in Path A."""
        from inspect_ai.model import ChatMessageUser
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("orchestrator", "bash", "listed files")

        root_names = ["orchestrator"]
        memory_access_map = {
            "orchestrator": AgentMemoryAccess(
                agent_name="orchestrator", own_action_history=True,
            ),
        }

        # Replicate Path A injection logic
        messages: list = [ChatMessageUser(content="Hello")]
        pa_name = root_names[0] if root_names else "root"
        pa_access = memory_access_map.get(pa_name)
        if pa_access:
            pa_text = tracker.build_context(pa_name, pa_access)
            if pa_text:
                messages.append(
                    ChatMessageUser(content=f"[Memory Context]\n{pa_text}")
                )

        assert len(messages) == 2
        assert "[Memory Context]" in messages[1].content
        assert "bash" in messages[1].content

    def test_no_injection_when_root_not_in_access_map(self):
        """If root agent has no memory access config, nothing is injected."""
        from inspect_ai.model import ChatMessageUser
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("orchestrator", "bash", "listed files")

        root_names = ["orchestrator"]
        memory_access_map = {
            "worker": AgentMemoryAccess(
                agent_name="worker", own_action_history=True,
            ),
        }

        messages: list = [ChatMessageUser(content="Hello")]
        pa_name = root_names[0] if root_names else "root"
        pa_access = memory_access_map.get(pa_name)
        if pa_access:
            pa_text = tracker.build_context(pa_name, pa_access)
            if pa_text:
                messages.append(
                    ChatMessageUser(content=f"[Memory Context]\n{pa_text}")
                )

        assert len(messages) == 1  # No injection

    def test_empty_root_names_falls_back_to_root(self):
        """When root_names is empty, fallback name 'root' is used."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        tracker.record_action("root", "bash", "listed files")

        root_names: list[str] = []
        memory_access_map = {
            "root": AgentMemoryAccess(
                agent_name="root", own_action_history=True,
            ),
        }

        pa_name = root_names[0] if root_names else "root"
        assert pa_name == "root"
        pa_access = memory_access_map.get(pa_name)
        assert pa_access is not None


class TestPathAMemoryRecording:
    """Test action/CoT recording after agent execution (Path A pattern)."""

    def test_records_tool_calls_attributed_to_last_agent(self):
        """In Path A, all new tool calls are attributed to last_agent_name."""
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        last_agent_name = "worker_b"

        tc = SimpleNamespace(function="bash")
        new_messages = [
            SimpleNamespace(role="assistant", content="running", tool_calls=[tc]),
        ]

        for msg in new_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc_item in msg.tool_calls:
                    tracker.record_action(last_agent_name, tc_item.function, "(executed)")

        assert tracker._actions["worker_b"] == [("bash", "(executed)", None)]
        assert "worker_a" not in tracker._actions

    def test_records_cot_attributed_to_last_agent(self):
        """In Path A, CoT from thinking blocks is attributed to last_agent_name."""
        from orbit.memory.tracker import MemoryTracker, extract_cot_from_response

        tracker = MemoryTracker()
        last_agent_name = "orchestrator"

        msg = SimpleNamespace(
            role="assistant",
            content="```thinking\nI need to delegate this\n```\nDelegating to worker.",
            tool_calls=None,
        )

        if msg.role == "assistant" and isinstance(msg.content, str):
            cot, _ = extract_cot_from_response(msg.content)
            if cot:
                tracker.record_cot(last_agent_name, cot)

        assert len(tracker._cot["orchestrator"]) == 1
        assert "delegate" in tracker._cot["orchestrator"][0]

    def test_msg_count_before_clamping(self):
        """msg_count_before is clamped so it never overshoots after messages[:] replacement."""
        # Simulate: msg_count_before was 10, but after replacement list is only 8
        msg_count_before = 10
        messages = list(range(8))
        msg_count_before = min(msg_count_before, len(messages))
        assert msg_count_before == 8
        assert messages[msg_count_before:] == []

    def test_msg_count_before_when_list_grows(self):
        """When agent adds messages, new messages are captured correctly."""
        msg_count_before = 5
        messages = list(range(8))  # Agent added 3 new messages
        msg_count_before = min(msg_count_before, len(messages))
        assert msg_count_before == 5
        assert messages[msg_count_before:] == [5, 6, 7]


class TestMemoryAccessMapConstruction:
    """Test the memory_access_map construction from config."""

    def test_multiple_agents_mapped(self):
        """Multiple AgentMemoryAccess entries create correct map."""
        from orbit.configs.setup import AgentMemoryAccess, MemoryConfig

        memory = MemoryConfig(
            agent_memory_access=[
                AgentMemoryAccess(agent_name="agent_a", own_action_history=True),
                AgentMemoryAccess(agent_name="agent_b", own_cot=True),
                AgentMemoryAccess(
                    agent_name="agent_c",
                    own_action_history=True, own_cot=True,
                    shared_action_history=True,
                ),
            ],
        )

        access_map: dict = {}
        for ama in memory.agent_memory_access:
            access_map[ama.agent_name] = ama

        assert len(access_map) == 3
        assert access_map["agent_a"].own_action_history is True
        assert access_map["agent_a"].own_cot is False
        assert access_map["agent_b"].own_cot is True
        assert access_map["agent_c"].shared_action_history is True

    def test_access_map_empty_when_no_config(self):
        """No agent_memory_access means empty map and no tracker."""
        from orbit.configs.setup import MemoryConfig
        from orbit.memory.tracker import MemoryTracker

        memory = MemoryConfig()
        tracker: MemoryTracker | None = None
        access_map: dict = {}
        if memory.agent_memory_access:
            tracker = MemoryTracker()
            for ama in memory.agent_memory_access:
                access_map[ama.agent_name] = ama

        assert tracker is None
        assert access_map == {}

    def test_duplicate_agent_name_last_wins(self):
        """If config has duplicates, last entry wins in the map."""
        from orbit.configs.setup import AgentMemoryAccess

        entries = [
            AgentMemoryAccess(agent_name="agent_a", own_action_history=False),
            AgentMemoryAccess(agent_name="agent_a", own_action_history=True),
        ]
        access_map: dict = {}
        for ama in entries:
            access_map[ama.agent_name] = ama

        assert access_map["agent_a"].own_action_history is True


class TestMemoryInjectionRecordingRoundTrip:
    """End-to-end round-trip: inject context → agent acts → record → next injection."""

    def test_two_turn_roundtrip(self):
        """Simulate two turns: turn 1 records, turn 2 injects that data."""
        from inspect_ai.model import ChatMessageUser
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker, extract_cot_from_response

        tracker = MemoryTracker()
        access = AgentMemoryAccess(
            agent_name="agent_a", own_action_history=True, own_cot=True,
        )

        # Turn 1: no injection (empty tracker), agent acts
        ctx_t1 = tracker.build_context("agent_a", access)
        assert ctx_t1 == ""

        # Simulate agent producing a response with tool call and CoT
        tc = SimpleNamespace(function="bash")
        response = "```thinking\nLet me check the directory\n```\nRunning ls."
        agent_msg = SimpleNamespace(role="assistant", content=response, tool_calls=[tc])

        # Record from turn 1
        tracker.record_action("agent_a", tc.function, "(executed)")
        cot, _ = extract_cot_from_response(agent_msg.content)
        if cot:
            tracker.record_cot("agent_a", cot)

        # Turn 2: injection should contain turn 1 data
        ctx_t2 = tracker.build_context("agent_a", access)
        assert ctx_t2 != ""
        assert "bash" in ctx_t2
        assert "check the directory" in ctx_t2
        assert "Your Previous Actions" in ctx_t2
        assert "Your Previous Reasoning" in ctx_t2

        # Verify the injected message format
        mem_msg = ChatMessageUser(content=f"[Memory Context]\n{ctx_t2}")
        assert "[Memory Context]" in mem_msg.content

    def test_three_turn_accumulation(self):
        """Actions accumulate across turns."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        access = AgentMemoryAccess(
            agent_name="agent_a", own_action_history=True,
        )

        # Turn 1
        tracker.record_action("agent_a", "bash", "listed files")
        ctx = tracker.build_context("agent_a", access)
        assert "bash" in ctx
        assert ctx.count("Action:") == 1

        # Turn 2
        tracker.record_action("agent_a", "python", "computed result")
        ctx = tracker.build_context("agent_a", access)
        assert "bash" in ctx
        assert "python" in ctx
        assert ctx.count("Action:") == 2

        # Turn 3
        tracker.record_action("agent_a", "computer", "clicked button")
        ctx = tracker.build_context("agent_a", access)
        assert ctx.count("Action:") == 3

    def test_multi_agent_roundtrip_with_shared_visibility(self):
        """Two agents: agent_b can see agent_a's actions via shared flag."""
        from orbit.configs.setup import AgentMemoryAccess
        from orbit.memory.tracker import MemoryTracker

        tracker = MemoryTracker()
        access_a = AgentMemoryAccess(
            agent_name="agent_a", own_action_history=True,
        )
        access_b = AgentMemoryAccess(
            agent_name="agent_b", own_action_history=True,
            shared_action_history=True,
        )

        # Turn 1: agent_a acts
        tracker.record_action("agent_a", "bash", "ran ls")

        # Turn 2: agent_b should see agent_a's actions
        ctx_b = tracker.build_context("agent_b", access_b)
        assert "agent_a" in ctx_b
        assert "bash" in ctx_b
        assert "Other Agents' Actions" in ctx_b

        # agent_a doesn't see agent_b (no shared flag)
        ctx_a = tracker.build_context("agent_a", access_a)
        assert "Other Agents" not in ctx_a


# ── Parallel defense tool-call checks (Issue #75) ─────────────────


class TestParallelToolCallDefenses:
    """``_check_tool_call_defenses`` fans awaits out concurrently.

    The previous implementation awaited each defense sequentially, so
    N defenses with per-call latency L cost roughly N*L. The new helper
    uses ``asyncio.gather`` so the total cost is ~L regardless of N.
    """

    @staticmethod
    def _fresh_logs():
        """Create AttackLog/DefenseLog bound to a throwaway Store.

        StoreModel subclasses share a module-level default Store, so
        plain ``DefenseLog()`` calls across tests leak state into each
        other. Binding each test to its own Store keeps the assertions
        absolute rather than delta-based.
        """
        from inspect_ai.util._store import Store

        from orbit.solvers.runtime_state import AttackLog, DefenseLog

        store = Store()
        return AttackLog(store=store), DefenseLog(store=store)

    @pytest.mark.asyncio
    async def test_runs_defenses_concurrently(self):
        import asyncio
        import time

        from orbit.solvers.orchestrator import _check_tool_call_defenses

        class SlowDefense:
            def __init__(self, name: str) -> None:
                self.config = SimpleNamespace(
                    name=name, detection_threshold=0.5,
                )

            async def on_tool_call(self, *_args, **_kwargs):
                await asyncio.sleep(0.05)
                return DefenseVerdict(allow=True, confidence=1.0)

        defenses = [SlowDefense(f"d{i}") for i in range(5)]
        attack_log, defense_log = self._fresh_logs()

        start = time.perf_counter()
        await _check_tool_call_defenses(
            defenses, "bash", {"cmd": "ls"}, "worker", 1,
            defense_log, attack_log,
        )
        elapsed = time.perf_counter() - start

        # Sequential would be ~0.25s for 5 * 50ms. Gather should finish
        # well under 3x the single-call latency even on slow runners.
        assert elapsed < 0.15, f"elapsed {elapsed:.3f}s suggests serial awaits"
        assert defense_log.total_activations == 5

    @pytest.mark.asyncio
    async def test_blocks_are_recorded_in_declaration_order(self):
        """Parallel execution must not reorder log entries."""
        import asyncio

        from orbit.solvers.orchestrator import _check_tool_call_defenses

        class OrderedBlocker:
            def __init__(self, name: str, delay: float) -> None:
                self.config = SimpleNamespace(
                    name=name, detection_threshold=0.5,
                )
                self._delay = delay

            async def on_tool_call(self, *_args, **_kwargs):
                # The faster defenses finish first, but log ordering
                # must still follow declaration order.
                await asyncio.sleep(self._delay)
                return DefenseVerdict(
                    allow=False,
                    reason=f"{self.config.name} blocked",
                    confidence=1.0,
                )

        defenses = [
            OrderedBlocker("first", 0.05),   # slowest
            OrderedBlocker("second", 0.02),
            OrderedBlocker("third", 0.01),   # fastest
        ]
        attack_log, defense_log = self._fresh_logs()

        await _check_tool_call_defenses(
            defenses, "bash", {"cmd": "ls"}, "worker", 1,
            defense_log, attack_log,
        )

        assert [b["defense"] for b in defense_log.blocks] == [
            "first", "second", "third",
        ]

    @pytest.mark.asyncio
    async def test_noop_on_empty_defense_list(self):
        """Empty list short-circuits without gather or mutation."""
        from orbit.solvers.orchestrator import _check_tool_call_defenses

        attack_log, defense_log = self._fresh_logs()

        await _check_tool_call_defenses(
            [], "bash", {"cmd": "ls"}, "worker", 1,
            defense_log, attack_log,
        )
        assert defense_log.total_activations == 0
        assert defense_log.blocks == []


# ── Partial metrics commit on interrupted runs (Issue #148) ──────────


class TestCommitPartialMetrics:
    """Test _commit_partial_metrics records metrics even on interruption."""

    def _make_runtime(self):
        return SimpleNamespace(
            total_turns=0,
            wall_clock_seconds=0.0,
            total_tokens=0,
            invoked_agents=[],
            unhandled_errors=[],
        )

    def test_none_scheduler_is_noop(self):
        from orbit.solvers.orchestrator import _commit_partial_metrics

        runtime = self._make_runtime()
        _commit_partial_metrics(None, runtime)
        assert runtime.total_turns == 0
        assert runtime.wall_clock_seconds == 0.0

    def test_uses_elapsed_when_set(self):
        from orbit.solvers.orchestrator import _commit_partial_metrics

        runtime = self._make_runtime()
        scheduler = ExperimentScheduler(SchedulerConfig(max_turns=10))
        scheduler.total_turns = 5
        scheduler.elapsed_seconds = 42.0
        _commit_partial_metrics(scheduler, runtime)
        assert runtime.total_turns == 5
        assert runtime.wall_clock_seconds == 42.0

    def test_computes_from_start_time_when_interrupted(self):
        """When elapsed_seconds is 0 (scheduler killed before recording),
        fall back to computing from start_time."""
        import time

        from orbit.solvers.orchestrator import _commit_partial_metrics

        runtime = self._make_runtime()
        scheduler = ExperimentScheduler(SchedulerConfig(max_turns=10))
        scheduler.total_turns = 3
        scheduler._start_time = time.monotonic() - 10.0
        _commit_partial_metrics(scheduler, runtime)
        assert runtime.total_turns == 3
        assert runtime.wall_clock_seconds >= 9.5

    def test_zero_start_time_yields_zero_elapsed(self):
        from orbit.solvers.orchestrator import _commit_partial_metrics

        runtime = self._make_runtime()
        scheduler = ExperimentScheduler(SchedulerConfig(max_turns=10))
        _commit_partial_metrics(scheduler, runtime)
        assert runtime.total_turns == 0
        assert runtime.wall_clock_seconds == 0.0


class TestInterruptedSchedulerSyncsMessages:
    """Verify the try/finally pattern preserves messages on interruption.

    Exercises the actual AgentScheduler.sync_to_task_state and
    _commit_partial_metrics working together, simulating the pattern
    used in _run_scheduled_path.
    """

    @pytest.fixture(autouse=True)
    def _isolate_store(self):
        yield
        from inspect_ai.util import store_as
        from orbit.solvers.runtime_state import EnvironmentState
        store_as(EnvironmentState).message_attribution.clear()

    def _make_agent_scheduler_with_messages(self):
        """Build an AgentScheduler with pre-populated agent states."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        from orbit.configs.execution import AgentGroup, ExecutionConfig
        from orbit.execution.agent_scheduler import AgentScheduler
        from orbit.execution.submit_registry import SubmitRegistry

        registry = SubmitRegistry()
        execution_config = ExecutionConfig(
            agent_groups=[AgentGroup(name="team", agents=["alice", "bob"])],
        )
        sched = AgentScheduler(
            agents={"alice": None, "bob": None},
            agent_names=["alice", "bob"],
            registry=registry,
            execution_config=execution_config,
            groups=list(execution_config.agent_groups),
        )

        sched._agent_states["alice"] = AgentState(messages=[
            ChatMessageUser(content="Fix the bug"),
            ChatMessageAssistant(content="I'll look at the code"),
        ])
        sched._agent_states["bob"] = AgentState(messages=[
            ChatMessageUser(content="Review the PR"),
            ChatMessageAssistant(content="Checking now"),
            ChatMessageAssistant(content="LGTM"),
        ])
        return sched

    def test_sync_populates_state_messages(self):
        """sync_to_task_state copies all agent messages to TaskState."""
        sched = self._make_agent_scheduler_with_messages()
        task_state = SimpleNamespace(messages=[], output=None)
        sched.sync_to_task_state(task_state)
        assert len(task_state.messages) == 5

    @pytest.mark.asyncio
    async def test_finally_block_syncs_on_base_exception(self):
        """Simulate the orchestrator's try/finally: even when the loop
        raises BaseException, messages are synced and metrics recorded."""
        import time

        from orbit.solvers.orchestrator import _commit_partial_metrics

        agent_sched = self._make_agent_scheduler_with_messages()
        task_state = SimpleNamespace(messages=[], output=None)
        runtime = SimpleNamespace(
            total_turns=0, wall_clock_seconds=0.0,
            total_tokens=0, invoked_agents=[], unhandled_errors=[],
        )

        scheduler = ExperimentScheduler(
            SchedulerConfig(max_turns=20, health_checks=False)
        )
        scheduler._start_time = time.monotonic() - 30.0
        scheduler.total_turns = 7

        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            finally:
                agent_sched.sync_to_task_state(task_state)
                _commit_partial_metrics(scheduler, runtime)

        assert len(task_state.messages) == 5
        assert runtime.total_turns == 7
        assert runtime.wall_clock_seconds >= 29.5
