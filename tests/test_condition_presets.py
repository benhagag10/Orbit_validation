"""Tests for BrowserART condition presets."""

from __future__ import annotations

import re

import pytest

from orbit.scenarios.browser.browserart.condition_presets import (
    CONDITION_REGISTRY,
    get_condition,
    get_condition_setup,
    list_conditions,
)


class TestConditionRegistry:
    def test_all_conditions_present(self):
        expected = {
            "single_agent", "star_batch", "star_batch_relaxed",
            "star_2_specialist", "star_3_specialist",
            "star_specialist", "star_specialist_relaxed", "star_step",
            "mesh_round_robin", "mesh_delegation",
            "mesh_delegation_cot", "mesh_delegation_full",
            "memory_none", "memory_own_actions", "memory_own_reasoning",
            "memory_shared_actions", "memory_full",
        }
        assert expected == set(CONDITION_REGISTRY.keys())

    def test_list_conditions(self):
        conditions = list_conditions()
        assert len(conditions) == 17
        assert conditions == sorted(conditions)

    def test_unknown_condition_raises(self):
        with pytest.raises(ValueError, match="Unknown condition"):
            get_condition_setup("nonexistent")


class TestAllConditionsProduceValidSetup:
    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_produces_valid_setup(self, condition):
        setup = get_condition_setup(condition)
        assert len(setup.agents) >= 1
        assert "topology_type" in setup.properties


class TestSingleAgentConditions:
    def test_single_agent(self):
        setup = get_condition_setup("single_agent")
        assert len(setup.agents) == 1
        assert len(setup.edges) == 0
        assert setup.properties["topology_type"] == "single"


class TestStarConditions:
    @pytest.mark.parametrize("condition", ["star_batch", "star_batch_relaxed"])
    def test_batch_conditions(self, condition):
        setup = get_condition_setup(condition)
        assert setup.properties["topology_type"] == "star"
        assert setup.properties["execution_style"] == "batch"
        assert len(setup.agents) == 2  # orchestrator + 1 executor
        assert len(setup.edges) == 1

    @pytest.mark.parametrize("condition", ["star_specialist", "star_specialist_relaxed"])
    def test_specialist_conditions(self, condition):
        setup = get_condition_setup(condition)
        assert setup.properties["topology_type"] == "star"
        assert setup.properties["execution_style"] == "specialist_dispatch"
        assert len(setup.agents) == 5  # orchestrator + 4 specialists
        assert len(setup.edges) == 4

    def test_step_condition(self):
        setup = get_condition_setup("star_step")
        assert setup.properties["topology_type"] == "star"
        assert setup.properties["execution_style"] == "step"
        assert len(setup.agents) == 2
        assert len(setup.edges) == 1


class TestMeshConditions:
    def test_round_robin(self):
        setup = get_condition_setup("mesh_round_robin")
        assert setup.properties["topology_type"] == "mesh_round_robin"
        assert setup.properties["execution_style"] == "round_robin"
        assert len(setup.agents) == 4
        assert len(setup.edges) == 12  # complete graph: 4*3
        assert "turn_order" in setup.properties

    def test_delegation(self):
        setup = get_condition_setup("mesh_delegation")
        assert setup.properties["topology_type"] == "mesh_delegation"
        assert setup.properties["execution_style"] == "delegation"
        assert len(setup.agents) == 4
        assert len(setup.edges) == 12


class TestMemoryConditions:
    def test_memory_none_minimal(self):
        setup = get_condition_setup("memory_none")
        assert setup.properties["memory_level"] == "memory_none"
        assert setup.memory.shared is True
        # memory_none: all flags False (goal_visible is True by default)
        for access in setup.memory.agent_memory_access:
            assert access.goal_visible is True
            assert access.own_action_history is False
            assert access.own_cot is False
            assert access.shared_action_history is False

    def test_increasing_access(self):
        """Each memory level should have at least as many flags as the previous."""
        levels = [
            "memory_none", "memory_own_actions", "memory_own_reasoning",
            "memory_shared_actions", "memory_full",
        ]

        def _count_true_flags(condition: str) -> int:
            setup = get_condition_setup(condition)
            access = setup.memory.agent_memory_access[0]
            count = 0
            for field in [
                "own_action_history", "own_cot",
                "shared_action_history", "shared_cot",
                "nl_instructions_accumulated",
            ]:
                if getattr(access, field):
                    count += 1
            return count

        counts = [_count_true_flags(level) for level in levels]
        # Each level should have >= previous level
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], (
                f"{levels[i]} has fewer flags ({counts[i]}) than {levels[i-1]} ({counts[i-1]})"
            )

    def test_memory_full_access(self):
        setup = get_condition_setup("memory_full")
        for access in setup.memory.agent_memory_access:
            assert access.own_action_history is True
            assert access.own_cot is True
            assert access.shared_action_history is True
            assert access.shared_cot is True
            assert access.nl_instructions_accumulated is True

    @pytest.mark.parametrize("condition", [
        "memory_none", "memory_own_actions", "memory_own_reasoning",
        "memory_shared_actions", "memory_full",
    ])
    def test_memory_star_topology(self, condition):
        setup = get_condition_setup(condition)
        assert setup.properties["topology_type"] == "star"
        assert len(setup.agents) == 5
        assert len(setup.edges) == 4


class TestConditionProperties:
    @pytest.mark.parametrize(
        "condition,expected_type",
        [
            ("single_agent", "single"),
            ("star_batch", "star"),
            ("star_specialist", "star"),
            ("star_step", "star"),
            ("mesh_round_robin", "mesh_round_robin"),
            ("mesh_delegation", "mesh_delegation"),
            ("memory_none", "star"),
        ],
    )
    def test_topology_types(self, condition, expected_type):
        setup = get_condition_setup(condition)
        assert setup.properties["topology_type"] == expected_type

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_agents_have_browser_tools(self, condition):
        """Non-orchestrator agents should use the browserart-service tools.

        PR 3 retires the legacy ``web_browser`` tool — every non-
        orchestrator agent must advertise the new ``browser_*`` scenario
        tools that dispatch to the persistent service.
        """
        setup = get_condition_setup(condition)
        for agent in setup.agents:
            if agent.role == "orchestrator":
                continue
            assert "web_browser" not in agent.tools, (
                f"{condition}: {agent.name} still lists legacy web_browser"
            )
            assert any(t.startswith("browser_") for t in agent.tools), (
                f"{condition}: {agent.name} missing browser_* tools"
            )


class TestPathCWiring:
    """Mesh conditions must supply an ExecutionConfig that routes to Path C."""

    MESH_CONDITIONS = [
        "mesh_round_robin",
        "mesh_delegation",
        "mesh_delegation_cot",
        "mesh_delegation_full",
    ]

    NON_MESH_CONDITIONS = [
        name for name in sorted(CONDITION_REGISTRY.keys())
        if not name.startswith("mesh")
    ]

    @pytest.mark.parametrize("condition", MESH_CONDITIONS)
    def test_mesh_has_path_c_execution(self, condition):
        cs = get_condition(condition)
        assert cs.execution is not None, f"{condition}: missing ExecutionConfig"
        assert cs.execution.scheduling_mode == "round_robin"
        assert cs.execution.observation.mode == "summary"
        # One group covering every peer in declaration order so the
        # AgentScheduler's round-robin cycle is deterministic.
        assert len(cs.execution.agent_groups) == 1
        group = cs.execution.agent_groups[0]
        setup_names = [a.name for a in cs.setup.agents]
        assert group.agents == setup_names, (
            f"{condition}: agent group order does not match setup order"
        )
        # Goal must be empty at preset time — config_builder patches in
        # the behavior text per-sample. If a preset ever pre-populates
        # goal, the patch in build_experiment_config will skip it.
        assert group.goal == "", (
            f"{condition}: agent group goal should be empty until the "
            f"config builder patches in the behavior text"
        )

    def test_mesh_peers_are_four_peer_specialists(self):
        """Every mesh condition has the 4 canonical peers with specialty."""
        expected_names = {
            "click_specialist", "fill_specialist",
            "scroll_specialist", "navigate_specialist",
        }
        for condition in self.MESH_CONDITIONS:
            cs = get_condition(condition)
            assert {a.name for a in cs.setup.agents} == expected_names
            for agent in cs.setup.agents:
                assert agent.role == "peer", condition
                assert agent.specialty, f"{condition}: {agent.name} no specialty"

    @pytest.mark.parametrize("condition", NON_MESH_CONDITIONS)
    def test_non_mesh_uses_path_a(self, condition):
        cs = get_condition(condition)
        assert cs.execution is None, (
            f"{condition}: non-mesh condition must not populate agent_groups"
        )


class TestExperimentalIsolation:
    """Verify conditions only vary the intended dimension."""

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_orchestrator_prompt_matches_actual_agents(self, condition):
        """Orchestrator prompt must only reference agents that actually exist."""
        setup = get_condition_setup(condition)
        orchestrator = next(
            (a for a in setup.agents if a.role == "orchestrator"), None
        )
        if orchestrator is None:
            return  # no orchestrator (single_agent, mesh conditions)

        worker_names = {a.name for a in setup.agents if a.role != "orchestrator"}
        prompt = orchestrator.system_prompt
        # Extract agent names referenced in the prompt (word-boundary match)
        all_possible = [
            "click_fill_specialist", "scroll_navigate_specialist",
            "click_specialist", "fill_specialist",
            "scroll_specialist", "navigate_specialist",
        ]
        # Check longer names first to avoid substring false positives
        for name in sorted(all_possible, key=len, reverse=True):
            # Use word boundary to avoid matching "click_fill_specialist"
            # when checking for "fill_specialist"
            if re.search(rf"(?<![a-z_]){re.escape(name)}(?![a-z_])", prompt):
                assert name in worker_names, (
                    f"Condition {condition!r}: orchestrator prompt mentions "
                    f"{name!r} but it is not in agents {sorted(worker_names)}"
                )

    def test_memory_conditions_match_star_specialist_except_memory(self):
        """Memory conditions must be identical to star_specialist except MemoryConfig."""
        baseline = get_condition_setup("star_specialist")
        for level in [
            "memory_none", "memory_own_actions", "memory_own_reasoning",
            "memory_shared_actions", "memory_full",
        ]:
            setup = get_condition_setup(level)
            # Same agents (names, roles, prompts)
            assert len(setup.agents) == len(baseline.agents)
            for a, b in zip(
                sorted(setup.agents, key=lambda x: x.name),
                sorted(baseline.agents, key=lambda x: x.name),
            ):
                assert a.name == b.name, f"{level}: agent name mismatch"
                assert a.role == b.role, f"{level}: role mismatch for {a.name}"
                assert a.system_prompt == b.system_prompt, (
                    f"{level}: prompt mismatch for {a.name}"
                )
                assert a.tools == b.tools, f"{level}: tools mismatch for {a.name}"
            # Same edges
            assert len(setup.edges) == len(baseline.edges)

    def test_mesh_memory_conditions_match_mesh_delegation_except_memory(self):
        """Mesh memory conditions must match mesh_delegation except MemoryConfig."""
        baseline = get_condition_setup("mesh_delegation")
        for level in ["mesh_delegation_cot", "mesh_delegation_full"]:
            setup = get_condition_setup(level)
            assert len(setup.agents) == len(baseline.agents)
            for a, b in zip(
                sorted(setup.agents, key=lambda x: x.name),
                sorted(baseline.agents, key=lambda x: x.name),
            ):
                assert a.name == b.name, f"{level}: agent name mismatch"
                assert a.role == b.role, f"{level}: role mismatch for {a.name}"
                assert a.system_prompt == b.system_prompt, (
                    f"{level}: prompt mismatch for {a.name}"
                )
                assert a.tools == b.tools, f"{level}: tools mismatch for {a.name}"
            assert len(setup.edges) == len(baseline.edges)
