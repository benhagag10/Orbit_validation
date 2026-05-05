"""Tests for SWE-Bench coding agent condition presets.

Unit tests for the condition registry, resolver, and preset factories,
plus integration tests that verify each condition works end-to-end through
the config builder pipeline with different issue counts.
"""

from __future__ import annotations

import json

import pytest

from orbit.configs.experiment import ExperimentConfig
from orbit.configs.setup import SetupConfig
from orbit.scenarios.coding.swe_bench.condition_presets import (
    CONDITION_REGISTRY,
    get_condition_setup,
    list_conditions,
    resolve_condition,
)
from orbit.scenarios.coding.swe_bench.configs import (
    IssueSpec,
    MultiIssueGroup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    instance_id: str,
    repo: str = "django/django",
    version: str = "4.2",
    base_commit: str = "abc123",
    created_at: str = "2023-01-01T00:00:00Z",
) -> IssueSpec:
    return IssueSpec(
        instance_id=instance_id,
        repo=repo,
        base_commit=base_commit,
        version=version,
        problem_statement=f"Fix {instance_id}",
        patch=f"diff --git a/file.py\n+fix",
        test_patch=f"diff --git a/tests/test.py\n+assert",
        fail_to_pass=["tests.test_foo.TestBar.test_baz"],
        pass_to_pass=["tests.test_foo.TestBar.test_existing"],
        environment_setup_commit="def456",
        hints_text="",
        created_at=created_at,
    )


def _make_group(
    n_issues: int = 2,
    repo: str = "django/django",
    version: str = "4.2",
) -> MultiIssueGroup:
    issues = [
        _make_issue(
            f"django__django-{i:03d}",
            repo=repo,
            version=version,
            created_at=f"2023-01-{i+1:02d}T00:00:00Z",
        )
        for i in range(n_issues)
    ]
    return MultiIssueGroup(
        group_id=f"django__django-v4-2-g0",
        repo=repo,
        version=version,
        common_base_commit="def456",
        docker_image="swebench/sweb.eval.x86_64.django__django:latest",
        issues=issues,
        pool_id="django__django-v4-2",
    )


# Expected agent counts per condition (template, before replication)
_CONDITION_TEMPLATE_AGENTS: dict[str, int] = {
    "single_agent": 1,
    "star_batch": 2,
    "star_batch_relaxed": 2,
    "star_specialist": 5,
    "star_specialist_relaxed": 5,
    "star_step": 2,
    "mesh_round_robin": 4,
    "mesh_delegation": 4,
    "memory_none": 5,
    "memory_own_actions": 5,
    "memory_own_reasoning": 5,
    "memory_shared_actions": 5,
    "memory_full": 5,
}

# Expected edge counts per condition (template, before replication)
_CONDITION_TEMPLATE_EDGES: dict[str, int] = {
    "single_agent": 0,
    "star_batch": 1,
    "star_batch_relaxed": 1,
    "star_specialist": 4,
    "star_specialist_relaxed": 4,
    "star_step": 1,
    "mesh_round_robin": 12,
    "mesh_delegation": 12,
    "memory_none": 4,
    "memory_own_actions": 4,
    "memory_own_reasoning": 4,
    "memory_shared_actions": 4,
    "memory_full": 4,
}


class TestConditionRegistry:
    def test_all_conditions_present(self):
        expected = {
            "single_agent", "star_batch", "star_batch_relaxed",
            "star_specialist", "star_specialist_relaxed", "star_step",
            "mesh_round_robin", "mesh_delegation",
            "memory_none", "memory_own_actions", "memory_own_reasoning",
            "memory_shared_actions", "memory_full",
        }
        assert expected == set(CONDITION_REGISTRY.keys())

    def test_list_conditions(self):
        conditions = list_conditions()
        assert len(conditions) == 13
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
        assert setup.agents[0].name == "coding_agent"


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

    def test_specialist_names(self):
        """Specialists should be coding-domain specific."""
        setup = get_condition_setup("star_specialist")
        names = {a.name for a in setup.agents if a.role != "orchestrator"}
        assert names == {
            "implementation_specialist",
            "debugging_specialist",
            "testing_specialist",
            "review_specialist",
        }


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
    def test_agents_have_coding_tools(self, condition):
        """Non-orchestrator agents should have coding tools."""
        setup = get_condition_setup(condition)
        expected_tools = {"bash", "python", "text_editor"}
        for agent in setup.agents:
            if agent.role != "orchestrator":
                assert set(agent.tools) == expected_tools, (
                    f"Agent {agent.name} in condition {condition} "
                    f"has tools {agent.tools}, expected {expected_tools}"
                )


class TestResolveCondition:
    def test_single_agent(self):
        assert resolve_condition(agents="single") == "single_agent"

    def test_star_batch(self):
        assert resolve_condition(agents="batch") == "star_batch"

    def test_star_batch_relaxed(self):
        assert resolve_condition(agents="batch", instructions="relaxed") == "star_batch_relaxed"

    def test_star_specialist(self):
        assert resolve_condition(agents="specialist") == "star_specialist"

    def test_mesh_round_robin(self):
        assert resolve_condition(agents="specialist", topology="round_robin") == "mesh_round_robin"

    def test_mesh_delegation(self):
        assert resolve_condition(agents="specialist", topology="delegation") == "mesh_delegation"

    def test_memory_full(self):
        assert resolve_condition(agents="specialist", memory="full") == "memory_full"

    def test_invalid_agents_raises(self):
        with pytest.raises(ValueError, match="Invalid --agents"):
            resolve_condition(agents="unknown")

    def test_invalid_topology_raises(self):
        with pytest.raises(ValueError, match="Invalid --topology"):
            resolve_condition(topology="invalid")

    def test_invalid_memory_raises(self):
        with pytest.raises(ValueError, match="Invalid --memory"):
            resolve_condition(memory="invalid")

    def test_unsupported_combination_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_condition(agents="batch", topology="round_robin")


# ===========================================================================
# Integration tests — conditions through the config builder pipeline
# ===========================================================================


class TestConditionReplication:
    """Verify each condition replicates correctly per issue."""

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    @pytest.mark.parametrize("n_issues", [1, 2, 3])
    def test_agent_count_scales_with_issues(self, condition: str, n_issues: int):
        """Each issue gets a full replica of the condition's topology."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(n_issues)
        config = build_experiment_config(group, topology_template=setup)

        template_agents = _CONDITION_TEMPLATE_AGENTS[condition]
        assert len(config.setup.agents) == template_agents * n_issues

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    @pytest.mark.parametrize("n_issues", [1, 2, 3])
    def test_edge_count_scales_with_issues(self, condition: str, n_issues: int):
        """Each issue gets a full replica of the condition's edges."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(n_issues)
        config = build_experiment_config(group, topology_template=setup)

        template_edges = _CONDITION_TEMPLATE_EDGES[condition]
        assert len(config.setup.edges) == template_edges * n_issues

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_agent_names_suffixed(self, condition: str):
        """All replicated agents should have _{idx} suffixes."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)

        for agent in config.setup.agents:
            assert agent.name.endswith("_0") or agent.name.endswith("_1"), (
                f"Agent {agent.name} missing issue index suffix"
            )

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_edges_reference_suffixed_agents(self, condition: str):
        """All edges should reference agent names with _{idx} suffixes."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        if len(setup.edges) == 0:
            pytest.skip("No edges in this condition")

        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)
        agent_names = {a.name for a in config.setup.agents}

        for edge in config.setup.edges:
            assert edge.from_agent in agent_names, (
                f"Edge from_agent {edge.from_agent} not in agents"
            )
            assert edge.to_agent in agent_names, (
                f"Edge to_agent {edge.to_agent} not in agents"
            )

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_no_cross_issue_edges(self, condition: str):
        """Edges should only connect agents within the same issue replica."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        if len(setup.edges) == 0:
            pytest.skip("No edges in this condition")

        group = _make_group(3)
        config = build_experiment_config(group, topology_template=setup)

        for edge in config.setup.edges:
            from_suffix = edge.from_agent.rsplit("_", 1)[-1]
            to_suffix = edge.to_agent.rsplit("_", 1)[-1]
            assert from_suffix == to_suffix, (
                f"Cross-issue edge: {edge.from_agent} → {edge.to_agent}"
            )


class TestConditionAgentGroups:
    """Verify agent groups are correctly created for multi-issue conditions."""

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_multi_issue_creates_groups(self, condition: str):
        """Multi-issue groups should have one AgentGroup per issue."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(3)
        config = build_experiment_config(group, topology_template=setup)

        assert len(config.execution.agent_groups) == 3
        for i, ag in enumerate(config.execution.agent_groups):
            assert ag.name == f"issue_{i}"
            template_agents = _CONDITION_TEMPLATE_AGENTS[condition]
            assert len(ag.agents) == template_agents

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_single_issue_no_groups(self, condition: str):
        """Single-issue groups should have no AgentGroups (Path A)."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(1)
        config = build_experiment_config(group, topology_template=setup)

        assert len(config.execution.agent_groups) == 0

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_group_agents_match_setup_agents(self, condition: str):
        """Every agent in an AgentGroup should exist in setup.agents."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)

        all_agent_names = {a.name for a in config.setup.agents}
        for ag in config.execution.agent_groups:
            for name in ag.agents:
                assert name in all_agent_names, (
                    f"AgentGroup references {name} not in setup.agents"
                )


class TestConditionIssuePromptInjection:
    """Verify the lead agent in each replica gets the issue prompt."""

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_lead_agent_gets_issue_prompt(self, condition: str):
        """The first agent in each replica should have the issue text."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)

        template_agents = _CONDITION_TEMPLATE_AGENTS[condition]
        # First agent in each replica (idx 0 and idx template_agents)
        for issue_idx in range(2):
            lead = config.setup.agents[issue_idx * template_agents]
            issue_id = group.issues[issue_idx].instance_id
            assert f"Fix {issue_id}" in lead.system_prompt, (
                f"Lead agent {lead.name} missing issue prompt for {issue_id}"
            )

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_non_lead_agents_no_issue_prompt(self, condition: str):
        """Non-lead agents should NOT have issue text injected."""
        setup = get_condition_setup(condition)
        if len(setup.agents) <= 1:
            pytest.skip("Single-agent condition, no non-lead agents")

        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)

        template_agents = _CONDITION_TEMPLATE_AGENTS[condition]
        for issue_idx in range(2):
            start = issue_idx * template_agents
            issue_id = group.issues[issue_idx].instance_id
            # Skip first agent (lead)
            for agent in config.setup.agents[start + 1 : start + template_agents]:
                assert f"Fix {issue_id}" not in agent.system_prompt, (
                    f"Non-lead agent {agent.name} has issue prompt for {issue_id}"
                )


class TestConditionConfigRoundtrip:
    """Verify configs built from conditions survive JSON serialization."""

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_config_json_roundtrip(self, condition: str):
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)

        data = json.loads(config.model_dump_json())
        restored = ExperimentConfig(**data)

        assert restored.name == config.name
        assert len(restored.setup.agents) == len(config.setup.agents)
        assert len(restored.setup.edges) == len(config.setup.edges)
        assert len(restored.execution.agent_groups) == len(
            config.execution.agent_groups
        )

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_setup_json_roundtrip(self, condition: str):
        """The SetupConfig itself should survive JSON roundtrip."""
        setup = get_condition_setup(condition)
        data = json.loads(setup.model_dump_json())
        restored = SetupConfig(**data)

        assert restored.properties == setup.properties
        assert len(restored.agents) == len(setup.agents)
        assert len(restored.edges) == len(setup.edges)
        for orig, rest in zip(setup.agents, restored.agents):
            assert orig.name == rest.name
            assert orig.role == rest.role
            assert orig.tools == rest.tools


class TestConditionEndToEnd:
    """Full pipeline: condition → config → sample."""

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_condition_to_sample(self, condition: str):
        """Full pipeline from condition through config builder to Sample."""
        from orbit.dataset.sample_factory import build_sample
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)
        sample = build_sample(config, sample_id=config.name)

        assert sample.id == config.name
        assert sample.sandbox is not None
        meta = sample.metadata["experiment"]
        assert "swe_bench_issues" in meta["metadata"]
        assert len(meta["setup"]["agents"]) == _CONDITION_TEMPLATE_AGENTS[condition] * 2

    @pytest.mark.parametrize(
        "condition,n_issues",
        [
            ("single_agent", 1),
            ("star_specialist", 2),
            ("star_batch", 3),
            ("mesh_round_robin", 2),
            ("memory_full", 4),
        ],
    )
    def test_condition_with_varying_issue_counts(self, condition: str, n_issues: int):
        """Verify representative conditions work with various issue counts."""
        from orbit.dataset.sample_factory import build_sample
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup(condition)
        group = _make_group(n_issues)
        config = build_experiment_config(group, topology_template=setup)
        sample = build_sample(config, sample_id=config.name)

        expected_agents = _CONDITION_TEMPLATE_AGENTS[condition] * n_issues
        assert len(config.setup.agents) == expected_agents
        assert sample.id == config.name

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_condition_with_attack_defense_presets(self, condition: str):
        """Condition presets work alongside attack/defense presets."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )
        from orbit.scenarios.coding.swe_bench.presets import (
            full_defense_preset,
            self_replication_attack_preset,
        )

        setup = get_condition_setup(condition)
        group = _make_group(2)
        attacks = self_replication_attack_preset()
        defenses = full_defense_preset()

        config = build_experiment_config(
            group, topology_template=setup, attacks=attacks, defenses=defenses,
        )

        assert len(config.attacks) == 1
        assert len(config.defenses) == 3
        assert len(config.setup.agents) == _CONDITION_TEMPLATE_AGENTS[condition] * 2

    def test_condition_metadata_in_agent_issue_map(self):
        """Agent-to-issue map should contain all replicated agents."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )

        setup = get_condition_setup("star_specialist")
        group = _make_group(2)
        config = build_experiment_config(group, topology_template=setup)

        agent_map = config.metadata["swe_bench_agent_issue_map"]
        # 5 agents per issue * 2 issues = 10 entries
        assert len(agent_map) == 10
        # Verify all agents in setup are in the map
        for agent in config.setup.agents:
            assert agent.name in agent_map


class TestConditionSchedulerTurns:
    """Verify max_turns is computed correctly for different conditions."""

    @pytest.mark.parametrize("condition", ["single_agent", "star_specialist", "mesh_round_robin"])
    def test_round_robin_turns_scale_with_issues(self, condition: str):
        """Round-robin mode: max_turns = per_agent * n_issues."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )
        from orbit.scenarios.coding.swe_bench.configs import SWEBenchScenarioConfig

        setup = get_condition_setup(condition)
        group = _make_group(3)
        sc = SWEBenchScenarioConfig(max_turns_per_agent=10)
        config = build_experiment_config(group, topology_template=setup, scenario_config=sc)

        # 3 issues × 10 turns = 30
        assert config.scheduler.max_turns == 30

    @pytest.mark.parametrize("condition", ["single_agent", "star_specialist", "mesh_round_robin"])
    def test_superstep_turns_no_scaling(self, condition: str):
        """Superstep mode: max_turns = per_agent (all run each turn)."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )
        from orbit.scenarios.coding.swe_bench.configs import SWEBenchScenarioConfig

        setup = get_condition_setup(condition)
        group = _make_group(3)
        sc = SWEBenchScenarioConfig(max_turns_per_agent=10, execution_mode="superstep")
        config = build_experiment_config(group, topology_template=setup, scenario_config=sc)

        assert config.scheduler.max_turns == 10

    def test_single_issue_no_scaling(self):
        """Single-issue groups use max_turns_per_agent directly."""
        from orbit.scenarios.coding.swe_bench.config_builder import (
            build_experiment_config,
        )
        from orbit.scenarios.coding.swe_bench.configs import SWEBenchScenarioConfig

        setup = get_condition_setup("star_specialist")
        group = _make_group(1)
        sc = SWEBenchScenarioConfig(max_turns_per_agent=15)
        config = build_experiment_config(group, topology_template=setup, scenario_config=sc)

        assert config.scheduler.max_turns == 15


# ---------------------------------------------------------------------------
# Regression guard: orchestrator prompts must use tool-call delegation
# ---------------------------------------------------------------------------

_TEXT_DELEGATION_PATTERNS = ["DISPATCH:", "DELEGATE:"]


class TestNoTextDelegationInOrchestratorPrompts:
    """Orchestrator prompts must not use DISPATCH/DELEGATE text format.

    mas_orchestrator uses react() + as_tool() which requires the model to
    make tool calls to delegate. Text-based delegation (DISPATCH/DELEGATE)
    is silently ignored by react(). See notes/fix_plan_group_a_detailed.md.
    """

    @pytest.mark.parametrize("condition", sorted(CONDITION_REGISTRY.keys()))
    def test_no_text_delegation(self, condition: str) -> None:
        setup = get_condition_setup(condition)
        for spec in setup.agents:
            if spec.role != "orchestrator":
                continue
            for pattern in _TEXT_DELEGATION_PATTERNS:
                assert pattern not in (spec.system_prompt or ""), (
                    f"Found '{pattern}' in orchestrator prompt for "
                    f"SWE-Bench condition '{condition}', agent '{spec.name}'. "
                    f"Orchestrator prompts must instruct tool-call delegation."
                )
