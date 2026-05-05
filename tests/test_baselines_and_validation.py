"""Tests for baselines and config validation."""

from __future__ import annotations

import pytest

from orbit.baselines.baselines import apply_baseline
from orbit.configs.attack import AttackConfig
from orbit.configs.baseline import BaselineMode
from orbit.configs.defense import DefenseConfig
from orbit.configs.experiment import ExperimentConfig
from orbit.configs.setup import AgentSpec, SetupConfig, TopologyEdge
from orbit.validation.validators import ConfigValidator


class TestApplyBaseline:
    def test_none_returns_unchanged(self, full_experiment):
        result = apply_baseline(full_experiment, BaselineMode.NONE)
        assert result is full_experiment

    def test_no_defense_strips_defenses(self, full_experiment):
        result = apply_baseline(full_experiment, BaselineMode.NO_DEFENSE)
        assert result.defenses == []
        assert len(result.attacks) > 0
        assert result.baseline_mode == BaselineMode.NO_DEFENSE

    def test_no_attack_strips_attacks(self, full_experiment):
        result = apply_baseline(full_experiment, BaselineMode.NO_ATTACK)
        assert result.attacks == []
        assert len(result.defenses) > 0
        assert result.baseline_mode == BaselineMode.NO_ATTACK

    def test_benign_strips_both(self, full_experiment):
        result = apply_baseline(full_experiment, BaselineMode.BENIGN)
        assert result.attacks == []
        assert result.defenses == []
        assert result.baseline_mode == BaselineMode.BENIGN

    def test_original_unchanged(self, full_experiment):
        apply_baseline(full_experiment, BaselineMode.BENIGN)
        # Original should still have attacks and defenses
        assert len(full_experiment.attacks) > 0
        assert len(full_experiment.defenses) > 0


class TestConfigValidator:
    def test_valid_minimal(self, minimal_experiment):
        errors = ConfigValidator.validate(minimal_experiment)
        assert errors == []

    def test_valid_full(self, full_experiment):
        errors = ConfigValidator.validate(full_experiment)
        assert errors == []

    def test_no_agents(self):
        config = ExperimentConfig(name="test", setup=SetupConfig())
        errors = ConfigValidator.validate(config)
        assert any("at least one agent" in e.lower() for e in errors)

    def test_duplicate_agent_names(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(
                agents=[
                    AgentSpec(name="a", role="worker"),
                    AgentSpec(name="a", role="worker"),
                ]
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("duplicate" in e.lower() for e in errors)

    def test_edge_references_unknown_agent(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(
                agents=[AgentSpec(name="a", role="worker")],
                edges=[TopologyEdge(from_agent="a", to_agent="nonexistent")],
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("unknown" in e.lower() for e in errors)

    def test_self_loop(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(
                agents=[AgentSpec(name="a", role="worker")],
                edges=[TopologyEdge(from_agent="a", to_agent="a")],
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("self-loop" in e.lower() for e in errors)

    def test_attack_references_unknown_agent(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(agents=[AgentSpec(name="a", role="worker")]),
            attacks=[
                AttackConfig(
                    name="atk",
                    attack_type="direct_injection",
                    target_agent="nonexistent",
                )
            ],
        )
        errors = ConfigValidator.validate(config)
        assert any("unknown target_agent" in e.lower() for e in errors)

    def test_defense_references_unknown_agent(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(agents=[AgentSpec(name="a", role="worker")]),
            defenses=[
                DefenseConfig(
                    name="def",
                    defense_type="monitor",
                    target_agents=["nonexistent"],
                )
            ],
        )
        errors = ConfigValidator.validate(config)
        assert any("unknown target_agent" in e.lower() for e in errors)

    def test_baseline_mode_no_defense_with_defenses(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(agents=[AgentSpec(name="a", role="worker")]),
            defenses=[
                DefenseConfig(
                    name="def",
                    defense_type="monitor",
                    target_agents=["a"],
                )
            ],
            baseline_mode=BaselineMode.NO_DEFENSE,
        )
        errors = ConfigValidator.validate(config)
        assert any("no_defense" in e.lower() for e in errors)

    def test_compromised_agent_without_attack(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(
                agents=[AgentSpec(name="a", role="worker", is_compromised=True)]
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("compromised" in e.lower() for e in errors)

    def test_orphan_agents(self):
        config = ExperimentConfig(
            name="test",
            setup=SetupConfig(
                agents=[
                    AgentSpec(name="a", role="worker"),
                    AgentSpec(name="b", role="worker"),
                    AgentSpec(name="c", role="worker"),
                ],
                edges=[TopologyEdge(from_agent="a", to_agent="b")],
            ),
        )
        errors = ConfigValidator.validate(config)
        assert any("orphan" in e.lower() for e in errors)
