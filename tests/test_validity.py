"""Validity tests — are constraints enforced?

Tests that bad inputs are rejected and invariants hold: frozen config
immutability, validator error messages, Pydantic Literal rejection, and
agent group constraints.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orbit.configs.attack import AttackConfig
from orbit.configs.baseline import BaselineMode
from orbit.configs.defense import DefenseConfig
from orbit.configs.execution import (
    AgentGroup,
    ExecutionConfig,
)
from orbit.configs.experiment import ExperimentConfig
from orbit.configs.setup import AgentSpec, SetupConfig, TopologyEdge
from orbit.validation.validators import ConfigValidator


# ── Frozen configs reject mutation ────────────────────────────────────────


def test_experiment_config_is_frozen():
    """Assigning to field raises."""
    config = ExperimentConfig(name="test")
    with pytest.raises(ValidationError):
        config.name = "changed"


def test_attack_config_is_frozen():
    """Assigning to field raises."""
    config = AttackConfig(name="test", attack_type="direct_injection")
    with pytest.raises(ValidationError):
        config.name = "changed"


def test_execution_config_is_frozen():
    """Assigning to field raises."""
    config = ExecutionConfig()
    with pytest.raises(ValidationError):
        config.scheduling_mode = "superstep"


# ── Validator catches bad configs ─────────────────────────────────────────


def test_validator_catches_no_agents():
    """Empty agents list → error."""
    config = ExperimentConfig(name="test", setup=SetupConfig())
    errors = ConfigValidator.validate(config)
    assert any("at least one agent" in e.lower() for e in errors)


def test_validator_catches_duplicate_agent_names():
    """Two agents named "a" → error."""
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


def test_validator_catches_unknown_edge_agent():
    """Edge referencing nonexistent agent → error."""
    config = ExperimentConfig(
        name="test",
        setup=SetupConfig(
            agents=[AgentSpec(name="a", role="worker")],
            edges=[TopologyEdge(from_agent="a", to_agent="nonexistent")],
        ),
    )
    errors = ConfigValidator.validate(config)
    assert any("unknown" in e.lower() for e in errors)


def test_validator_catches_self_loop():
    """Edge from "a" to "a" → error."""
    config = ExperimentConfig(
        name="test",
        setup=SetupConfig(
            agents=[AgentSpec(name="a", role="worker")],
            edges=[TopologyEdge(from_agent="a", to_agent="a")],
        ),
    )
    errors = ConfigValidator.validate(config)
    assert any("self-loop" in e.lower() for e in errors)


def test_validator_catches_unknown_attack_target():
    """Attack targeting nonexistent agent → error."""
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


def test_validator_catches_unknown_defense_target():
    """Defense targeting nonexistent agent → error."""
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


def test_validator_catches_orphan_agents():
    """Multi-agent with disconnected agent → error."""
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


def test_validator_catches_baseline_inconsistency():
    """BENIGN mode with attacks still present → error."""
    config = ExperimentConfig(
        name="test",
        setup=SetupConfig(agents=[AgentSpec(name="a", role="worker")]),
        attacks=[
            AttackConfig(
                name="atk",
                attack_type="direct_injection",
                target_agent="a",
            )
        ],
        baseline_mode=BaselineMode.BENIGN,
    )
    errors = ConfigValidator.validate(config)
    assert any("benign" in e.lower() for e in errors)


# ── Agent group validation ────────────────────────────────────────────────


def test_validator_catches_duplicate_group_names():
    """Two groups named "g0" → error."""
    config = ExperimentConfig(
        name="test",
        setup=SetupConfig(
            agents=[
                AgentSpec(name="a", role="worker"),
                AgentSpec(name="b", role="worker"),
            ],
        ),
        execution=ExecutionConfig(
            agent_groups=[
                AgentGroup(name="g0", agents=["a"]),
                AgentGroup(name="g0", agents=["b"]),
            ]
        ),
    )
    errors = ConfigValidator.validate(config)
    assert any("duplicate" in e.lower() for e in errors)


def test_validator_catches_empty_group():
    """Group with no agents → error."""
    config = ExperimentConfig(
        name="test",
        setup=SetupConfig(agents=[AgentSpec(name="a", role="worker")]),
        execution=ExecutionConfig(
            agent_groups=[AgentGroup(name="g0", agents=[])]
        ),
    )
    errors = ConfigValidator.validate(config)
    assert any("no agents" in e.lower() for e in errors)


def test_validator_catches_agent_in_multiple_groups():
    """Agent "a" in two different groups → error."""
    config = ExperimentConfig(
        name="test",
        setup=SetupConfig(agents=[AgentSpec(name="a", role="worker")]),
        execution=ExecutionConfig(
            agent_groups=[
                AgentGroup(name="g0", agents=["a"]),
                AgentGroup(name="g1", agents=["a"]),
            ]
        ),
    )
    errors = ConfigValidator.validate(config)
    assert any("groups" in e.lower() for e in errors)


# ── Type-level rejection ──────────────────────────────────────────────────


def test_unknown_attack_type_warns(caplog):
    """Unknown attack_type logs a warning but does not raise."""
    import logging
    with caplog.at_level(logging.WARNING):
        config = AttackConfig(name="test", attack_type="nonexistent_type")
    assert config.attack_type == "nonexistent_type"
    assert "nonexistent_type" in caplog.text


def test_unknown_defense_type_warns(caplog):
    """Unknown defense_type logs a warning but does not raise."""
    import logging
    with caplog.at_level(logging.WARNING):
        config = DefenseConfig(name="test", defense_type="nonexistent_type")
    assert config.defense_type == "nonexistent_type"
    assert "nonexistent_type" in caplog.text


def test_unknown_mechanism_rejected_by_pydantic():
    """Invalid edge mechanism → ValidationError."""
    with pytest.raises(ValidationError):
        TopologyEdge(from_agent="a", to_agent="b", mechanism="teleport")


# ── Valid configs pass ────────────────────────────────────────────────────


def test_minimal_single_agent_valid():
    """Simplest valid config → no errors."""
    config = ExperimentConfig(
        name="minimal",
        setup=SetupConfig(agents=[AgentSpec(name="a", role="worker")]),
    )
    errors = ConfigValidator.validate(config)
    assert errors == []


def test_full_multi_agent_with_attacks_valid():
    """Complex config → no errors."""
    config = ExperimentConfig(
        name="full",
        setup=SetupConfig(
            agents=[
                AgentSpec(name="supervisor", role="supervisor"),
                AgentSpec(name="worker", role="worker"),
            ],
            edges=[
                TopologyEdge(from_agent="supervisor", to_agent="worker")
            ],
        ),
        attacks=[
            AttackConfig(
                name="atk",
                attack_type="direct_injection",
                target_agent="worker",
                payload="test",
                success_criteria="HACKED",
            )
        ],
        defenses=[
            DefenseConfig(
                name="def",
                defense_type="monitor",
                target_agents=["worker"],
                config={"blocked_patterns": ["HACKED"]},
            )
        ],
    )
    errors = ConfigValidator.validate(config)
    assert errors == []
