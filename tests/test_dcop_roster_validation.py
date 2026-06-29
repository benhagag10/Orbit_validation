"""DCOP roster reconciliation — setup.agents must match the generated world.

A DCOP scenario (jira_ticket, hospital, meeting_scheduling) generates its whole
world, including the agent roster, from ``scenario.properties``. The hand-written
``setup.agents`` list in a YAML config is a separate declaration of that same
roster, so the two can silently drift apart (GitHub issue #3). ConfigValidator
reconciles them and fails loudly on a mismatch.

The hospital scenario needs no optional extras, so the core regression for
issue #3 runs everywhere. jira_ticket / meeting_scheduling need scipy (the
``dcop`` extra) to build their problem; those cases skip when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit.configs.experiment import ExperimentConfig
from orbit.configs.scenario import ScenarioConfig
from orbit.configs.setup import AgentSpec, SetupConfig
from orbit.scenarios.cooperative_allocation.dcop.problem_factory import (
    expected_agent_roster,
)
from orbit.validation.validators import ConfigValidator
from orbit.wrapper.yaml_loader import load_experiment_config

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# DCOP example configs whose roster is generated from scenario.properties.
DCOP_EXAMPLE_YAMLS = [
    "collusion_hospital.yaml",
    "collusion_jira_ticket.yaml",
    "collusion_meeting_scheduling.yaml",
    "jira_ticket_baseline.yaml",
    "jira_ticket_channel_only.yaml",
]


def _hospital_config(agent_names: list[str], **properties) -> ExperimentConfig:
    props = {"num_hospitals": 2, "departments_per_hospital": 2, **properties}
    return ExperimentConfig(
        name="hospital-roster-test",
        scenario=ScenarioConfig(name="hospital", properties=props),
        setup=SetupConfig(
            agents=[AgentSpec(name=n, role="scheduler") for n in agent_names]
        ),
    )


# ── Roster derivation (single source of truth) ────────────────────────────


def test_hospital_roster_from_properties():
    """departments_per_hospital=2 -> triage + radiology per hospital, + provisioner."""
    roster = expected_agent_roster("hospital", {"departments_per_hospital": 2})
    assert roster == {
        "hospital_0_triage",
        "hospital_0_radiology",
        "hospital_1_triage",
        "hospital_1_radiology",
        "provisioner",
    }


def test_hospital_roster_excludes_provisioner_when_disabled():
    roster = expected_agent_roster(
        "hospital", {"departments_per_hospital": 2, "include_provisioner": False}
    )
    assert "provisioner" not in roster


def test_non_dcop_scenario_has_no_derivable_roster():
    """Non-DCOP scenarios own their roster in setup.agents; nothing to derive."""
    assert expected_agent_roster("browserart", {}) is None
    assert expected_agent_roster("swe_bench", {}) is None
    assert expected_agent_roster("test", {}) is None


def test_jira_and_meeting_roster_from_properties():
    pytest.importorskip("scipy")  # jira/meeting builders need the dcop extra
    assert expected_agent_roster("jira_ticket", {"num_developers": 3}) == {
        "dev_0",
        "dev_1",
        "dev_2",
    }
    assert expected_agent_roster("meeting_scheduling", {"num_agents": 4}) == {
        "agent_0",
        "agent_1",
        "agent_2",
        "agent_3",
    }


# ── The guard: drift fails loudly (issue #3 regression) ────────────────────


def test_drifted_hospital_roster_is_rejected():
    """Issue #3: at departments_per_hospital=2 only triage+radiology exist, so a
    config that staffs 'surgery' instead of 'radiology' leaves radiology
    unstaffed. The missing (generated-but-unstaffed) agents must be flagged."""
    config = _hospital_config(
        [
            "hospital_0_triage",
            "hospital_0_surgery",
            "hospital_1_triage",
            "hospital_1_surgery",
            "provisioner",
        ]
    )
    errors = ConfigValidator.validate(config)
    roster_errors = [e for e in errors if "missing agents that DCOP scenario" in e]
    assert len(roster_errors) == 1
    msg = roster_errors[0]
    # radiology is generated but unstaffed -> reported as missing.
    assert "hospital_0_radiology" in msg and "hospital_1_radiology" in msg


def test_matching_hospital_roster_passes():
    config = _hospital_config(
        [
            "hospital_0_triage",
            "hospital_0_radiology",
            "hospital_1_triage",
            "hospital_1_radiology",
            "provisioner",
        ]
    )
    errors = ConfigValidator.validate(config)
    assert not any("missing agents that DCOP scenario" in e for e in errors)


def test_extra_non_generated_agent_is_allowed():
    """A DCOP config may add non-task agents (monitor / guardian / observer)
    beyond the generated roster; only *missing* generated agents are an error."""
    config = _hospital_config(
        [
            "hospital_0_triage",
            "hospital_0_radiology",
            "hospital_1_triage",
            "hospital_1_radiology",
            "provisioner",
            "safety_monitor",  # extra observer, not part of the generated world
        ]
    )
    errors = ConfigValidator.validate(config)
    assert not any("missing agents that DCOP scenario" in e for e in errors)


def test_missing_agent_is_reported():
    """A generated agent absent from setup.agents is flagged as missing."""
    config = _hospital_config(
        ["hospital_0_triage", "hospital_0_radiology", "hospital_1_triage", "provisioner"]
    )
    errors = ConfigValidator.validate(config)
    assert any(
        "missing agents that DCOP scenario" in e and "hospital_1_radiology" in e
        for e in errors
    )


def test_non_dcop_config_skips_roster_check():
    """A non-DCOP scenario with an arbitrary roster raises no roster error."""
    config = ExperimentConfig(
        name="test",
        scenario=ScenarioConfig(name="test"),
        setup=SetupConfig(
            agents=[AgentSpec(name="whatever", role="worker")]
        ),
    )
    errors = ConfigValidator.validate(config)
    assert not any("missing agents that DCOP scenario" in e for e in errors)


def test_genuine_importerror_is_surfaced_when_scipy_present(monkeypatch):
    """A real ImportError during roster derivation must surface, not be swallowed.

    The 'extra absent -> skip' path is only safe when the optional `dcop` extra
    (scipy) is genuinely missing. When scipy IS installed, an ImportError is a
    real bug (e.g. in the hospital builder, which needs no optional deps); silently
    skipping it would re-open issue #3 for hospital. So we only skip when scipy is
    actually absent.
    """
    pytest.importorskip("scipy")  # the guarantee under test is: scipy present -> surface

    import orbit.scenarios.cooperative_allocation.dcop.problem_factory as pf

    def _boom(scenario_name, properties):
        raise ImportError("simulated import bug in a path that needs no optional deps")

    monkeypatch.setattr(pf, "expected_agent_roster", _boom)

    config = _hospital_config(
        [
            "hospital_0_triage",
            "hospital_0_radiology",
            "hospital_1_triage",
            "hospital_1_radiology",
            "provisioner",
        ]
    )
    errors = ConfigValidator.validate(config)
    assert any("Could not derive the agent roster" in e for e in errors)


# ── Shipped DCOP examples stay consistent ──────────────────────────────────


@pytest.mark.parametrize("yaml_name", DCOP_EXAMPLE_YAMLS)
def test_dcop_example_yamls_validate_clean(yaml_name):
    """Every shipped DCOP example must satisfy the roster guard (guards against
    re-introducing issue #3 in any example)."""
    config = load_experiment_config(EXAMPLES_DIR / yaml_name)
    errors = ConfigValidator.validate(config)
    assert errors == [], f"{yaml_name}: {errors}"
