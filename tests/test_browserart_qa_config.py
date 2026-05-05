"""
BrowserART QA — Layer 1: Config construction & validation.

Parametrized across (behavior × condition × attack/defense) to verify that
every combination produces a valid ExperimentConfig with correct metadata.

Total: (10 HBB + 10 extension) × 13 conditions × 8 combos = 2,080 cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# BrowserART upstream data (hbb.json) is not redistributed by Orbit
# (upstream is CC BY-NC-ND 4.0). Skip when the user has not fetched.
_DATA_DIR = (
    Path(__file__).parent.parent
    / "orbit" / "scenarios" / "browser" / "browserart" / "data"
)
pytestmark = pytest.mark.skipif(
    not (_DATA_DIR / "hbb.json").exists(),
    reason=(
        "BrowserART hbb.json not fetched. Run "
        "`uv run python scripts/fetch_browserart_data.py` to enable."
    ),
)

from orbit.scenarios.browser.browserart.condition_presets import get_condition_setup
from orbit.scenarios.browser.browserart.config_builder import build_experiment_config
from orbit.scenarios.browser.browserart.configs import (
    BrowserARTBehavior,
    BrowserARTScenarioConfig,
)
from orbit.scenarios.browser.browserart.dataset_builder import load_browserart_behaviors
from orbit.configs.experiment import ExperimentConfig

from tests.conftest_browserart_qa import (
    CANONICAL_CONDITIONS,
    CONDITION_AGENT_COUNT,
    CONDITION_TOPOLOGY_TYPE,
    EXTENSION_IDS,
    PRESET_COMBOS,
    SELECTED_HBB_IDS,
    build_config_for_test,
    get_attack_configs,
    get_defense_configs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hbb_behaviors() -> list[BrowserARTBehavior]:
    return load_browserart_behaviors(BrowserARTScenarioConfig(dataset="hbb"))

@pytest.fixture(scope="module")
def extension_behaviors() -> list[BrowserARTBehavior]:
    return load_browserart_behaviors(BrowserARTScenarioConfig(dataset="hbb_extension"))

@pytest.fixture(scope="module")
def hbb_by_id(hbb_behaviors) -> dict[int, BrowserARTBehavior]:
    return {b.behavior_id: b for b in hbb_behaviors}

@pytest.fixture(scope="module")
def ext_by_id(extension_behaviors) -> dict[int, BrowserARTBehavior]:
    return {b.behavior_id: b for b in extension_behaviors}


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

def _combo_id(combo):
    atk, dfn = combo
    return f"atk={atk or 'none'}_def={dfn or 'none'}"


# ---------------------------------------------------------------------------
# Test: every (HBB behavior × condition × combo) builds without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("behavior_id", SELECTED_HBB_IDS, ids=[f"hbb_{i}" for i in SELECTED_HBB_IDS])
@pytest.mark.parametrize("condition", CANONICAL_CONDITIONS)
@pytest.mark.parametrize("combo", PRESET_COMBOS, ids=[_combo_id(c) for c in PRESET_COMBOS])
def test_hbb_config_builds_without_error(behavior_id, condition, combo, hbb_by_id):
    """Every HBB (behavior × condition × attack/defense) produces a valid config."""
    behavior = hbb_by_id[behavior_id]
    attack_name, defense_name = combo
    config = build_config_for_test(behavior, condition, attack_name, defense_name)
    assert isinstance(config, ExperimentConfig)
    assert config.name.startswith("browserart_")


@pytest.mark.parametrize("behavior_id", EXTENSION_IDS, ids=[f"ext_{i}" for i in EXTENSION_IDS])
@pytest.mark.parametrize("condition", CANONICAL_CONDITIONS)
@pytest.mark.parametrize("combo", PRESET_COMBOS, ids=[_combo_id(c) for c in PRESET_COMBOS])
def test_extension_config_builds_without_error(behavior_id, condition, combo, ext_by_id):
    """Every extension (behavior × condition × attack/defense) produces a valid config."""
    behavior = ext_by_id[behavior_id]
    attack_name, defense_name = combo
    config = build_config_for_test(behavior, condition, attack_name, defense_name)
    assert isinstance(config, ExperimentConfig)
    assert config.name.startswith("browserart_")


# ---------------------------------------------------------------------------
# Test: metadata completeness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("behavior_id", SELECTED_HBB_IDS[:3])
@pytest.mark.parametrize("condition", CANONICAL_CONDITIONS)
def test_config_metadata_complete(behavior_id, condition, hbb_by_id):
    """All required metadata keys are present."""
    behavior = hbb_by_id[behavior_id]
    config = build_config_for_test(behavior, condition)
    md = config.metadata

    required_keys = [
        "browserart_behavior_id",
        "browserart_behavior",
        "browserart_text_behavior",
        "browserart_website",
        "browserart_start_url",
        "browserart_semantic_category",
        "browserart_evaluation_category",
        "browserart_source",
        "browserart_dataset_source",
    ]
    for key in required_keys:
        assert key in md, f"Missing metadata key: {key}"

    assert md["browserart_behavior_id"] == behavior_id
    assert md["browserart_dataset_source"] == "BrowserART"


# ---------------------------------------------------------------------------
# Test: topology agent count matches condition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition", CANONICAL_CONDITIONS)
def test_topology_agent_count(condition):
    """Each condition produces the expected number of agents."""
    setup = get_condition_setup(condition)
    expected = CONDITION_AGENT_COUNT[condition]
    assert len(setup.agents) == expected, (
        f"Condition {condition}: expected {expected} agents, got {len(setup.agents)}"
    )


# ---------------------------------------------------------------------------
# Test: topology edge count matches structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition", CANONICAL_CONDITIONS)
def test_topology_edge_count(condition):
    """Edge counts match topology structure."""
    setup = get_condition_setup(condition)
    topo_type = setup.properties.get("topology_type", "single")

    if topo_type == "single":
        assert len(setup.edges) == 0
    elif topo_type == "star":
        # Star: one edge per worker (orch → worker)
        n_workers = len(setup.agents) - 1
        assert len(setup.edges) == n_workers
    elif topo_type in ("mesh_round_robin", "mesh_delegation"):
        # Complete graph: n*(n-1) directed edges
        n = len(setup.agents)
        assert len(setup.edges) == n * (n - 1)


# ---------------------------------------------------------------------------
# Test: condition_type in metadata when condition is used
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition", [c for c in CANONICAL_CONDITIONS if c != "single_agent"])
def test_condition_type_in_metadata(condition, hbb_by_id):
    """Conditions with topology_type != single embed browserart_condition_type."""
    behavior = hbb_by_id[0]
    config = build_config_for_test(behavior, condition)
    md = config.metadata
    # condition_type is set in properties for all non-single conditions
    setup = config.setup
    assert "condition_type" in setup.properties
    assert setup.properties["condition_type"] == condition
    # Also propagated to metadata
    assert md.get("browserart_condition_type") == condition


# ---------------------------------------------------------------------------
# Test: extension metadata flags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("behavior_id", EXTENSION_IDS[:3])
def test_extension_metadata_flags(behavior_id, ext_by_id):
    """Extension configs have multi_step=True and correct min_steps."""
    behavior = ext_by_id[behavior_id]
    config = build_config_for_test(behavior, "single_agent")
    md = config.metadata
    assert md["browserart_multi_step"] is True
    assert md["browserart_min_steps"] is not None
    assert md["browserart_min_steps"] >= 25


# ---------------------------------------------------------------------------
# Test: extension dataset source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("behavior_id", EXTENSION_IDS[:3])
def test_extension_dataset_source(behavior_id, ext_by_id):
    """Extension configs show Orbit-Extension as dataset source."""
    behavior = ext_by_id[behavior_id]
    config = build_config_for_test(behavior, "single_agent")
    assert config.metadata["browserart_dataset_source"] == "Orbit-Extension"
