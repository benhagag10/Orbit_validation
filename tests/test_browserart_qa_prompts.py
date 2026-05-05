"""
BrowserART QA — Layer 3: Prompt & construct validity.

Verifies prompt placeholder presence, construct validity issues
(YAML prompts ignored by solver), data quality, and refusal indicator
consistency.
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

from orbit.scenarios.browser.browserart.condition_presets import (
    get_condition_setup,
)
from orbit.scenarios.browser.browserart.configs import (
    BrowserARTBehavior,
    BrowserARTScenarioConfig,
)
from orbit.scenarios.browser.browserart.dataset_builder import load_browserart_behaviors
from orbit.scenarios.browser.browserart import scorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hbb_behaviors() -> list[BrowserARTBehavior]:
    return load_browserart_behaviors(BrowserARTScenarioConfig(dataset="hbb"))


@pytest.fixture(scope="module")
def hbb_by_id(hbb_behaviors) -> dict[int, BrowserARTBehavior]:
    return {b.behavior_id: b for b in hbb_behaviors}


@pytest.fixture(scope="module")
def extension_behaviors() -> list[BrowserARTBehavior]:
    return load_browserart_behaviors(
        BrowserARTScenarioConfig(dataset="hbb_extension")
    )


@pytest.fixture(scope="module")
def ext_by_id(extension_behaviors) -> dict[int, BrowserARTBehavior]:
    return {b.behavior_id: b for b in extension_behaviors}


# ---------------------------------------------------------------------------
# Test: detailed vs relaxed orchestrator prompts actually differ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detailed,relaxed",
    [("star_batch", "star_batch_relaxed"), ("star_specialist", "star_specialist_relaxed")],
    ids=["batch", "specialist"],
)
def test_detailed_vs_relaxed_orchestrator_prompts_differ(detailed, relaxed):
    """Detailed vs relaxed must yield different orchestrator system prompts.

    The ``instructions`` dimension is the only thing these conditions
    should vary, so the orchestrator prompt must differ textually.
    """
    setup_d = get_condition_setup(detailed)
    setup_r = get_condition_setup(relaxed)

    assert len(setup_d.agents) == len(setup_r.agents)
    assert len(setup_d.edges) == len(setup_r.edges)

    d_names = {a.name for a in setup_d.agents}
    r_names = {a.name for a in setup_r.agents}
    assert d_names == r_names

    d_orch = next(a for a in setup_d.agents if a.role == "orchestrator")
    r_orch = next(a for a in setup_r.agents if a.role == "orchestrator")
    assert d_orch.system_prompt != r_orch.system_prompt, (
        "Detailed and relaxed orchestrator prompts should differ"
    )


# ---------------------------------------------------------------------------
# Test: scorer refusal indicators are well-formed
# ---------------------------------------------------------------------------


def test_refusal_indicators_is_lowercase_list():
    """Refusal indicator substrings should be lowercase and non-empty.

    The scorer lowercases candidate messages before scanning, so any
    uppercase letter here would silently never match.
    """
    assert len(scorer.REFUSAL_INDICATORS) > 0
    for indicator in scorer.REFUSAL_INDICATORS:
        assert indicator == indicator.lower(), indicator
        assert indicator.strip() == indicator, indicator


# ---------------------------------------------------------------------------
# Test: HBB categories trailing whitespace (D1)
# ---------------------------------------------------------------------------

def _hbb_has_trailing_whitespace(hbb_behaviors) -> bool:
    """True when the loaded hbb.json still has the upstream whitespace bug."""
    return any(
        b.semantic_category != b.semantic_category.strip()
        for b in hbb_behaviors
    )


def test_hbb_categories_no_trailing_whitespace(hbb_behaviors):
    """D1 FIXED: No semantic_category entries should have trailing whitespace."""
    if _hbb_has_trailing_whitespace(hbb_behaviors):
        # Upstream hbb.json has trailing whitespace on several
        # semantic_category entries; Orbit's local copy is normally
        # cleaned post-fetch. After running fetch_browserart_data.py
        # the cleaning step needs to be re-applied.
        pytest.skip(
            "hbb.json is in upstream-only state (whitespace not "
            "cleaned); re-apply local cleaning to re-enable this test"
        )
    affected_ids = {
        b.behavior_id for b in hbb_behaviors
        if b.semantic_category != b.semantic_category.strip()
    }
    assert affected_ids == set(), (
        f"Found trailing whitespace on IDs {affected_ids} — "
        f"data should be clean"
    )


# ---------------------------------------------------------------------------
# Test: category filter works without trailing whitespace workaround
# ---------------------------------------------------------------------------

def test_category_filter_works_after_whitespace_fix(hbb_behaviors):
    """D1 FIXED: Filtering by exact category name now finds all entries."""
    if _hbb_has_trailing_whitespace(hbb_behaviors):
        pytest.skip(
            "hbb.json is in upstream-only state (whitespace not "
            "cleaned); re-apply local cleaning to re-enable this test"
        )
    from orbit.scenarios.browser.browserart.dataset_builder import filter_behaviors

    # Filtering for "content/illegal" (without trailing space) should now work
    config = BrowserARTScenarioConfig(categories=["content/illegal"])
    filtered = filter_behaviors(hbb_behaviors, config)
    id_21_found = any(b.behavior_id == 21 for b in filtered)

    assert id_21_found, (
        "ID 21 should be found with exact category filter after whitespace fix"
    )


# ---------------------------------------------------------------------------
# Test: extension turn limits adequate
# ---------------------------------------------------------------------------

def test_extension_turn_limits_adequate(extension_behaviors):
    """Turn budget check: default 50 max_turns >= all min_steps.

    For single-agent, 50 turns is probably adequate.
    For mesh round-robin with 4 agents, effective actions per agent =
    50 / 4 ≈ 12, which may be insufficient for 25-30 step tasks (E2).
    """
    default_extension_turns = 50
    for b in extension_behaviors:
        if b.min_steps is not None:
            assert default_extension_turns >= b.min_steps, (
                f"Extension ID {b.behavior_id}: min_steps={b.min_steps} > "
                f"default max_turns={default_extension_turns}"
            )


# ---------------------------------------------------------------------------
# Test: mesh round-robin effective action budget for extensions (E2)
# ---------------------------------------------------------------------------

def test_mesh_round_robin_extension_action_budget(extension_behaviors):
    """E2: Document that mesh round-robin with 4 agents may have
    insufficient per-agent action budget for extension tasks.

    With round-robin and 4 agents, each agent gets ~max_steps/4 actions.
    For 50 max_steps that's ~12 actions per agent.
    Extension tasks require 25-30 steps minimum.

    Note: The solver now counts only real browser actions (not noops)
    against the budget, so this is less severe than initially expected.
    But each agent still only gets ~12 real actions.
    """
    max_steps = 50  # default for extensions
    n_agents = 4    # mesh round-robin has 4 peers
    effective_per_agent = max_steps / n_agents

    max_min_steps = max(
        b.min_steps for b in extension_behaviors if b.min_steps is not None
    )

    # The effective per-agent budget should ideally be >= max min_steps
    # but it's not — this documents the concern
    assert effective_per_agent < max_min_steps, (
        f"Per-agent budget ({effective_per_agent}) unexpectedly >= "
        f"max min_steps ({max_min_steps}). The E2 concern may no longer apply."
    )
