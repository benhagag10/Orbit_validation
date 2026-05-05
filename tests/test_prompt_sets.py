"""Tests for the BrowserART PromptSet decoupling layer (Issue #103).

These tests cover the new :class:`PromptSet` abstraction in
:mod:`orbit.scenarios.browser.browserart.prompts` and the :class:`ConditionSpec`
machinery in :mod:`orbit.scenarios.browser.browserart.condition_presets` that
lets the same topology shape be used with different prompt sets without
duplicating a factory function per condition.
"""

from __future__ import annotations

import pytest

from orbit.scenarios.browser.browserart.condition_presets import (
    CONDITION_REGISTRY,
    ConditionSpec,
    WorkerRole,
    WORKER_ROLE_PRESETS,
    build_condition,
    get_condition,
    get_condition_setup,
    register_condition,
    _CONDITION_SPECS,
    _SHAPE_BUILDERS,
)
from orbit.scenarios.browser.browserart.prompts import (
    PROMPT_SETS,
    PromptSet,
    get_prompt_set,
    register_prompt_set,
)


class TestPromptSetRegistry:
    def test_known_sets_present(self):
        expected = {
            "single_agent",
            "star_batch_detailed",
            "star_batch_relaxed",
            "star_step",
            "star_2_specialist",
            "star_3_specialist",
            "star_specialist_detailed",
            "star_specialist_relaxed",
            "mesh_round_robin",
            "mesh_delegation",
        }
        assert expected.issubset(set(PROMPT_SETS.keys()))

    def test_get_prompt_set_returns_instance(self):
        ps = get_prompt_set("single_agent")
        assert isinstance(ps, PromptSet)
        assert ps.name == "single_agent"
        assert ps.single is not None

    def test_get_prompt_set_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown prompt set"):
            get_prompt_set("does_not_exist")


class TestPromptSetShape:
    @pytest.mark.parametrize(
        "name,expected_slots",
        [
            ("single_agent", {"single"}),
            ("star_batch_detailed", {"orchestrator", "workers"}),
            ("star_specialist_detailed", {"orchestrator", "workers"}),
            ("mesh_round_robin", {"peer"}),
            ("mesh_delegation", {"peer"}),
        ],
    )
    def test_shape_populates_expected_slots(self, name, expected_slots):
        ps = get_prompt_set(name)
        populated = set()
        if ps.single is not None:
            populated.add("single")
        if ps.orchestrator is not None:
            populated.add("orchestrator")
        if ps.workers:
            populated.add("workers")
        if ps.peer is not None:
            populated.add("peer")
        assert populated == expected_slots, (
            f"{name}: slots {populated} != expected {expected_slots}"
        )

    def test_four_specialist_sets_share_worker_prompts(self):
        """Detailed and relaxed star_specialist variants differ only in
        the orchestrator prompt — the specialist worker prompts are the
        same object (shared mapping). This is what lets the refactor
        encode 'same workers, different orchestrator' without duplicating
        dict literals."""
        detailed = get_prompt_set("star_specialist_detailed")
        relaxed = get_prompt_set("star_specialist_relaxed")
        assert detailed.orchestrator != relaxed.orchestrator
        assert dict(detailed.workers) == dict(relaxed.workers)


class TestConditionSpecCatalog:
    def test_every_condition_has_spec(self):
        assert set(_CONDITION_SPECS.keys()) == set(CONDITION_REGISTRY.keys())

    @pytest.mark.parametrize("name", sorted(_CONDITION_SPECS.keys()))
    def test_every_spec_resolves_shape_and_prompt_set(self, name):
        spec = _CONDITION_SPECS[name]
        assert spec.shape in _SHAPE_BUILDERS, (
            f"{name}: unknown shape {spec.shape!r}"
        )
        assert spec.prompt_set in PROMPT_SETS, (
            f"{name}: unknown prompt_set {spec.prompt_set!r}"
        )
        if spec.shape != "single":
            assert spec.workers in WORKER_ROLE_PRESETS, (
                f"{name}: unknown workers preset {spec.workers!r}"
            )

    def test_memory_variants_share_prompt_set_with_star_specialist(self):
        """memory_* conditions must reference the same prompt_set as
        star_specialist. The ONLY variable across memory levels is the
        memory_level field — that's the test's invariant."""
        baseline = _CONDITION_SPECS["star_specialist"]
        for level in [
            "memory_none",
            "memory_own_actions",
            "memory_own_reasoning",
            "memory_shared_actions",
            "memory_full",
        ]:
            spec = _CONDITION_SPECS[level]
            assert spec.shape == baseline.shape, f"{level}: shape drift"
            assert spec.prompt_set == baseline.prompt_set, (
                f"{level}: prompt_set drift — should match star_specialist"
            )
            assert spec.workers == baseline.workers, f"{level}: workers drift"
            assert spec.memory_level == level


class TestBuildConditionWithCustomPromptSet:
    """Exercising the core decoupling: same topology, different prompts."""

    def test_swap_prompt_set_changes_all_prompts(self):
        """Build star_specialist with a custom PromptSet and verify the
        resulting SetupConfig picks up the overridden text — the topology
        shape stays identical."""
        baseline = get_condition_setup("star_specialist")

        custom = PromptSet(
            name="test_swap",
            orchestrator="CUSTOM ORCH PROMPT",
            workers={
                "click_specialist": "CUSTOM CLICK",
                "fill_specialist": "CUSTOM FILL",
                "scroll_specialist": "CUSTOM SCROLL",
                "navigate_specialist": "CUSTOM NAV",
            },
        )
        spec = _CONDITION_SPECS["star_specialist"]
        built = build_condition(spec, prompt_set=custom).setup

        # Same topology: same agents, same edges.
        assert [a.name for a in built.agents] == [a.name for a in baseline.agents]
        assert len(built.edges) == len(baseline.edges)

        # Different prompts: each slot picked up from custom set.
        prompts_by_name = {a.name: a.system_prompt for a in built.agents}
        assert prompts_by_name["orchestrator"] == "CUSTOM ORCH PROMPT"
        assert prompts_by_name["click_specialist"] == "CUSTOM CLICK"
        assert prompts_by_name["fill_specialist"] == "CUSTOM FILL"
        assert prompts_by_name["scroll_specialist"] == "CUSTOM SCROLL"
        assert prompts_by_name["navigate_specialist"] == "CUSTOM NAV"

    def test_missing_worker_slot_raises_clearly(self):
        incomplete = PromptSet(
            name="test_incomplete",
            orchestrator="ORCH",
            workers={"click_specialist": "CLICK"},
            # fill_/scroll_/navigate_specialist prompts missing
        )
        spec = _CONDITION_SPECS["star_specialist"]
        with pytest.raises(ValueError, match="workers\\['fill_specialist'\\]"):
            build_condition(spec, prompt_set=incomplete)

    def test_missing_single_slot_raises_clearly(self):
        incomplete = PromptSet(name="test_empty")
        spec = _CONDITION_SPECS["single_agent"]
        with pytest.raises(ValueError, match="single"):
            build_condition(spec, prompt_set=incomplete)

    def test_missing_peer_slot_raises_clearly(self):
        incomplete = PromptSet(name="test_empty")
        spec = _CONDITION_SPECS["mesh_round_robin"]
        with pytest.raises(ValueError, match="peer"):
            build_condition(spec, prompt_set=incomplete)


class TestRegisterCustomCondition:
    """Conditions can be added without editing condition_presets.py."""

    def test_register_and_build_new_condition(self):
        custom_ps = PromptSet(
            name="issue103_custom_ps",
            orchestrator="TEST ORCH",
            workers={"executor": "TEST EXEC"},
        )
        register_prompt_set(custom_ps)

        spec = ConditionSpec(
            shape="star",
            prompt_set="issue103_custom_ps",
            workers="one_executor",
            properties={
                "condition_type": "issue103_custom",
                "execution_style": "batch",
            },
        )
        register_condition("issue103_custom", spec)

        try:
            setup = get_condition_setup("issue103_custom")
            prompts_by_name = {a.name: a.system_prompt for a in setup.agents}
            assert prompts_by_name["orchestrator"] == "TEST ORCH"
            assert prompts_by_name["executor"] == "TEST EXEC"
            assert setup.properties["condition_type"] == "issue103_custom"
        finally:
            # Clean up so we don't pollute other tests that iterate
            # CONDITION_REGISTRY via fresh imports.
            _CONDITION_SPECS.pop("issue103_custom", None)
            PROMPT_SETS.pop("issue103_custom_ps", None)


class TestWorkerRolePresets:
    def test_all_presets_referenced_by_some_condition(self):
        """Every worker-role preset should be in use by at least one
        condition. Keeps the presets from drifting out of sync."""
        in_use = {
            spec.workers
            for spec in _CONDITION_SPECS.values()
            if spec.workers is not None
        }
        assert in_use == set(WORKER_ROLE_PRESETS.keys())

    def test_worker_role_is_frozen(self):
        r = WorkerRole(name="x", role="y")
        with pytest.raises(Exception):
            r.name = "changed"  # type: ignore[misc]
