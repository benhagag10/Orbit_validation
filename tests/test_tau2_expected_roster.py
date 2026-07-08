"""Pin tau2's static roster map to the real topology builders + factory parity.

``topologies.expected_agent_names(condition, domain)`` is a hand-maintained map
consumed by the shared shorthand resolver to arbitrate an inline ``setup.agents``
block against the roster a ``tau2_condition`` will actually build (issue #32). If
it drifts from the real ``build_topology`` output, the resolver silently
mis-arbitrates. :class:`TestExpectedRosterMatchesBuilders` walks every
condition × domain and asserts the map equals the agent names of the *real* plan
built from a vendored task, so the two cannot diverge.

:class:`TestFactoryDefensePresetParity` covers issue #35: the ``tau2`` ``@task``
factory now exposes a ``defense_preset`` parameter (YAML already had
``*_defense_preset`` shorthand; this closes the entry-point gap), and
``get_defense_preset`` rejects an unknown name with a helpful message.

Data is vendored, so the roster tests run offline; a load failure skips with the
reason rather than failing (e.g. a stripped checkout).
"""

from __future__ import annotations

import inspect

import pytest

from orbit.scenarios.customer_service.tau2.task import (
    _SUPPORTED_CONDITIONS,
    _SUPPORTED_DOMAINS,
)
from orbit.scenarios.customer_service.tau2.topologies import (
    build_topology,
    expected_agent_names,
)


def _load_first_task(domain: str, condition: str):
    """Load a real vendored task + policy + scenario_config for a combo.

    Mirrors ``tau2.task._tau2_expand``'s loading path. Returns
    ``(scenario_config, task, policy_text)``. ``mode`` is derived from the
    condition exactly as the factory derives it (``solo`` ⇒ solo mode, else
    dual_control), which ``build_topology`` requires to match ``condition``.
    """
    from orbit.scenarios.customer_service.tau2.configs import Tau2ScenarioConfig
    from orbit.scenarios.customer_service.tau2.dataset_builder import (
        load_policy,
        load_tau2_tasks,
    )

    mode = "solo" if condition == "solo" else "dual_control"
    scenario_config = Tau2ScenarioConfig(
        domain=domain,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        topology=condition,  # type: ignore[arg-type]
    )
    tasks = load_tau2_tasks(scenario_config)
    if not tasks:
        raise RuntimeError(f"no vendored tau2 tasks for domain={domain!r}")
    return scenario_config, tasks[0], load_policy(scenario_config)


class TestExpectedRosterMatchesBuilders:
    """expected_agent_names(condition, domain) == the real plan's roster."""

    @pytest.mark.parametrize("domain", _SUPPORTED_DOMAINS)
    @pytest.mark.parametrize("condition", _SUPPORTED_CONDITIONS)
    def test_static_roster_matches_real_plan(self, condition, domain):
        # solo is an Orbit-specific airline-only baseline; every other combo is
        # valid (all three domains run dual-control and its multi-agent shapes).
        if condition == "solo" and domain != "airline":
            pytest.skip("condition='solo' is airline-only (Orbit baseline)")

        try:
            scenario_config, task, policy = _load_first_task(domain, condition)
        except Exception as exc:  # noqa: BLE001 - vendored data may be absent
            pytest.skip(f"could not load vendored tau2 {domain} data: {exc}")

        plan = build_topology(condition, task, scenario_config, policy)
        real_roster = {spec.name for spec in plan.agents}

        expected = expected_agent_names(condition, domain)
        assert expected is not None, (
            f"expected_agent_names({condition!r}, {domain!r}) returned None for a "
            f"supported condition — the static map is missing an entry."
        )
        assert expected == real_roster, (
            f"expected_agent_names({condition!r}, {domain!r})={sorted(expected)} "
            f"drifted from the real build_topology roster {sorted(real_roster)}. "
            f"Update the rosters map in topologies.expected_agent_names."
        )

    def test_unknown_condition_returns_none(self):
        assert expected_agent_names("not_a_condition", "airline") is None


class TestFactoryDefensePresetParity:
    """Issue #35: the tau2 factory exposes defense_preset; unknown names error."""

    def test_tau2_factory_exposes_defense_preset(self):
        from orbit.scenarios.customer_service.tau2.task import tau2

        assert "defense_preset" in inspect.signature(tau2).parameters

    def test_tau2_unknown_defense_preset_lists_valid_names(self):
        from orbit.scenarios.customer_service.tau2.presets import (
            get_defense_preset,
            list_defense_presets,
        )

        with pytest.raises(ValueError) as excinfo:
            get_defense_preset("does_not_exist")
        message = str(excinfo.value)
        assert "does_not_exist" in message
        for name in list_defense_presets():
            assert name in message

    @pytest.mark.parametrize("preset", ["prompt_vaccination", "dual_llm"])
    def test_presets_target_all_agents(self, preset):
        # The presets leave target_agents empty (== all agents in the
        # condition-derived roster). See the presets module's docstring.
        from orbit.scenarios.customer_service.tau2.presets import get_defense_preset

        for defense in get_defense_preset(preset):
            assert defense.target_agents == []
