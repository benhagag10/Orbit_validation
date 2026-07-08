"""Equivalence gate for the conditions-as-YAML refactor.

Every shipped condition YAML must reconstruct exactly the SetupConfig (and, where present, the
ExecutionConfig) that the scenario's registry produces. This is the regression lock that makes the
YAML the source of truth safe: if a YAML ever drifts from what the code builds, this fails.
"""
from __future__ import annotations

import importlib

import pytest

from orbit.scenarios import conditions as shared

# scenario -> registry module. CONFIG-source scenarios only (tau2/agentharm build rosters
# per-sample and are intentionally not converted).
_REGISTRY_MODULES = {
    "browserart": "orbit.scenarios.browser.browserart.condition_presets",
    "swe_bench": "orbit.scenarios.coding.swe_bench.condition_presets",
    "osworld": "orbit.scenarios.desktop.osworld.condition_presets",
    "redcode_gen": "orbit.scenarios.coding.redcode_gen.condition_presets",
    "bigcodebench": "orbit.scenarios.coding.bigcodebench.condition_presets",
    "code_ipi": "orbit.scenarios.coding.code_ipi.condition_presets",
    "converse": "orbit.scenarios.customer_service.converse.condition_presets",
}


def _cases():
    for scenario in _REGISTRY_MODULES:
        for name in shared.list_conditions(scenario):
            yield scenario, name


@pytest.mark.parametrize("scenario,name", list(_cases()))
def test_shipped_yaml_matches_registry(scenario: str, name: str):
    module = importlib.import_module(_REGISTRY_MODULES[scenario])
    if hasattr(module, "get_condition"):
        expected = module.get_condition(name)
        exp_setup, exp_exec = expected.setup, expected.execution
    else:
        exp_setup, exp_exec = module.get_condition_setup(name), None

    loaded = shared.get_condition(scenario, name)
    assert loaded.setup == exp_setup, f"{scenario}/{name}: setup drift"
    assert loaded.execution == exp_exec, f"{scenario}/{name}: execution drift"


def test_shipped_condition_names_match_registry():
    for scenario, modpath in _REGISTRY_MODULES.items():
        module = importlib.import_module(modpath)
        assert sorted(shared.list_conditions(scenario)) == sorted(module.list_conditions()), (
            f"{scenario}: shipped YAML set differs from registry list_conditions()"
        )
