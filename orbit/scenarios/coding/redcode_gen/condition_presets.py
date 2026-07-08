"""RedCode-Gen condition presets — thin binding over the shared YAML-backed loader.

Conditions ship as runnable configs in the general orbit syntax under
``orbit/scenarios/conditions/redcode_gen/*.yaml``. This module binds the shared loader
(:mod:`orbit.scenarios.conditions`) to this scenario; there is no per-scenario registry to
maintain. Edit or add a condition by editing/adding its YAML.
"""

from __future__ import annotations

from typing import Callable

from orbit.configs.setup import SetupConfig
from orbit.scenarios import conditions as _shared

_SCENARIO = "redcode_gen"


def get_condition_setup(condition: str) -> SetupConfig:
    """Return the SetupConfig for a named condition (loaded from its shipped YAML)."""
    return _shared.get_condition_setup(_SCENARIO, condition)


def list_conditions() -> list[str]:
    """Return a sorted list of available condition names."""
    return _shared.list_conditions(_SCENARIO)


# Back-compat: name -> zero-arg factory that rebuilds the SetupConfig from its YAML.
CONDITION_REGISTRY: dict[str, Callable[[], SetupConfig]] = {
    name: (lambda n=name: get_condition_setup(n)) for name in list_conditions()
}
