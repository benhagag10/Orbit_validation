"""Shared loader for shipped condition configs.

A *condition* is nothing more than a pre-baked :class:`~orbit.configs.setup.SetupConfig`
(agents + edges + memory) — a slice of the top-level :class:`~orbit.configs.experiment.ExperimentConfig`.
Each condition ships as a runnable YAML in the general orbit syntax under
``orbit/scenarios/conditions/<scenario>/<name>.yaml``; users can ``orbit run`` any of them directly.

This module is the single place that loads those configs. Every CONFIG-source scenario resolves its
named conditions from here instead of reimplementing a per-scenario registry, so the condition
system has one shared shape rather than eight divergent ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from orbit.configs.execution import ExecutionConfig
from orbit.configs.setup import SetupConfig

_CONDITIONS_ROOT = Path(__file__).parent / "conditions"


@dataclass(frozen=True)
class ConditionSetup:
    """A condition's topology plus optional execution config.

    ``execution`` is ``None`` for shared-state (non-scheduled) conditions. The scenario task drops
    both into :class:`~orbit.configs.experiment.ExperimentConfig` verbatim.
    """

    setup: SetupConfig
    execution: ExecutionConfig | None = None


def _scenario_dir(scenario: str) -> Path:
    return _CONDITIONS_ROOT / scenario


def list_conditions(scenario: str) -> list[str]:
    """Return the sorted names of conditions shipped for ``scenario``."""
    return sorted(p.stem for p in _scenario_dir(scenario).glob("*.yaml"))


def _load_doc(scenario: str, name: str) -> dict:
    path = _scenario_dir(scenario) / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_conditions(scenario)) or "(none)"
        raise ValueError(
            f"Unknown condition {name!r} for scenario {scenario!r}. Available: {available}"
        )
    return yaml.safe_load(path.read_text())


def get_condition(scenario: str, name: str) -> ConditionSetup:
    """Load a named condition for ``scenario`` as a :class:`ConditionSetup`."""
    doc = _load_doc(scenario, name)
    setup = SetupConfig.model_validate(doc["setup"])
    raw_exec = doc.get("execution")
    execution = ExecutionConfig.model_validate(raw_exec) if raw_exec else None
    return ConditionSetup(setup=setup, execution=execution)


def get_condition_setup(scenario: str, name: str) -> SetupConfig:
    """Load only the :class:`~orbit.configs.setup.SetupConfig` for a named condition."""
    return get_condition(scenario, name).setup


def condition_path(scenario: str, name: str) -> Path:
    """Return the on-disk path of a condition's shipped YAML (for docs/tooling)."""
    return _scenario_dir(scenario) / f"{name}.yaml"
