"""
Baseline application logic.

Modifies ExperimentConfig to create baseline variants by stripping
attacks and/or defenses. Uses Pydantic model_copy(update=...) since
configs are frozen.

No direct Inspect interaction. Pure config transformation.
"""

from __future__ import annotations

from orbit.configs.baseline import BaselineMode
from orbit.configs.experiment import ExperimentConfig


def apply_baseline(
    config: ExperimentConfig,
    mode: BaselineMode,
) -> ExperimentConfig:
    """Create a baseline variant of an experiment config.

    Args:
        config: Original experiment configuration.
        mode: Which baseline to apply:
            - NONE: Return config unchanged
            - NO_DEFENSE: Strip defenses, keep attacks
            - NO_ATTACK: Strip attacks, keep defenses
            - BENIGN: Strip both attacks and defenses

    Returns:
        New ExperimentConfig with the baseline applied.
        Original config is NOT mutated (frozen Pydantic model).
    """
    if mode == BaselineMode.NONE:
        return config

    updates: dict = {"baseline_mode": mode}

    if mode == BaselineMode.NO_DEFENSE:
        updates["defenses"] = []
    elif mode == BaselineMode.NO_ATTACK:
        updates["attacks"] = []
    elif mode == BaselineMode.BENIGN:
        updates["attacks"] = []
        updates["defenses"] = []

    return config.model_copy(update=updates)
