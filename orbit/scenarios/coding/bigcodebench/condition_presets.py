"""BigCodeBench condition presets.

Thin re-export of the RedCode-Gen condition registry. The topology
variants (single_agent, star_*, mesh_*, memory_*) are already generic
coding-team setups — no malware-specific wiring — so the same builders
work unchanged for benign BigCodeBench tasks.

Having this module keeps the scenario's public surface parallel with
``redcode_gen`` so scripts and configs can switch scenarios without
learning a second API.
"""

from __future__ import annotations

from orbit.scenarios.coding.redcode_gen.condition_presets import (  # noqa: F401
    CONDITION_REGISTRY,
    get_condition_setup,
    list_conditions,
)
