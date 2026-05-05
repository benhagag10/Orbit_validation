"""AgentHarm scenario: harmful tool-use safety benchmark.

Orbit port of AgentHarm (Andriushchenko et al., ICLR 2025). See
``README.md`` in this directory for attribution and license details.

Importing this package registers all 80 harmful + 80 benign vendored tools
with Orbit's ``_TOOL_REGISTRY`` under the naming convention
``agentharm_{harmful,benign}_<tool_name>`` so that per-sample ``AgentSpec``
tool lists can reference them by name.
"""

from __future__ import annotations

import logging

from orbit.agents.agent_factory import _TOOL_REGISTRY
from orbit.scenarios.agentharm.vendor import (
    benign_tools as _benign_tools_pkg,
    harmful_tools as _harmful_tools_pkg,
)

logger = logging.getLogger(__name__)


def _register_agentharm_tools() -> int:
    """Populate ``_TOOL_REGISTRY`` with the vendored AgentHarm tools.

    Idempotent — safe to call multiple times. Returns the number of
    entries that were added on this call.
    """
    added = 0
    specs = (
        ("agentharm_harmful_", "orbit.scenarios.agentharm.vendor.harmful_tools", _harmful_tools_pkg),
        ("agentharm_benign_", "orbit.scenarios.agentharm.vendor.benign_tools", _benign_tools_pkg),
    )
    for prefix, module_path, pkg in specs:
        for tool_name in pkg.__all__:
            registry_key = f"{prefix}{tool_name}"
            if registry_key in _TOOL_REGISTRY:
                continue
            _TOOL_REGISTRY[registry_key] = (module_path, tool_name, {})
            added += 1
    return added


_registered = _register_agentharm_tools()
if _registered:
    logger.debug("Registered %d AgentHarm tools with orbit _TOOL_REGISTRY", _registered)
