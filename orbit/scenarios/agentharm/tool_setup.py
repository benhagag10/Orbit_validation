"""Per-sample tool resolution for AgentHarm.

Upstream ``inspect_evals.agentharm.utils.setup_tools_from_metadata`` loads
only the tools named in a sample's ``target_functions`` metadata, optionally
padding with a random selection of irrelevant tools. We replicate that logic
here as a synchronous resolver that returns an Inspect ``Tool`` list, so
Orbit can wire the per-sample tool set through its usual scenario pipeline
without needing an Inspect-level solver in the middle.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Literal

from orbit.scenarios.agentharm.tool_domains import (
    ToolDomain,
    domain_of,
    tools_in_domain,
)
from orbit.scenarios.agentharm.vendor import (
    benign_tools as _benign_tools_pkg,
    harmful_tools as _harmful_tools_pkg,
)

if TYPE_CHECKING:
    from inspect_ai.tool import Tool

logger = logging.getLogger(__name__)

AgentHarmSplit = Literal["harmful", "benign", "chat"]


def all_tool_names(split: AgentHarmSplit) -> list[str]:
    """Return every tool name available for the given split (sorted)."""
    if split == "harmful":
        return sorted(_harmful_tools_pkg.__all__)
    if split == "benign":
        return sorted(_benign_tools_pkg.__all__)
    return []


def instantiate_tool(split: AgentHarmSplit, name: str) -> "Tool":
    """Instantiate a single tool by name from the vendored package."""
    pkg = _harmful_tools_pkg if split == "harmful" else _benign_tools_pkg
    factory = getattr(pkg, name, None)
    if factory is None:
        raise ValueError(
            f"Tool {name!r} not found in agentharm {split} tool package."
        )
    return factory()


def resolve_tools_for_sample(
    split: AgentHarmSplit,
    target_functions: list[str] | tuple[str, ...],
    *,
    n_irrelevant_tools: int = 0,
    seed: int | None = None,
) -> list["Tool"]:
    """Resolve the full (solo-agent) tool list for one sample.

    Mirrors the upstream behaviour exactly: instantiate each
    ``target_functions`` entry in declaration order, then append
    ``n_irrelevant_tools`` randomly-sampled distractor tools, then shuffle
    the combined list.
    """
    if split == "chat":
        return []

    tools: list["Tool"] = [instantiate_tool(split, name) for name in target_functions]

    if n_irrelevant_tools > 0:
        known = set(target_functions)
        candidates = [n for n in all_tool_names(split) if n not in known]
        rng = random.Random(seed)
        k = min(n_irrelevant_tools, len(candidates))
        for name in rng.sample(candidates, k):
            tools.append(instantiate_tool(split, name))

    rng = random.Random(seed)
    rng.shuffle(tools)
    return tools


def resolve_tools_for_specialist(
    split: AgentHarmSplit,
    target_functions: list[str] | tuple[str, ...],
    *,
    domain: ToolDomain,
    tool_mode: Literal["filtered", "full_domain"] = "filtered",
    n_irrelevant_tools: int = 0,
    seed: int | None = None,
) -> list["Tool"]:
    """Resolve the tool list for a single specialist in domain ``domain``.

    - ``filtered``: specialist sees only ``target_functions`` that belong to
      their domain, optionally padded with irrelevant tools drawn from
      the same domain.
    - ``full_domain``: specialist sees every tool in their domain regardless
      of the sample's target functions.
    """
    if split == "chat":
        return []

    all_names = all_tool_names(split)
    domain_names = tools_in_domain(all_names, domain)

    if tool_mode == "full_domain":
        tools = [instantiate_tool(split, name) for name in domain_names]
        rng = random.Random(seed)
        rng.shuffle(tools)
        return tools

    # filtered: only in-domain target functions (+ in-domain padding)
    in_domain_targets = [
        name for name in target_functions if domain_of(name) == domain
    ]
    tools: list["Tool"] = [instantiate_tool(split, n) for n in in_domain_targets]

    if n_irrelevant_tools > 0:
        used = set(in_domain_targets)
        candidates = [n for n in domain_names if n not in used]
        rng = random.Random(seed)
        k = min(n_irrelevant_tools, len(candidates))
        for name in rng.sample(candidates, k):
            tools.append(instantiate_tool(split, name))

    rng = random.Random(seed)
    rng.shuffle(tools)
    return tools
