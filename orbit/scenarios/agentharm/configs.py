"""Frozen config models for the AgentHarm scenario.

Based on: https://github.com/UKGovernmentBEIS/inspect_evals
(``src/inspect_evals/agentharm``) and
https://huggingface.co/datasets/ai-safety-institute/AgentHarm
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentHarmSplit = Literal["harmful", "benign", "chat"]
AgentHarmPartition = Literal["test_public", "validation"]
AgentHarmToolMode = Literal["filtered", "full_domain"]

AGENTHARM_CATEGORIES = (
    "Copyright",
    "Cybercrime",
    "Disinformation",
    "Drugs",
    "Fraud",
    "Harassment",
    "Hate",
    "Sexual",
)


class AgentHarmBehavior(BaseModel, frozen=True):
    """A single AgentHarm behavior loaded from the pinned dataset."""

    id: str
    id_original: str
    name: str
    category: str
    prompt: str
    target_functions: tuple[str, ...]
    grading_function: str | None = None
    detailed_prompt: bool | None = None
    hint_included: bool | None = None


class AgentHarmScenarioConfig(BaseModel, frozen=True):
    """AgentHarm scenario configuration."""

    data_path: str | None = None
    """Override the bundled ``data/`` directory path."""

    split: AgentHarmSplit = "harmful"
    """Which subset to load: ``harmful``, ``benign``, or ``chat``."""

    partition: AgentHarmPartition = "test_public"
    """Which partition within the split: ``test_public`` (default) or
    ``validation``."""

    categories: tuple[str, ...] | None = None
    """Filter to a subset of the 8 harm categories."""

    behavior_ids: tuple[str, ...] | None = None
    """Filter to specific behavior IDs."""

    detailed_prompt: bool | None = None
    """If set, keep only behaviors with ``detailed_prompt`` matching."""

    hint_included: bool | None = None
    """If set, keep only behaviors with ``hint_included`` matching."""

    max_samples: int | None = None
    """Limit the number of samples after filtering."""

    seed: int = 0
    """Seed for deterministic subsampling and irrelevant-tool selection."""

    judge_model: str = "openai/gpt-4.1"
    """Model used for both the refusal judge and the grader semantic judge."""

    n_irrelevant_tools: int = 0
    """Number of irrelevant tools to pad each sample's tool list with
    (upstream default: 0)."""

    tool_mode: AgentHarmToolMode = "filtered"
    """How specialist-condition agents see tools. ``filtered`` = each
    specialist sees only the target_functions in their domain (+ padding
    from their domain). ``full_domain`` = each specialist sees every tool
    in their domain regardless of the sample's target_functions."""

    prompt_technique: Literal["standard", "cot", "react", "refusal"] = "standard"
    """Upstream prompt-technique suffix appended to the system prompt."""

    system_prompt_variant: Literal["default", "llama-3.1"] = "default"
    """Upstream system prompt variant."""

    max_turns: int = 10
    """Maximum agent turns per sample."""

    max_time_seconds: float = 300.0
    """Maximum wall-clock time per sample."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Scenario-level extension fields."""
