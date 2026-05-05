"""
Model filter infrastructure -- wraps ModelAPI to intercept generate() calls.

Provides a general-purpose filter chain that any defense can use to wrap
agent models. Filters run at the ModelAPI level (inside Inspect's retry/
caching/concurrency machinery) and support runtime adaptivity via
per-call activation checks.

Interaction with Inspect:
    - FilteredModelAPI subclasses ModelAPI, wrapping an inner API's generate()
    - wrap_model() constructs a new Model with the filtered API
    - SystemPromptFilter prepends ChatMessageSystem to input
    - OutputContentFilter regex-scans ModelOutput completion text

Design:
    - Onion model: input filters run low-priority-first, output filters
      run high-priority-first (reverse order)
    - Each filter has should_activate(ctx) checked every generate() call
    - fail_mode controls error behavior: "open" skips failed filters,
      "closed" re-raises exceptions to block the request
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
)

if TYPE_CHECKING:
    from inspect_ai.tool import ToolInfo

    ToolChoice = Any  # Inspect's ToolChoice type

logger = logging.getLogger(__name__)


@dataclass
class FilterContext:
    """Mutable per-agent context passed to filters on every generate() call.

    The orchestrator updates ``turn`` and ``phase`` each turn; filters
    read these to decide whether to activate.
    """

    agent_name: str
    """Name of the agent this context belongs to."""

    turn: int = 0
    """Current experiment turn (updated by orchestrator)."""

    phase: str = "pre_deployment"
    """Current experiment phase (updated by orchestrator)."""

    state: dict[str, Any] = field(default_factory=dict)
    """Arbitrary per-agent state that filters can read/write."""


class ModelFilter(ABC):
    """Base class for model filters.

    Subclasses override ``on_input`` and/or ``on_output`` to intercept
    generate() calls. Both have default pass-through implementations
    so filters only need to implement what they care about.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""
        ...

    @property
    @abstractmethod
    def priority(self) -> int:
        """Ordering priority. Lower runs first on input, last on output."""
        ...

    def should_activate(self, ctx: FilterContext) -> bool:
        """Check whether this filter should run for the current call.

        Returns True by default (always active). Override for conditional
        activation based on turn, phase, agent name, etc.
        """
        return True

    async def on_input(
        self,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
        ctx: FilterContext,
    ) -> tuple[list[ChatMessage], list[ToolInfo], ToolChoice, GenerateConfig]:
        """Filter/transform inputs before they reach the model.

        Default implementation passes through unchanged.
        """
        return messages, tools, tool_choice, config

    async def on_output(
        self,
        output: ModelOutput,
        ctx: FilterContext,
    ) -> ModelOutput:
        """Filter/transform output after the model returns.

        Default implementation passes through unchanged.
        """
        return output


class FilteredModelAPI(ModelAPI):
    """ModelAPI wrapper that applies a chain of ModelFilter instances.

    Input filters run in priority order (low first). Output filters
    run in reverse priority order (high first).

    Error behavior is controlled by ``fail_mode``:
    - "open": filter exceptions are logged and skipped
    - "closed": filter exceptions are re-raised, blocking the request (default)
    """

    def __init__(
        self,
        inner: ModelAPI,
        filters: list[ModelFilter],
        context: FilterContext,
        fail_mode: Literal["open", "closed"] = "closed",
    ) -> None:
        # Intentionally skip ModelAPI.__init__ to avoid re-applying
        # API key logic -- we delegate everything to inner.
        self._inner = inner
        self._filters = sorted(filters, key=lambda f: f.priority)
        self._context = context
        self._fail_mode = fail_mode

    # Delegate all non-overridden attributes to inner API
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, Any]:
        """Run input filters, call inner generate, run output filters."""
        ctx = self._context

        # Input filters: low priority first
        msgs, tls, tc, cfg = input, tools, tool_choice, config
        for filt in self._filters:
            try:
                if filt.should_activate(ctx):
                    msgs, tls, tc, cfg = await filt.on_input(
                        msgs, tls, tc, cfg, ctx
                    )
            except Exception:
                if self._fail_mode == "closed":
                    logger.error(
                        "Filter %s.on_input failed (fail_mode=closed), "
                        "blocking request",
                        filt.name,
                    )
                    raise
                logger.exception(
                    "Filter %s.on_input failed, skipping", filt.name
                )

        # Call inner API
        result = await self._inner.generate(msgs, tls, tc, cfg)

        # Unpack tuple return format
        model_call = None
        if isinstance(result, tuple):
            raw_output, model_call = result
            if isinstance(raw_output, Exception):
                # Don't filter exceptions, just re-pack
                return result
            output = raw_output
        else:
            output = result

        # Output filters: high priority first (reverse order)
        for filt in reversed(self._filters):
            try:
                if filt.should_activate(ctx):
                    output = await filt.on_output(output, ctx)
            except Exception:
                if self._fail_mode == "closed":
                    logger.error(
                        "Filter %s.on_output failed (fail_mode=closed), "
                        "blocking request",
                        filt.name,
                    )
                    raise
                logger.exception(
                    "Filter %s.on_output failed, skipping", filt.name
                )

        # Re-pack if original was a tuple
        if model_call is not None:
            return output, model_call
        return output


def wrap_model(
    model: Model,
    filters: list[ModelFilter],
    context: FilterContext,
    fail_mode: Literal["open", "closed"] = "closed",
) -> Model:
    """Wrap a Model's API with a filter chain.

    If *filters* is empty, returns the original model unchanged (zero overhead).

    Args:
        model: The Inspect Model to wrap.
        filters: ModelFilter instances to apply.
        context: Per-agent FilterContext for activation checks.
        fail_mode: Error behavior for filters -- "closed" (default) re-raises
            to block the request, "open" skips failed filters.

    Returns:
        A new Model with FilteredModelAPI, or the original if no filters.
    """
    if not filters:
        return model
    filtered_api = FilteredModelAPI(model.api, filters, context, fail_mode=fail_mode)
    return Model(filtered_api, model.config, model.model_args)


# -- Built-in filters ────────────────────────────────────────────────────


class SystemPromptFilter(ModelFilter):
    """Prepends a system message to model input.

    Used by ModelDefense to inject safety instructions. Supports
    ``target_agents`` filtering to only apply to specific agents.
    """

    def __init__(
        self,
        prompt: str,
        target_agents: list[str] | None = None,
    ) -> None:
        self._prompt = prompt
        self._target_agents = target_agents

    @property
    def name(self) -> str:
        return "system_prompt_filter"

    @property
    def priority(self) -> int:
        return 10

    def should_activate(self, ctx: FilterContext) -> bool:
        if self._target_agents:
            return ctx.agent_name in self._target_agents
        return True

    async def on_input(
        self,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
        ctx: FilterContext,
    ) -> tuple[list[ChatMessage], list[ToolInfo], ToolChoice, GenerateConfig]:
        system_msg = ChatMessageSystem(content=self._prompt)
        return [system_msg] + list(messages), tools, tool_choice, config


class OutputContentFilter(ModelFilter):
    """Regex-scans output completion text and blocks on match.

    When a pattern matches, replaces the completion with a refusal message.
    Supports ``target_agents`` filtering.
    """

    DEFAULT_REFUSAL = "I cannot comply with that request."

    def __init__(
        self,
        blocked_patterns: list[str],
        target_agents: list[str] | None = None,
        refusal_message: str | None = None,
    ) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in blocked_patterns]
        self._target_agents = target_agents
        self._refusal = refusal_message or self.DEFAULT_REFUSAL

    @property
    def name(self) -> str:
        return "output_content_filter"

    @property
    def priority(self) -> int:
        return 90

    def should_activate(self, ctx: FilterContext) -> bool:
        if self._target_agents:
            return ctx.agent_name in self._target_agents
        return True

    async def on_output(
        self,
        output: ModelOutput,
        ctx: FilterContext,
    ) -> ModelOutput:
        text = output.completion
        if not text:
            return output

        for pattern in self._patterns:
            if pattern.search(text):
                logger.info(
                    "OutputContentFilter blocked output from agent '%s' "
                    "(matched pattern: %s)",
                    ctx.agent_name,
                    pattern.pattern,
                )
                # Record the block in context state so the orchestrator
                # can log it to DefenseLog at the appropriate scope
                ctx.state.setdefault("content_blocks", []).append({
                    "agent": ctx.agent_name,
                    "turn": ctx.turn,
                    "pattern": pattern.pattern,
                })

                # Replace completion with refusal
                return output.model_copy(
                    update={"completion": self._refusal}
                )

        return output
