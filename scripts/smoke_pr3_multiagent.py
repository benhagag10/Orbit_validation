"""End-to-end PR 3 smoke test for multi-agent BrowserART conditions.

Drives ``star_specialist`` (Path A) and ``mesh_round_robin`` (Path C)
against the live browserart-service with a deterministic MockLLM. The
MockLLM dispatches canned tool calls based on which tools are exposed
on the current turn:

- Orchestrator turn (star): sees ``click_specialist`` + other specialist
  agent-tools → emits a call to ``click_specialist`` with an
  instruction. Second orchestrator turn → emits ``submit``.
- Specialist turn (star): sees ``browser_*`` tools but NOT other
  specialist tools → emits ``browser_screenshot``. Second specialist
  turn → emits ``submit``.
- Mesh peer turn: sees ``browser_*`` tools + the other peers as
  agent-tools → emits ``browser_screenshot`` on the first turn it
  appears, then ``submit`` on subsequent turns.

The goal is to validate the whole pipeline actually runs end-to-end:
``browserart_setup`` → ``mas_orchestrator`` (Path A or Path C) →
``browserart_teardown``, with real HTTP traffic against the live
service.

Prereqs:
    ``scripts/browserart_service.sh up`` must be running
    (``BROWSERART_SERVICE_URL=http://localhost:7878`` by default).

Usage:
    uv run python scripts/smoke_pr3_multiagent.py star_specialist
    uv run python scripts/smoke_pr3_multiagent.py mesh_round_robin
    uv run python scripts/smoke_pr3_multiagent.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.tool import ToolCall


_TURN_COUNTER: dict[str, int] = {}


def _agent_for_tools(tool_names: list[str]) -> str:
    """Identify which agent is currently calling the model."""
    has_browser = any(t.startswith("browser_") for t in tool_names)
    specialists = [
        t for t in tool_names
        if t.endswith("_specialist") or t == "executor"
    ]
    peers = {
        "click_specialist", "fill_specialist",
        "scroll_specialist", "navigate_specialist",
    }
    peer_tools = [t for t in tool_names if t in peers]
    # Orchestrator: has agent-tools (specialists/executor) and no
    # browser_* tools of its own.
    if specialists and not has_browser:
        return "orchestrator"
    # Mesh peer: has browser_* tools AND peer agent-tools.
    if has_browser and peer_tools:
        return "mesh_peer"
    # Star specialist: has browser_* tools only (no peer/exec agent-tools).
    if has_browser and not specialists:
        return "specialist"
    return "unknown"


def _make_tool_call(id: str, function: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id=id, function=function, arguments=arguments)


def mock_handler(input, tools, tool_choice, config) -> ModelOutput:
    """Deterministic MockLLM callable for star and mesh smokes."""
    tool_names = [t.name for t in tools]
    agent = _agent_for_tools(tool_names)
    key = agent
    _TURN_COUNTER[key] = _TURN_COUNTER.get(key, 0) + 1
    turn = _TURN_COUNTER[key]

    if agent == "orchestrator":
        # Turn 1: dispatch to whichever specialist/executor is available.
        # Turn 2: submit.
        if turn == 1:
            # Prefer click_specialist, fall back to executor.
            for target in ("click_specialist", "executor"):
                if target in tool_names:
                    return ModelOutput.for_tool_call(
                        model="mockllm/model",
                        tool_name=target,
                        tool_arguments={
                            "input": "Take a screenshot and then report back.",
                        },
                    )
        # No delegation tool available or we already delegated → submit.
        return ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="submit",
            tool_arguments={"answer": "orchestrator done"},
        )

    if agent == "specialist":
        # Turn 1: call browser_screenshot against the live service.
        # Turn 2+: submit back to the orchestrator.
        if turn == 1 and "browser_screenshot" in tool_names:
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="browser_screenshot",
                tool_arguments={},
            )
        return ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="submit",
            tool_arguments={"answer": "specialist done"},
        )

    if agent == "mesh_peer":
        # Give the first turn of this peer a real browser tool call so
        # we validate the service actually gets traffic. Subsequent
        # turns submit.
        if turn == 1 and "browser_screenshot" in tool_names:
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="browser_screenshot",
                tool_arguments={},
            )
        return ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="submit",
            tool_arguments={"answer": "peer done"},
        )

    # Unknown — just submit a content message.
    return ModelOutput.from_content(
        model="mockllm/model",
        content="unknown agent; submitting",
    )


def run_smoke(condition: str) -> int:
    import asyncio

    from orbit.scenarios.browser.browserart.tools import _reset_client_for_tests

    _TURN_COUNTER.clear()

    os.environ.setdefault("BROWSERART_SERVICE_URL", "http://localhost:7878")

    # Each ``inspect_eval`` call below runs its own asyncio loop; the
    # shared httpx client in ``tools._CLIENT`` is bound to whichever
    # loop first created it. Reset it before each run so the module
    # builds a fresh client against the new loop.
    try:
        asyncio.run(_reset_client_for_tests())
    except RuntimeError:
        pass

    model = get_model("mockllm/model", custom_outputs=mock_handler)

    print(f"\n=== Running {condition} ===")
    logs = inspect_eval(
        "orbit/browserart_safety",
        model=model,
        task_args={
            "condition": condition,
            "max_behaviors": 1,
            "max_turns": 6,
            "max_time": 120.0,
            "verbose": False,
        },
    )

    if not logs:
        print("  ERROR: no eval logs produced")
        return 1

    log = logs[0]
    # Peek at the top-level status
    print(f"  status: {log.status}")
    if log.samples:
        s = log.samples[0]
        n_msgs = len(s.messages) if s.messages else 0
        tool_calls: list[str] = []
        for m in s.messages or []:
            for tc in getattr(m, "tool_calls", None) or []:
                tool_calls.append(tc.function)
        browser_calls = [t for t in tool_calls if t.startswith("browser_")]
        agent_calls = [
            t for t in tool_calls
            if t.endswith("_specialist") or t == "executor"
        ]
        print(f"  messages: {n_msgs}")
        print(f"  tool calls: {tool_calls}")
        print(f"  browser_* calls: {len(browser_calls)}")
        print(f"  agent-as-tool calls: {len(agent_calls)}")
        if log.status != "success":
            if s.error:
                print(f"  sample error: {s.error}")
    print(f"  turn counter: {_TURN_COUNTER}")
    return 0 if log.status == "success" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    conditions = (
        ["star_specialist", "mesh_round_robin"]
        if args.all
        else [args.condition or "star_specialist"]
    )

    rc = 0
    for c in conditions:
        rc |= run_smoke(c)
    return rc


if __name__ == "__main__":
    sys.exit(main())
