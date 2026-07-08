"""
Shared LLM-as-a-judge utilities for NeurIPS evaluator scripts.

Provides a reusable LLM judge call wrapper, trajectory condensation,
and scorer claim extraction. All three NeurIPS check scripts import
from this module.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from _browserart_log_utils import (
    extract_browserart_metadata,
    extract_score_data,
    message_text,
)

logger = logging.getLogger(__name__)


def call_llm_judge(
    prompt: str,
    model: str = "openai/gpt-4.1",
    max_tokens: int = 1000,
) -> dict | None:
    """Call an LLM judge and return parsed JSON response, or None on failure."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed — skipping LLM judge call")
        return None

    model_name = model.split("/")[-1] if "/" in model else model

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        resp_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("LLM judge API call failed: %s", e)
        return None

    try:
        json_match = re.search(r"\{[\s\S]*\}", resp_text)
        if not json_match:
            logger.warning("No JSON found in LLM judge response")
            return None
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("LLM judge response JSON is malformed")
        return None


def condense_trajectory(
    messages: list[dict], max_messages: int = 20
) -> str:
    """Build a condensed trajectory string for LLM review."""
    parts: list[str] = []

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    if sys_msgs:
        text = message_text(sys_msgs[0])
        parts.append(f"[SYSTEM] {text[:300]}")

    relevant = [m for m in messages if m.get("role") in ("assistant", "user")]
    for msg in relevant[-max_messages:]:
        role = msg.get("role", "?").upper()
        text = message_text(msg)[:500]
        parts.append(f"[{role}] {text}")

    return "\n\n".join(parts)


def extract_scorer_claims(sample: dict) -> dict[str, Any]:
    """Extract scorer output claims from a BrowserART sample.

    Returns a dict with keys: attack_success, propagation_depth,
    execution_reach, agent_scores, stage, behavior, is_benign.
    """
    ba_value, ba_meta = extract_score_data(sample)
    ba_fields = extract_browserart_metadata(sample)

    claims: dict[str, Any] = {
        "behavior": ba_fields.get("browserart_behavior", "unknown"),
        "is_benign": bool(ba_meta.get("is_benign", False)),
    }

    if isinstance(ba_value, dict):
        claims["attack_success"] = ba_value.get("attack_success", None)
        claims["propagation_depth"] = ba_value.get("propagation_depth", None)
        claims["execution_reach"] = ba_value.get("execution_reach", None)
    else:
        claims["attack_success"] = None
        claims["propagation_depth"] = None
        claims["execution_reach"] = None

    claims["agent_scores"] = ba_meta.get("agent_scores", {})

    stage = "unknown"
    if ba_meta.get("planning_refusal"):
        stage = "planning_refusal"
    elif ba_meta.get("execution_refusal"):
        stage = "execution_refusal"
    elif ba_meta.get("harmful_actions"):
        stage = "harmful_actions"
    elif ba_meta.get("harmful_task"):
        stage = "harmful_task"
    claims["stage"] = stage

    return claims
