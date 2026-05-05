"""
BrowserART dataset loading and filtering.

Loads BrowserART behaviors from bundled JSON data and applies
category/count/seed filters.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from orbit.scenarios.browser.browserart.configs import (
    BrowserARTBehavior,
    BrowserARTScenarioConfig,
)

logger = logging.getLogger(__name__)

DATASET_FILES = {
    "hbb": "hbb.json",
    "hbb_hr": "hbb_hr.json",
    "hbb_benign": "hbb_benign.json",
    "hbb_extension": "hbb_extension.json",
}


def _resolve_data_path(config: BrowserARTScenarioConfig) -> Path:
    """Resolve the path to the dataset JSON file.

    Resolution order:
    1. config.data_path (explicit override)
    2. Bundled package data: orbit/scenarios/browserart/data/
    """
    filename = DATASET_FILES.get(config.dataset, "hbb.json")

    # 1. Explicit path
    if config.data_path:
        path = Path(config.data_path)
        if path.is_file():
            return path
        candidate = path / filename
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"BrowserART data not found at {config.data_path}")

    # 2. Bundled package data
    try:
        data_pkg = importlib.resources.files(
            "orbit.scenarios.browser.browserart.data"
        )
        data_file = data_pkg / filename
        data_file.read_text(encoding="utf-8")
        module_dir = Path(__file__).parent / "data" / filename
        if module_dir.exists():
            return module_dir
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    module_dir = Path(__file__).parent / "data" / filename
    if module_dir.exists():
        return module_dir

    # hbb_extension.json is Orbit-original (Apache-2.0) and is always
    # bundled. The other BrowserART datasets are upstream-derived
    # (CC BY-NC-ND 4.0 per upstream LICENSE file) and are not
    # redistributed — users must fetch them with the helper script.
    if filename == "hbb_extension.json":
        raise FileNotFoundError(
            f"BrowserART dataset '{filename}' not found at {module_dir}. "
            "This is Orbit-original content that should be bundled with "
            "the package; re-install Orbit to recover."
        )
    raise FileNotFoundError(
        f"BrowserART dataset '{filename}' is not vendored in Orbit "
        f"(upstream scaleapi/browser-art is CC BY-NC-ND 4.0 and is not "
        f"redistributable). Fetch it with:\n"
        f"    uv run python scripts/fetch_browserart_data.py\n"
        f"or pass data_path=<dir> to a directory that contains "
        f"{filename}. See orbit/scenarios/browserart/data/README.md."
    )


def load_browserart_behaviors(
    config: BrowserARTScenarioConfig | None = None,
) -> list[BrowserARTBehavior]:
    """Load BrowserART behaviors from JSON.

    Args:
        config: Scenario configuration. Uses defaults if None.

    Returns:
        List of BrowserARTBehavior models.
    """
    if config is None:
        config = BrowserARTScenarioConfig()

    path = _resolve_data_path(config)
    logger.info("Loading BrowserART dataset from %s", path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data.get("behaviors", [])

    # Merge Orbit-original custom benign tasks (restored from pre-#143).
    if config.dataset == "hbb_benign":
        custom_path = path.parent / "hbb_benign_custom.json"
        if custom_path.exists():
            with open(custom_path, encoding="utf-8") as f:
                custom = json.load(f)
            logger.info("Merging %d custom benign tasks from %s", len(custom), custom_path)
            items.extend(custom if isinstance(custom, list) else [])
    behaviors: list[BrowserARTBehavior] = []

    for item in items:
        # Support both BrowserART format (id/behavior/website) and
        # WebArena format (task_id/intent/start_url) used by benign tasks.
        known_keys = {
            "id", "behavior", "text_behavior", "website",
            "semantic_category", "evaluation_category", "source",
            "context", "text_behavior_with_context",
            "multi_step", "min_steps",
            # WebArena-format keys (benign tasks)
            "task_id", "intent", "start_url", "sites",
            "require_login", "geolocation", "instantiation_dict",
            "require_reset", "intent_template_id",
        }
        extra = {k: v for k, v in item.items() if k not in known_keys}

        # Resolve fields: prefer BrowserART keys, fall back to WebArena keys.
        behavior_id = item.get("id", item.get("task_id", len(behaviors)))
        behavior = item.get("behavior", "") or item.get("intent", "")
        website = item.get("website", "") or item.get("start_url", "")

        behaviors.append(
            BrowserARTBehavior(
                behavior_id=behavior_id,
                behavior=behavior,
                text_behavior=item.get("text_behavior", "") or behavior,
                website=website,
                semantic_category=item.get("semantic_category", "benign/utility"),
                evaluation_category=item.get("evaluation_category", "closed-ended"),
                source=item.get("source", "BrowserART"),
                context=item.get("context"),
                text_behavior_with_context=item.get(
                    "text_behavior_with_context", ""
                ) or behavior,
                multi_step=item.get("multi_step", False),
                min_steps=item.get("min_steps"),
                extra=extra,
            )
        )

    logger.info("Loaded %d BrowserART behaviors", len(behaviors))
    return behaviors


def filter_behaviors(
    behaviors: list[BrowserARTBehavior],
    config: BrowserARTScenarioConfig,
) -> list[BrowserARTBehavior]:
    """Apply category, source, count, and seed filters.

    Args:
        behaviors: Full list of loaded behaviors.
        config: Scenario config with filter parameters.

    Returns:
        Filtered list of behaviors.
    """
    result = list(behaviors)

    if config.task_ids:
        allowed = set(config.task_ids)
        result = [b for b in result if b.behavior_id in allowed]

    # Exclude unreliable benign tasks unless specific task_ids requested.
    # Two categories:
    # 1. External websites (google.com) — DNS, TLS, rate limiting, CAPTCHAs
    # 2. Broken local sites — pages lack required UI elements (no comment
    #    box, no upvote button, 404s) making the task impossible
    _BROKEN_BENIGN_IDS = {
        324,  # local:github_pr — page lacks code editing interface
        325,  # local:github_pr — page lacks comment submission UI
        326,  # local:github_pr — page lacks docstring editing interface
        327,  # local:github_pr — page lacks code review comment interface
        347,  # local:forum_posts — page has no upvote functionality
        348,  # local:forum_posts — page has no comment functionality
        349,  # local:contact_form — form not found on page
        351,  # local:hospital_db_system — no appointment scheduling UI
        352,  # local:gitlab_issue — login page returns 404
    }
    if config.dataset in ("hbb_benign",) and not config.task_ids:
        before = len(result)
        result = [
            b for b in result
            if not b.website.startswith("http://")
            and not b.website.startswith("https://")
            and b.behavior_id not in _BROKEN_BENIGN_IDS
        ]
        if len(result) < before:
            logger.info(
                "Excluded %d benign behaviors (external sites + broken pages)",
                before - len(result),
            )

    if config.categories:
        result = [
            b for b in result if b.semantic_category in config.categories
        ]

    if config.sources:
        result = [
            b
            for b in result
            if any(s.lower() in b.source.lower() for s in config.sources)
        ]

    if config.max_behaviors and len(result) > config.max_behaviors:
        if config.seed is not None:
            rng = random.Random(config.seed)
            result = rng.sample(result, config.max_behaviors)
        else:
            result = result[: config.max_behaviors]

    logger.info(
        "Filtered to %d behaviors (categories=%s, sources=%s, max=%s)",
        len(result),
        config.categories,
        config.sources,
        config.max_behaviors,
    )
    return result


def get_behavior_statistics(
    behaviors: list[BrowserARTBehavior],
) -> dict:
    """Compute summary statistics about loaded behaviors.

    Returns:
        Dict with total, per-category counts, per-source counts,
        and evaluation category breakdown.
    """
    stats: dict = {
        "total": len(behaviors),
        "by_category": defaultdict(int),
        "by_source": defaultdict(int),
        "by_evaluation": defaultdict(int),
    }

    for b in behaviors:
        stats["by_category"][b.semantic_category] += 1
        stats["by_source"][b.source] += 1
        stats["by_evaluation"][b.evaluation_category] += 1

    stats["by_category"] = dict(
        sorted(stats["by_category"].items(), key=lambda x: -x[1])
    )
    stats["by_source"] = dict(
        sorted(stats["by_source"].items(), key=lambda x: -x[1])
    )
    stats["by_evaluation"] = dict(
        sorted(stats["by_evaluation"].items(), key=lambda x: -x[1])
    )

    return stats
