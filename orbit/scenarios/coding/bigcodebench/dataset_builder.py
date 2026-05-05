"""Dataset builder for BigCodeBench tasks.

Loads the ``bigcode/bigcodebench`` dataset from HuggingFace via the
shared loader that ``inspect_evals.bigcodebench`` already uses, and
converts each record into a frozen :class:`BigCodeBenchTask`.

Only the Complete split is supported — it's the direct function-signature
→ completion format that mirrors RedCode-Gen.
"""

from __future__ import annotations

import logging
import random

from orbit.scenarios.coding.bigcodebench.configs import (
    BigCodeBenchScenarioConfig,
    BigCodeBenchTask,
)

logger = logging.getLogger(__name__)


def load_bigcodebench_tasks(
    config: BigCodeBenchScenarioConfig | None = None,
) -> list[BigCodeBenchTask]:
    """Load BigCodeBench tasks from the HuggingFace dataset.

    Args:
        config: Scenario configuration; uses defaults if ``None``.

    Returns:
        List of :class:`BigCodeBenchTask` models, one per dataset record.
    """
    if config is None:
        config = BigCodeBenchScenarioConfig()

    from inspect_evals.utils.huggingface import load_dataset

    logger.info("Loading BigCodeBench dataset (version=%s)", config.version)
    records = load_dataset("bigcode/bigcodebench", split=config.version)

    tasks: list[BigCodeBenchTask] = []
    for record in records:
        task_id_str = record["task_id"]
        # task_id is "BigCodeBench/N" — pull the integer N for sample ids.
        try:
            task_id_int = int(task_id_str.split("/")[-1])
        except (ValueError, IndexError):
            logger.warning("Could not parse task id %r", task_id_str)
            continue

        tasks.append(
            BigCodeBenchTask(
                task_id=task_id_int,
                task_id_str=task_id_str,
                complete_prompt=record["complete_prompt"],
                test=record["test"],
                libs=record.get("libs", ""),
                canonical_solution=record.get("canonical_solution", ""),
            )
        )

    logger.info("Loaded %d BigCodeBench tasks", len(tasks))
    return tasks


def filter_tasks(
    tasks: list[BigCodeBenchTask],
    config: BigCodeBenchScenarioConfig,
) -> list[BigCodeBenchTask]:
    """Apply subset / max_tasks filters from the scenario config."""
    filtered = tasks

    if config.subset is not None:
        allowed = set(config.subset)
        filtered = [t for t in filtered if t.task_id in allowed]
        logger.info("Filtered to subset (%d ids): %d tasks",
                    len(allowed), len(filtered))

    if config.max_tasks is not None and len(filtered) > config.max_tasks:
        rng = random.Random(config.seed)
        filtered = rng.sample(filtered, config.max_tasks)
        logger.info("Sampled %d tasks (seed=%s)",
                    config.max_tasks, config.seed)

    return filtered
