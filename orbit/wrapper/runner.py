"""
Experiment runner.

Bridges YAML configs to Inspect's eval() and eval_set() APIs.
Handles single experiment runs and multi-config benchmark suites.

Interaction with Inspect:
    - Calls eval() for single experiment execution
    - Calls eval_set() for benchmark suites (multiple configs/models)
    - Passes Task objects built from ExperimentConfig
    - Returns Inspect's EvalLog results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai import eval, eval_set
from inspect_ai.log import EvalLog

from inspect_ai import Task

from orbit.baselines.baselines import apply_baseline
from orbit.configs.baseline import BaselineMode
from orbit.configs.experiment import ExperimentConfig
from orbit.tasks.security_benchmark import security_benchmark
from orbit.validation.validators import ConfigValidator
from orbit.wrapper.yaml_loader import (
    apply_config_overrides,
    load_experiment_config,
    load_experiment_configs,
)


def run_experiment(
    config_path: str | Path,
    model: str | list[str] | None = None,
    config_overrides: dict[str, str] | None = None,
    **eval_kwargs: Any,
) -> list[EvalLog]:
    """Run a single experiment from a YAML config file.

    Args:
        config_path: Path to the YAML experiment configuration.
        model: Model(s) to use. Overrides config.models if provided.
        config_overrides: Dotted key=value overrides applied to the config.
        **eval_kwargs: Additional kwargs passed to Inspect's eval().

    Returns:
        List of EvalLog results.
    """
    config = load_experiment_config(config_path)
    if config_overrides:
        config = apply_config_overrides(config, config_overrides)

    task = _build_task(config)

    # Determine model list
    models = _resolve_models(model, config.models)

    return eval(task, model=models, **eval_kwargs)


def run_benchmark_suite(
    config_paths: list[str | Path],
    models: list[str] | None = None,
    baselines: bool = True,
    log_dir: str = "./logs",
    config_overrides: dict[str, str] | None = None,
    **eval_kwargs: Any,
) -> tuple[bool, list[EvalLog]]:
    """Run a suite of experiments across configs and models.

    Produces a full benchmark matrix: configs x models x baselines.
    Uses Inspect's eval_set() for efficient batch execution.

    Args:
        config_paths: List of YAML experiment config paths.
        models: Models to evaluate. Overrides per-config models.
        baselines: If True, auto-generate baseline variants for each config.
        log_dir: Directory for eval logs. Required by eval_set().
        config_overrides: Dotted key=value overrides applied to each config.
        **eval_kwargs: Additional kwargs passed to Inspect's eval_set().

    Returns:
        Tuple of (success, list[EvalLog]) from eval_set().
    """
    configs = load_experiment_configs(config_paths)

    if config_overrides:
        configs = [apply_config_overrides(c, config_overrides) for c in configs]

    # Build tasks from all config variants
    tasks = []
    for config in configs:
        if baselines:
            for mode in BaselineMode:
                variant_config = apply_baseline(config, mode)
                tasks.append(_build_task(variant_config))
        else:
            tasks.append(_build_task(config))

    # Determine model list
    all_models = models
    if not all_models:
        # Collect unique models from all configs
        seen: set[str] = set()
        all_models = []
        for config in configs:
            for m in config.models:
                if m not in seen:
                    seen.add(m)
                    all_models.append(m)

    return eval_set(tasks, log_dir, model=all_models, **eval_kwargs)


def dry_run(
    config_path: str | Path,
    model: str | list[str] | None = None,
    config_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load config, resolve models, validate -- without calling eval().

    Args:
        config_path: Path to the YAML experiment configuration.
        model: Model(s) to use. Overrides config.models if provided.
        config_overrides: Dotted key=value overrides applied to the config.

    Returns:
        Dict with config summary and validation results.
    """
    config = load_experiment_config(config_path)
    if config_overrides:
        config = apply_config_overrides(config, config_overrides)

    models = _resolve_models(model, config.models)
    errors = ConfigValidator.validate(config)

    return {
        "name": config.name,
        "description": config.description,
        "models": models if isinstance(models, list) else [models],
        "agents": [a.name for a in config.setup.agents],
        "attacks": [a.name for a in config.attacks],
        "defenses": [d.name for d in config.defenses],
        "baseline_mode": config.baseline_mode.value,
        "epochs": config.epochs,
        "max_turns": config.max_turns,
        "scheduler_max_turns": config.scheduler.max_turns,
        "validation_errors": errors,
        "valid": len(errors) == 0,
    }


def _build_task(config: ExperimentConfig) -> Task:
    """Build an Inspect Task from an ExperimentConfig.

    Dispatches to scenario-specific task factories when the scenario
    name is recognised (``browserart``, ``swe_bench``, ``osworld``),
    otherwise falls back to the generic ``security_benchmark`` task.
    """
    scenario_name = config.scenario.name if config.scenario else ""

    if scenario_name == "browserart":
        from orbit.scenarios.browser.browserart.task import browserart_safety

        meta = config.metadata or {}
        return browserart_safety(
            category=",".join(meta["browserart_categories"])
            if meta.get("browserart_categories")
            else None,
            max_behaviors=meta.get("browserart_max_behaviors"),
            seed=meta.get("browserart_seed"),
            classifier_model=meta.get(
                "browserart_classifier_model", "openai/gpt-4.1"
            ),
            data_path=meta.get("browserart_data_path"),
            max_turns=config.scheduler.max_turns,
            max_time=config.scheduler.max_time_seconds,
            topology=meta.get("browserart_topology", "default"),
            headless=meta.get("browserart_headless", True),
            record_video_dir=meta.get("browserart_record_video_dir"),
            condition=meta.get("browserart_condition"),
            dataset=meta.get("browserart_dataset", "hbb"),
            task_ids=",".join(str(x) for x in meta["browserart_task_ids"])
            if isinstance(meta.get("browserart_task_ids"), list)
            else meta.get("browserart_task_ids"),
        )

    if scenario_name == "swe_bench":
        from orbit.scenarios.coding.swe_bench.task import swe_bench_multi_issue

        meta = config.metadata or {}

        # Use the YAML's setup as topology template when agents are defined,
        # so per-agent model/temperature overrides are preserved.
        topology_arg = meta.get("swe_bench_topology", "default")
        if config.setup.agents and topology_arg == "default":
            topology_arg = config.setup

        return swe_bench_multi_issue(
            num_issues=meta.get("swe_bench_num_issues", 2),
            seed=meta.get("swe_bench_seed"),
            repos=",".join(meta["swe_bench_repos"])
            if meta.get("swe_bench_repos")
            else None,
            mode=meta.get("swe_bench_mode", "round_robin"),
            topology=topology_arg,
            attack_preset=meta.get("swe_bench_attack_preset"),
            defense_preset=meta.get("swe_bench_defense_preset"),
            max_turns=config.scheduler.max_turns if config.scheduler else 30,
            max_time=config.scheduler.max_time_seconds
            if config.scheduler
            else 900.0,
            max_groups=meta.get("swe_bench_max_groups"),
        )

    if scenario_name == "osworld":
        from orbit.scenarios.desktop.osworld.task import osworld_safety

        meta = config.metadata or {}
        return osworld_safety(
            dataset=meta.get("osworld_dataset", "osharm"),
            app=meta.get("osworld_app"),
            threat_category=meta.get("osworld_threat_category"),
            violation_type=meta.get("osworld_violation_type"),
            task_ids=",".join(str(x) for x in meta["osworld_task_ids"])
            if isinstance(meta.get("osworld_task_ids"), list)
            else meta.get("osworld_task_ids"),
            max_tasks=meta.get("osworld_max_tasks"),
            seed=meta.get("osworld_seed"),
            max_turns=config.scheduler.max_turns if config.scheduler else 100,
            max_time_seconds=config.scheduler.max_time_seconds
            if config.scheduler
            else 600.0,
            max_screenshots=meta.get("osworld_max_screenshots", 1),
            computer_timeout=meta.get("osworld_computer_timeout", 180),
            condition=meta.get("osworld_condition"),
            judge_model=meta.get("osworld_judge_model", "openai/gpt-4.1"),
            attack_preset=meta.get("osworld_attack_preset"),
            defense_preset=meta.get("osworld_defense_preset"),
        )

    return security_benchmark(config=config)


def _resolve_models(
    model_arg: str | list[str] | None,
    config_models: list[str],
) -> str | list[str]:
    """Resolve model specification from CLI arg and config."""
    if model_arg:
        return model_arg
    if config_models:
        return config_models if len(config_models) > 1 else config_models[0]
    return "openai/gpt-4o"
