"""Shared helpers for scenario ``@task`` factories and metadata readers.

Every scenario repeats the same three blocks by hand (issue #16): reconstruct
its scenario-config model from Sample metadata (the ``orbit run`` path),
resolve the topology-selection parameters (``condition`` / ``agents`` /
``topology_file``), and assemble the template ``ExperimentConfig`` it hands to
:func:`orbit.tasks.builder.build_scenario_task`. Each hand-rolled copy is a
place for the factory and the metadata reader to desync — the root cause of
the issue-#9 shorthand-drop class. These helpers derive all three from the
scenario's declarations (its Pydantic scenario-config model and its preset
registries), so a field/param is declared once.
"""

from __future__ import annotations

import types
import typing
from typing import TYPE_CHECKING, Any, Callable

from orbit.scenarios.params import csv_tuple, split_csv

if TYPE_CHECKING:
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.setup import SetupConfig


def _is_multi_valued(annotation: Any) -> tuple[bool, bool]:
    """Whether a model-field annotation is list-like -> (multi, as_tuple)."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(annotation):
            multi, as_tuple = _is_multi_valued(arg)
            if multi:
                return multi, as_tuple
        return False, False
    if origin is list:
        return True, False
    if origin is tuple:
        return True, True
    return False, False


def scenario_config_from_metadata(
    config: "ExperimentConfig",
    model_cls: type,
    *,
    prefix: str,
    aliases: dict[str, str] | None = None,
    defaults: dict[str, Any] | None = None,
    scheduler_fields: dict[str, str] | None = None,
    overrides: dict[str, Callable[["ExperimentConfig", dict], Any]] | None = None,
):
    """Reconstruct a scenario-config model from ``config.metadata``.

    The ``-T`` factory serialises the resolved model under
    ``{prefix}_scenario_config`` (returned verbatim); a hand-authored ``orbit
    run`` YAML instead carries flat ``{prefix}_{field}`` keys, which are read
    off the model's own field declarations: list/tuple-typed fields get
    tolerant CSV parsing, absent keys fall back to the model's defaults, and
    turn/time budgets come from ``config.scheduler`` so the two configs cannot
    disagree.

    Args:
        config: The experiment config whose metadata to read.
        model_cls: The scenario-config Pydantic model.
        prefix: Metadata key prefix (``{prefix}_{field}``).
        aliases: field name -> metadata key *suffix*, for legacy keys whose
            spelling differs from the field (``apps`` <- ``osworld_app``).
            The aliased key is read IN ADDITION to the field-named key (the
            field-named key wins if both are present).
        defaults: field name -> value used when no key is present, overriding
            the model default (osworld's per-task default dataset).
        scheduler_fields: field name -> ``config.scheduler`` attribute.
            Default: ``max_turns`` and ``max_time_seconds`` mapped to their
            scheduler namesakes when the model declares them.
        overrides: field name -> ``callable(config, metadata) -> value`` for
            fields whose reconstruction is genuinely scenario-specific; a
            returned ``None`` means "leave to the model default".
    """
    meta = config.metadata or {}
    dump_key = f"{prefix}_scenario_config"
    if dump_key in meta:
        return model_cls(**meta[dump_key])

    if scheduler_fields is None:
        scheduler_fields = {
            f: f
            for f in ("max_turns", "max_time_seconds")
            if f in model_cls.model_fields
        }
    aliases = aliases or {}
    defaults = defaults or {}
    overrides = overrides or {}

    kwargs: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        if name in overrides:
            value = overrides[name](config, meta)
            if value is not None:
                kwargs[name] = value
            continue
        if name in scheduler_fields:
            kwargs[name] = getattr(config.scheduler, scheduler_fields[name])
            continue
        value = meta.get(f"{prefix}_{name}")
        if value is None and name in aliases:
            value = meta.get(f"{prefix}_{aliases[name]}")
        if value is None:
            if name in defaults:
                kwargs[name] = defaults[name]
            continue  # absent -> model default applies
        multi, as_tuple = _is_multi_valued(field.annotation)
        if multi:
            value = csv_tuple(value) if as_tuple else split_csv(value)
        kwargs[name] = value
    return model_cls(**kwargs)


def template_experiment_config(
    *,
    scenario_name: str,
    description: str,
    prefix: str,
    scenario_config: Any,
    max_turns: int,
    max_time_seconds: float,
    setup: "SetupConfig | None" = None,
    execution: Any = None,
    attacks: list | None = None,
    defenses: list | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> "ExperimentConfig":
    """The template ``ExperimentConfig`` every ``@task`` factory hands to the
    shared builder — scenario-config dump under ``{prefix}_scenario_config``
    (the standard metadata round-trip), scheduler budget mirrored from the
    scenario config, ``halt_on_convergence`` off (dataset scenarios halt on
    their own criteria).
    """
    from orbit.configs.experiment import ExperimentConfig
    from orbit.configs.metrics import MetricsConfig
    from orbit.configs.scenario import ScenarioConfig
    from orbit.configs.scheduler import SchedulerConfig

    metadata = {f"{prefix}_scenario_config": scenario_config.model_dump()}
    if extra_metadata:
        metadata.update(extra_metadata)

    kwargs: dict[str, Any] = {
        "name": scenario_name,
        "description": description,
        "scenario": ScenarioConfig(name=scenario_name, description=description),
        "attacks": attacks or [],
        "defenses": defenses or [],
        "scheduler": SchedulerConfig(
            max_turns=max_turns,
            max_time_seconds=max_time_seconds,
            halt_on_convergence=False,
        ),
        "metrics": MetricsConfig(),
        "metadata": metadata,
    }
    if setup is not None:
        kwargs["setup"] = setup
    if execution is not None:
        kwargs["execution"] = execution
    return ExperimentConfig(**kwargs)


def resolve_topology_params(
    *,
    condition: str | None = None,
    agents: str | None = None,
    topology: str | None = None,
    memory: str | None = None,
    instructions: str | None = None,
    topology_file: str | None = "default",
    get_condition_setup: Callable[[str], "SetupConfig"],
    resolve_condition: Callable[..., str] | None = None,
    default_setup: Callable[[], "SetupConfig"] | None = None,
) -> tuple["SetupConfig | None", str | None]:
    """Resolve the standard topology-selection ``-T`` params to a template.

    Three mutually exclusive sources, identical semantics for every scenario
    (previously four hand-rolled variants): a named ``condition`` preset;
    human-readable ``agents``/``topology``/``memory``/``instructions`` knobs
    resolved to a condition via the scenario's ``resolve_condition``; or a
    ``topology_file`` YAML. Returns ``(setup, resolved_condition_name)`` —
    setup is ``None`` when the scenario has no default template and nothing
    was selected.
    """
    has_condition = condition is not None
    has_agents = agents is not None
    has_file = topology_file not in (None, "default")
    has_sub = any(p is not None for p in (topology, memory, instructions))

    if has_condition and has_agents:
        raise ValueError("Cannot specify both --condition and --agents. Use one or the other.")
    if has_condition and has_file:
        raise ValueError("Cannot specify both --condition and --topology_file.")
    if has_agents and has_file:
        raise ValueError("Cannot specify both --agents and --topology_file.")
    if (has_condition or has_file) and has_sub:
        raise ValueError("--topology, --memory, --instructions are only used with --agents.")
    if has_sub and not has_agents:
        raise ValueError("--topology, --memory, --instructions require --agents.")

    if has_agents:
        if resolve_condition is None:
            raise ValueError("This scenario does not support --agents; use --condition.")
        condition = resolve_condition(
            agents=agents,
            topology=topology or "star",
            memory=memory or "none",
            instructions=instructions or "detailed",
        )
    if condition is not None:
        return get_condition_setup(condition), condition
    if has_file:
        import yaml

        from orbit.configs.setup import SetupConfig

        with open(topology_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Tolerate a full experiment YAML being passed: unwrap its setup block.
        if "setup" in data and ("scenario" in data or "name" in data):
            data = data["setup"]
        return SetupConfig(**data), None
    return (default_setup() if default_setup is not None else None), None
