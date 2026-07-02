"""Regression tests for issue #9 — config-complete Task construction.

Issue #9: ``orbit run <config>.yaml`` silently dropped topology/attacks/defenses
for browserart, swe_bench and osworld because the runner flattened the config
into a lossy kwarg subset before calling a per-scenario factory.

The fix makes Task construction scenario-agnostic: every scenario routes through
``security_benchmark(config)`` -> ``orbit.tasks.builder.build_scenario_task``,
which is total over ``ExperimentConfig`` and threads ``config.setup`` (topology),
``config.attacks`` and ``config.defenses`` through the scenario's plugin. These
tests assert that nothing set in the config can be dropped.
"""

from __future__ import annotations

import pytest

from orbit.configs.attack import AttackConfig
from orbit.configs.defense import DefenseConfig
from orbit.configs.experiment import ExperimentConfig
from orbit.configs.scenario import ScenarioConfig
from orbit.configs.scheduler import SchedulerConfig
from orbit.configs.setup import AgentSpec, SetupConfig, TopologyEdge
from orbit.dataset.metadata import MASMetadata


def _exp(scenario_name: str, **overrides) -> ExperimentConfig:
    base = dict(
        name=f"{scenario_name}-completeness",
        setup=SetupConfig(
            agents=[
                AgentSpec(name="orchestrator", role="orchestrator", tools=[]),
                AgentSpec(name="worker", role="executor", tools=["web_browser"]),
            ],
            edges=[TopologyEdge(from_agent="orchestrator", to_agent="worker", mechanism="direct_run")],
        ),
        scenario=ScenarioConfig(name=scenario_name),
        attacks=[
            AttackConfig(
                name="inj",
                attack_type="indirect_injection",
                target_agent="worker",
                payload="ignore safety",
                success_criteria="deviates",
            )
        ],
        defenses=[
            DefenseConfig(
                name="vax",
                defense_type="prompt_vaccination",
                target_agents=["orchestrator", "worker"],
            )
        ],
        scheduler=SchedulerConfig(max_turns=5, max_time_seconds=60.0),
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def _sample_configs(task) -> list[ExperimentConfig]:
    return [MASMetadata(**s.metadata).experiment for s in task.dataset]


class TestGenericPathThreadsWholeConfig:
    """The DEFAULT plugin (unregistered scenario) carries the whole config."""

    def test_single_sample_has_full_topology_attack_defense(self):
        from orbit.tasks.builder import build_scenario_task

        config = _exp("custom_scenario")  # no registered plugin -> DEFAULT_PLUGIN
        task = build_scenario_task(config, variant="default")

        configs = _sample_configs(task)
        assert len(configs) == 1
        exp = configs[0]
        assert [a.name for a in exp.setup.agents] == ["orchestrator", "worker"]
        assert len(exp.setup.edges) == 1
        assert [a.name for a in exp.attacks] == ["inj"]
        assert [d.name for d in exp.defenses] == ["vax"]

    def test_security_benchmark_entry_point_is_lossless(self):
        from orbit.tasks.security_benchmark import security_benchmark

        config = _exp("custom_scenario")
        task = security_benchmark(config=config)
        exp = _sample_configs(task)[0]
        assert len(exp.setup.agents) == 2
        assert exp.attacks and exp.defenses


class TestBrowserARTPathThreadsWholeConfig:
    """browserart routed through the runner keeps topology/attacks/defenses.

    Requires the BrowserART (hbb) dataset to be available locally; skipped
    otherwise.
    """

    def _build_or_skip(self, config: ExperimentConfig):
        from orbit.wrapper.runner import _build_task

        try:
            return _build_task(config)
        except Exception as exc:  # missing dataset / optional dep
            pytest.skip(f"BrowserART dataset/deps unavailable: {exc}")

    def test_multi_agent_attack_defense_not_dropped(self):
        config = _exp("browserart", metadata={"browserart_task_ids": "0"})
        task = self._build_or_skip(config)

        configs = _sample_configs(task)
        assert configs, "expected at least one expanded behavior config"
        for exp in configs:
            # The 2-agent star topology survives (was collapsed to 1 before #9).
            assert [a.name for a in exp.setup.agents] == ["orchestrator", "worker"]
            assert [a.name for a in exp.attacks] == ["inj"]
            assert [d.name for d in exp.defenses] == ["vax"]


class TestBuilderEnforcesDimensionFloor:
    """The shared builder makes a silent attack/defense drop structurally
    impossible — for every scenario, present or future — regardless of whether
    a plugin author remembered to thread the dimensions through ``expand``.

    ``expand`` MAY transform attacks per sample (swe_bench resolves an ``auto``
    side-task per group) or ADD scenario-specific ones (code_ipi prepends a
    per-task injection); it may never DROP a declared type. The builder enforces
    that floor centrally.
    """

    def _plugin(self, expand):
        from orbit.scenarios.registry import ScenarioPlugin

        return ScenarioPlugin(name="custom_scenario", expand=expand)

    def test_dropping_both_dimensions_raises(self):
        from orbit.tasks.builder import ScenarioExpansionError, build_scenario_task

        config = _exp("custom_scenario")
        plugin = self._plugin(
            lambda c: [c.model_copy(update={"attacks": [], "defenses": []})]
        )
        with pytest.raises(ScenarioExpansionError):
            build_scenario_task(config, plugin=plugin)

    def test_dropping_only_defense_raises(self):
        from orbit.tasks.builder import ScenarioExpansionError, build_scenario_task

        config = _exp("custom_scenario")
        plugin = self._plugin(lambda c: [c.model_copy(update={"defenses": []})])
        with pytest.raises(ScenarioExpansionError):
            build_scenario_task(config, plugin=plugin)

    def test_transforming_attack_same_type_passes(self):
        from orbit.tasks.builder import build_scenario_task

        config = _exp("custom_scenario")

        def expand(c):  # same attack_type, different payload (swe_bench `auto`)
            transformed = c.attacks[0].model_copy(update={"payload": "resolved-per-sample"})
            return [c.model_copy(update={"attacks": [transformed]})]

        task = build_scenario_task(config, plugin=self._plugin(expand))
        exp = _sample_configs(task)[0]
        assert [a.attack_type for a in exp.attacks] == ["indirect_injection"]
        assert exp.attacks[0].payload == "resolved-per-sample"

    def test_adding_scenario_specific_attack_passes(self):
        from orbit.tasks.builder import build_scenario_task

        config = _exp("custom_scenario")
        extra = AttackConfig(
            name="ipi", attack_type="indirect_injection", payload="scenario-specific"
        )

        def expand(c):  # code_ipi-style: scenario attack on top of the declared one
            return [c.model_copy(update={"attacks": [extra, *c.attacks]})]

        task = build_scenario_task(config, plugin=self._plugin(expand))
        exp = _sample_configs(task)[0]
        assert len(exp.attacks) == 2

    def test_resolve_dropping_attack_raises(self):
        # The floor is snapshotted PRE-resolve, so a resolve() that strips a
        # declared attack (not via a baseline) is caught, not silently obeyed.
        from orbit.scenarios.registry import ScenarioPlugin
        from orbit.tasks.builder import ScenarioExpansionError, build_scenario_task

        config = _exp("custom_scenario")
        plugin = ScenarioPlugin(
            name="custom_scenario",
            resolve=lambda c: c.model_copy(update={"attacks": []}),
        )
        with pytest.raises(ScenarioExpansionError):
            build_scenario_task(config, plugin=plugin)


class TestBuilderEnforcesTopologyFloor:
    """A ``threads_setup`` plugin (the default) may not collapse the agent count
    or drop shared memory during expansion — the "star silently degenerates to a
    single agent" failure. ``threads_setup=False`` scenarios are exempt but warn.
    """

    def _collapsing_expand(self, c):
        one = c.setup.model_copy(update={"agents": c.setup.agents[:1], "edges": []})
        return [c.model_copy(update={"setup": one})]

    def test_collapsing_topology_raises(self):
        from orbit.scenarios.registry import ScenarioPlugin
        from orbit.tasks.builder import ScenarioExpansionError, build_scenario_task

        config = _exp("custom_scenario")  # 2 agents
        plugin = ScenarioPlugin(name="custom_scenario", expand=self._collapsing_expand)
        with pytest.raises(ScenarioExpansionError, match="collapsed the topology"):
            build_scenario_task(config, plugin=plugin)

    def test_dropping_shared_memory_raises(self):
        from orbit.configs.setup import MemoryConfig, SetupConfig
        from orbit.scenarios.registry import ScenarioPlugin
        from orbit.tasks.builder import ScenarioExpansionError, build_scenario_task

        config = _exp(
            "custom_scenario",
            setup=SetupConfig(
                agents=[
                    AgentSpec(name="a", role="orchestrator", tools=[]),
                    AgentSpec(name="b", role="executor", tools=[]),
                ],
                memory=MemoryConfig(shared=True),
            ),
        )

        def expand(c):
            no_shared = c.setup.model_copy(
                update={"memory": c.setup.memory.model_copy(update={"shared": False})}
            )
            return [c.model_copy(update={"setup": no_shared})]

        plugin = ScenarioPlugin(name="custom_scenario", expand=expand)
        with pytest.raises(ScenarioExpansionError, match="shared memory"):
            build_scenario_task(config, plugin=plugin)

    def test_threads_setup_false_is_exempt_and_warns(self, caplog):
        import logging

        from orbit.scenarios.registry import ScenarioPlugin
        from orbit.tasks.builder import build_scenario_task

        config = _exp("custom_scenario")  # 2 agents
        plugin = ScenarioPlugin(
            name="custom_scenario",
            expand=self._collapsing_expand,
            threads_setup=False,
        )
        with caplog.at_level(logging.WARNING, logger="orbit.tasks.builder"):
            task = build_scenario_task(config, plugin=plugin)  # must NOT raise
        assert len(_sample_configs(task)[0].setup.agents) == 1
        assert any("IGNORED" in r.message for r in caplog.records)


class TestExpandingPluginsPreserveDimensions:
    """Every real expanding plugin must thread config.attacks/defenses into the
    per-sample configs it produces. agentharm and tau2 — the two that dropped
    them before this fix — are locked here against regression.
    """

    def _build_or_skip(self, config: ExperimentConfig):
        from orbit.scenarios.registry import get_scenario
        from orbit.tasks.builder import build_scenario_task

        plugin = get_scenario(config.scenario.name)
        assert plugin is not None, f"no plugin for {config.scenario.name}"
        try:
            return build_scenario_task(config, plugin=plugin)
        except Exception as exc:  # missing dataset / optional dep
            pytest.skip(f"{config.scenario.name} data/deps unavailable: {exc}")

    def test_agentharm_threads_attacks_defenses(self):
        from orbit.scenarios.agentharm.configs import AgentHarmScenarioConfig

        meta = {
            "agentharm_scenario_config": AgentHarmScenarioConfig().model_dump(),
            "agentharm_condition": "single_agent",
        }
        task = self._build_or_skip(_exp("agentharm", metadata=meta))
        configs = _sample_configs(task)
        assert configs
        for exp in configs:
            assert "indirect_injection" in [a.attack_type for a in exp.attacks]
            assert "prompt_vaccination" in [d.defense_type for d in exp.defenses]

    def test_tau2_threads_attacks_defenses(self):
        task = self._build_or_skip(_exp("tau2_airline"))
        configs = _sample_configs(task)
        assert configs
        for exp in configs:
            assert "indirect_injection" in [a.attack_type for a in exp.attacks]
            assert "prompt_vaccination" in [d.defense_type for d in exp.defenses]


class TestShorthandResolvedOnOrbitRun:
    """`orbit run` must resolve scenario *shorthand* (preset / condition metadata
    keys) into the full config — the deleted runner dispatch was the only
    resolver, so a shipped YAML relying on a preset/condition silently degraded.
    These lock in the ``ScenarioPlugin.resolve`` hook (same issue-#9 class, on
    the reader path the builder invariant alone could not see).
    """

    def _build_or_skip(self, path: str):
        import os

        from orbit.wrapper.runner import _build_task
        from orbit.wrapper.yaml_loader import load_experiment_config

        if not os.path.exists(path):
            pytest.skip(f"{path} not present")
        try:
            return _build_task(load_experiment_config(path))
        except Exception as exc:  # missing dataset / optional dep
            pytest.skip(f"{path} data/deps unavailable: {exc}")

    def test_swe_bench_attack_preset_resolved_not_benign(self):
        # examples/swe_bench_ipi_preset.yaml declares attacks: [] and relies on
        # metadata.swe_bench_attack_preset — must NOT silently run benign.
        task = self._build_or_skip("examples/swe_bench_ipi_preset.yaml")
        exp = _sample_configs(task)[0]
        assert [a.attack_type for a in exp.attacks] == ["codebase_injection"], (
            "swe_bench_attack_preset must resolve into an attack on `orbit run`"
        )

    def test_osworld_condition_resolved_to_correct_memory(self):
        # 08 names memory_shared_actions; must restore shared=True + shared-action
        # access (was degrading to default non-shared memory — a P0 violation).
        task = self._build_or_skip(
            "examples/osharm_misuse_conditions/08_star_shared_memory.yaml"
        )
        exp = _sample_configs(task)[0]
        mem = exp.setup.memory
        assert mem.shared is True
        assert mem.agent_memory_access and all(
            a.shared_action_history for a in mem.agent_memory_access
        ), "osworld_condition=memory_shared_actions must restore shared-action access"


class TestBaselineDoesNotReacquireStrippedDimensions:
    """A plugin's ``resolve`` must not re-materialise a *preset* attack/defense
    that a baseline intentionally stripped. Otherwise the ``benign``/``no_attack``
    control silently runs WITH the attack — a contaminated reference the
    attack-effect delta is measured against (a construct-validity defect the
    dimension floor cannot see, since post-resolve and produced configs agree).
    """

    def test_baseline_gate_helpers(self):
        from orbit.configs.baseline import BaselineMode
        from orbit.tasks.builder import baseline_keeps_attacks, baseline_keeps_defenses

        def cfg(mode):
            return _exp("custom_scenario").model_copy(update={"baseline_mode": mode})

        assert baseline_keeps_attacks(cfg(BaselineMode.NONE))
        assert baseline_keeps_attacks(cfg(BaselineMode.NO_DEFENSE))
        assert not baseline_keeps_attacks(cfg(BaselineMode.NO_ATTACK))
        assert not baseline_keeps_attacks(cfg(BaselineMode.BENIGN))
        assert baseline_keeps_defenses(cfg(BaselineMode.NONE))
        assert baseline_keeps_defenses(cfg(BaselineMode.NO_ATTACK))
        assert not baseline_keeps_defenses(cfg(BaselineMode.NO_DEFENSE))
        assert not baseline_keeps_defenses(cfg(BaselineMode.BENIGN))

    def test_swe_bench_preset_stripped_by_benign_baseline(self):
        from orbit.baselines.baselines import apply_baseline
        from orbit.configs.baseline import BaselineMode
        from orbit.scenarios.registry import get_scenario

        plugin = get_scenario("swe_bench")
        if plugin is None or plugin.resolve is None:
            pytest.skip("swe_bench plugin unavailable")
        base = _exp(
            "swe_bench",
            setup=SetupConfig(agents=[AgentSpec(name="solver", role="worker")]),
            attacks=[],
            defenses=[],
            metadata={"swe_bench_attack_preset": "codebase_injection"},
        )
        # baseline=none: the preset resolves into a real attack.
        assert [a.attack_type for a in plugin.resolve(base).attacks] == [
            "codebase_injection"
        ]
        # benign / no_attack: it must stay stripped (no re-injection).
        for mode in (BaselineMode.BENIGN, BaselineMode.NO_ATTACK):
            stripped = apply_baseline(base, mode)
            assert plugin.resolve(stripped).attacks == [], (
                f"{mode.value} baseline re-acquired the preset attack"
            )


class TestEpochsHonoredForPluginScenarios:
    """A YAML ``epochs: N`` must reach the Task for every scenario, not only the
    generic path — it was silently dropped for plugin scenarios (build_task_kwargs
    returned no epochs), so ``epochs`` was a no-op on the ``orbit run`` path.
    """

    @staticmethod
    def _epochs(task) -> int:
        e = task.epochs
        return e.epochs if hasattr(e, "epochs") else e

    def test_generic_and_plugin_scenarios_get_epochs(self):
        from orbit.scenarios.registry import ScenarioPlugin
        from orbit.tasks.builder import build_scenario_task

        config = _exp("custom_scenario").model_copy(update={"epochs": 4})
        # Generic path.
        assert self._epochs(build_scenario_task(config)) == 4
        # A plugin whose build_task_kwargs sets unrelated knobs still gets epochs.
        plugin = ScenarioPlugin(
            name="custom_scenario", build_task_kwargs=lambda c: {"message_limit": 60}
        )
        task = build_scenario_task(config, plugin=plugin)
        assert self._epochs(task) == 4
        assert task.message_limit == 60

    def test_plugin_can_override_epochs(self):
        from inspect_ai import Epochs
        from orbit.scenarios.registry import ScenarioPlugin
        from orbit.tasks.builder import build_scenario_task

        config = _exp("custom_scenario").model_copy(update={"epochs": 4})
        plugin = ScenarioPlugin(  # tau2-style: pins its own epochs
            name="custom_scenario", build_task_kwargs=lambda c: {"epochs": Epochs(1)}
        )
        assert self._epochs(build_scenario_task(config, plugin=plugin)) == 1


class TestConditionSetupMutualExclusivity:
    """Topology may come from an inline ``setup`` OR a ``*_condition`` preset,
    never both. Declaring both is a construct-validity error (it silently
    collapsed a condition to the lone validator placeholder before); a
    condition-only config resolves to the preset's real topology.
    """

    def _plugin(self, name):
        from orbit.scenarios.registry import get_scenario

        plugin = get_scenario(name)
        if plugin is None or plugin.resolve is None:
            pytest.skip(f"{name} plugin unavailable")
        return plugin

    def test_browserart_condition_only_resolves_topology(self):
        plugin = self._plugin("browserart")
        c = _exp(
            "browserart",
            setup=SetupConfig(agents=[]),
            attacks=[],
            defenses=[],
            metadata={"browserart_condition": "star_specialist"},
        )
        resolved = plugin.resolve(c)
        assert len(resolved.setup.agents) > 1, (
            "condition-only config must resolve to the multi-agent topology, "
            "not collapse to a placeholder"
        )

    def test_browserart_condition_plus_setup_raises(self):
        plugin = self._plugin("browserart")
        c = _exp(
            "browserart",
            setup=SetupConfig(agents=[AgentSpec(name="placeholder", role="worker")]),
            attacks=[],
            defenses=[],
            metadata={"browserart_condition": "star_specialist"},
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            plugin.resolve(c)

    def test_osworld_condition_plus_setup_raises(self):
        plugin = self._plugin("osworld")
        c = _exp(
            "osworld",
            setup=SetupConfig(agents=[AgentSpec(name="placeholder", role="worker")]),
            attacks=[],
            defenses=[],
            metadata={"osworld_condition": "star_specialist", "osworld_dataset": "osharm"},
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            plugin.resolve(c)

    def test_resolve_is_idempotent(self):
        plugin = self._plugin("browserart")
        c = _exp(
            "browserart",
            setup=SetupConfig(agents=[]),
            attacks=[],
            defenses=[],
            metadata={"browserart_condition": "star_specialist"},
        )
        once = plugin.resolve(c)
        twice = plugin.resolve(once)  # must not raise (trigger key consumed)
        assert [a.name for a in once.setup.agents] == [a.name for a in twice.setup.agents]


class TestCentralConfigValidationSurfaced:
    """ConfigValidator runs centrally in the builder, so an unknown target_agent
    is surfaced even for scenarios that override build_setup and skip the setup
    solver where validation used to be the only gate. Logged (not raised) to
    avoid false-positives on runtime agent-expansion targets (issue #7).
    """

    def test_unknown_target_agent_warns(self, caplog):
        import logging

        from orbit.tasks.builder import build_scenario_task

        bad = _exp(
            "custom_scenario",
            attacks=[
                AttackConfig(
                    name="x",
                    attack_type="indirect_injection",
                    target_agent="ghost",  # not in setup.agents
                    payload="p",
                    success_criteria="s",
                )
            ],
        )
        with caplog.at_level(logging.WARNING, logger="orbit.tasks.builder"):
            build_scenario_task(bad)
        assert any("validation issue" in r.message for r in caplog.records)


class TestKnownScenarioImportFailsLoud:
    """A registered scenario whose module fails to import must raise, never
    silently fall back to the generic benign single-sample task (that would run
    a declared scenario as a benign run with no error — the issue-#9 class on
    the reader path). A genuinely unknown name still falls back, but loudly.
    """

    def test_broken_known_scenario_raises(self, monkeypatch):
        import orbit.scenarios.registry as reg

        # Point a known scenario at a module that raises on import, and clear any
        # cached plugin so the loader is re-run.
        monkeypatch.setitem(reg._LOADERS, "swe_bench", "orbit._does_not_exist_xyz")
        monkeypatch.delitem(reg._PLUGINS, "swe_bench", raising=False)
        with pytest.raises(reg.ScenarioImportError):
            reg.get_scenario("swe_bench")

    def test_unknown_scenario_falls_back_to_generic(self, caplog):
        import logging

        from orbit.scenarios.registry import get_scenario
        from orbit.tasks.builder import DEFAULT_PLUGIN, build_scenario_task

        assert get_scenario("totally_made_up_scenario") is None
        config = _exp("totally_made_up_scenario")
        with caplog.at_level(logging.WARNING, logger="orbit.tasks.builder"):
            task = build_scenario_task(config)
        assert any("No scenario plugin registered" in r.message for r in caplog.records)
        # Generic path still threads the whole config (attacks/defenses/topology).
        exp = _sample_configs(task)[0]
        assert exp.attacks and exp.defenses and len(exp.setup.agents) == 2
