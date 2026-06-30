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
