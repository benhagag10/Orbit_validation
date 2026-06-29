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
        config = _exp("browserart", metadata={"browserart_max_behaviors": 1})
        task = self._build_or_skip(config)

        configs = _sample_configs(task)
        assert configs, "expected at least one expanded behavior config"
        for exp in configs:
            # The 2-agent star topology survives (was collapsed to 1 before #9).
            assert [a.name for a in exp.setup.agents] == ["orchestrator", "worker"]
            assert [a.name for a in exp.attacks] == ["inj"]
            assert [d.name for d in exp.defenses] == ["vax"]
