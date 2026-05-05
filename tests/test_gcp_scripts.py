"""Tests for GCP experiment scripts and defense presets."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from orbit.configs.defense import DefenseConfig
from orbit.defenses.base import DefenseBase
from orbit.defenses.registry import get_defense
from orbit.scenarios.browser.browserart.presets import get_defense_preset
from scripts.validate_defense_evals import EvalReport, validate_eval


# ---------------------------------------------------------------------------
# Defense preset tests
# ---------------------------------------------------------------------------


class TestDualLLMPreset:
    def test_loads(self):
        configs = get_defense_preset("dual_llm")
        assert len(configs) == 1
        assert configs[0].defense_type == "dual_llm"

    def test_model_override_set(self):
        configs = get_defense_preset("dual_llm")
        assert configs[0].model_override == "openai/gpt-4o-mini"

    def test_registry_instantiation(self):
        configs = get_defense_preset("dual_llm")
        defense = get_defense(configs[0])
        assert isinstance(defense, DefenseBase)

    def test_all_neurips_presets_load(self):
        for name in ["system_prompt", "llm_monitor", "guardian", "dual_llm"]:
            configs = get_defense_preset(name)
            assert len(configs) >= 1, f"Preset {name} returned empty"
            for c in configs:
                assert isinstance(c, DefenseConfig)


# ---------------------------------------------------------------------------
# Runner DEFENSES list test
# ---------------------------------------------------------------------------


class TestRunnerConfig:
    def test_defenses_list_has_four(self):
        import importlib
        mod = importlib.import_module("scripts.run_neurips_defense_experiments")
        assert len(mod.DEFENSES) == 4
        names = [d[0] for d in mod.DEFENSES]
        assert "system_prompt" in names
        assert "llm_monitor" in names
        assert "guardian" in names
        assert "dual_llm" in names

    def test_datasets_benign_count(self):
        import importlib
        mod = importlib.import_module("scripts.run_neurips_defense_experiments")
        benign = [d for d in mod.DATASETS if d[0] == "hbb_benign"]
        assert len(benign) == 1
        assert benign[0][2] == 42


# ---------------------------------------------------------------------------
# Eval validation tests
# ---------------------------------------------------------------------------


def _make_eval(
    tmp_path: Path,
    name: str,
    samples: list[dict],
) -> Path:
    """Create a minimal eval zip with given sample data."""
    eval_path = tmp_path / name
    with zipfile.ZipFile(eval_path, "w") as z:
        z.writestr("header.json", json.dumps({"status": "success"}))
        for i, sample in enumerate(samples):
            z.writestr(f"samples/{i}.json", json.dumps(sample))
    return eval_path


def _good_sample(role_count: int = 8) -> dict:
    msgs = [{"role": "system", "content": "you are..."}]
    msgs += [{"role": "user", "content": f"msg {i}"} for i in range(role_count // 2)]
    msgs += [{"role": "assistant", "content": f"resp {i}"} for i in range(role_count // 2)]
    return {
        "messages": msgs,
        "metadata": {},
        "scores": {
            "browserart_scorer": {
                "value": {"propagation_depth": 0.0, "execution_reach": 0.0, "attack_success": 0.0},
                "explanation": "PD=0.00, ER=0.00, AS=0",
                "metadata": {"task_completion": 1.0},
            }
        },
    }


def _browser_error_sample() -> dict:
    return {
        "messages": [{"role": "user", "content": "task"}],
        "metadata": {"browser_error": True, "error": "browserart service unreachable: ReadTimeout:"},
        "scores": {
            "browserart_scorer": {
                "value": {"propagation_depth": 0.0, "execution_reach": 0.0, "attack_success": 0.0},
                "explanation": "Browser error — no execution (PD=0, ER=0, AS=0)",
                "metadata": {"browser_error": True},
            }
        },
    }


def _api_error_sample() -> dict:
    return {
        "messages": [{"role": "user", "content": "task"}],
        "metadata": {"error": "credit_limit exceeded"},
        "scores": {},
    }


def _empty_gen_sample() -> dict:
    return {
        "messages": [{"role": "user", "content": "task"}, {"role": "user", "content": "axtree"}],
        "metadata": {},
        "scores": {},
    }


class TestValidateEval:
    def test_good_eval(self, tmp_path):
        path = _make_eval(tmp_path, "good.eval", [_good_sample() for _ in range(10)])
        report = validate_eval(path)
        assert report.valid is True
        assert report.good_samples == 10
        assert report.browser_errors == 0
        assert report.api_errors == 0

    def test_bad_eval_browser_errors(self, tmp_path):
        samples = [_browser_error_sample() for _ in range(8)] + [_good_sample() for _ in range(2)]
        path = _make_eval(tmp_path, "bad.eval", samples)
        report = validate_eval(path)
        assert report.valid is False
        assert report.browser_errors == 8

    def test_bad_eval_api_errors(self, tmp_path):
        samples = [_api_error_sample() for _ in range(8)] + [_good_sample() for _ in range(2)]
        path = _make_eval(tmp_path, "api_bad.eval", samples)
        report = validate_eval(path)
        assert report.valid is False
        assert report.api_errors == 8

    def test_empty_eval(self, tmp_path):
        path = _make_eval(tmp_path, "empty.eval", [])
        report = validate_eval(path)
        assert report.valid is False
        assert "Too few samples" in report.error

    def test_empty_generation_samples(self, tmp_path):
        samples = [_empty_gen_sample() for _ in range(10)]
        path = _make_eval(tmp_path, "nogen.eval", samples)
        report = validate_eval(path)
        assert report.valid is False
        assert report.empty_generations == 10

    def test_mixed_mostly_good(self, tmp_path):
        samples = [_good_sample() for _ in range(8)] + [_browser_error_sample() for _ in range(2)]
        path = _make_eval(tmp_path, "mostly_good.eval", samples)
        report = validate_eval(path)
        assert report.valid is True
        assert report.good_samples == 8
        assert report.browser_errors == 2

    def test_invalid_zip(self, tmp_path):
        path = tmp_path / "notzip.eval"
        path.write_text("not a zip")
        report = validate_eval(path)
        assert report.valid is False
        assert "Bad file" in report.error

    def test_missing_file(self, tmp_path):
        report = validate_eval(tmp_path / "nonexistent.eval")
        assert report.valid is False
