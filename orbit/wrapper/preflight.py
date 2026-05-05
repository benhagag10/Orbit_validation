"""Preflight check system for Orbit.

Runs quick prerequisite checks before experiments to catch missing
dependencies, unreachable services, and configuration errors early.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

import click

ScenarioKind = Literal[
    "browserart", "swe-bench", "osworld", "redcode_gen", "tau2", "generic"
]


@dataclass
class CheckResult:
    """Result of a single preflight check."""

    name: str
    passed: bool
    message: str = ""
    warning: bool = False


def _check_dotenv_for_api_keys() -> bool:
    """Check if .env file in the working directory contains an API key."""
    from pathlib import Path

    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.is_file():
        return False
    try:
        for line in dotenv_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            value = stripped.split("=", 1)[1].strip().strip("\"'")
            if key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") and value:
                return True
    except OSError:
        return False
    return False


def check_common() -> list[CheckResult]:
    """Check common prerequisites: Python version, API keys."""
    results: list[CheckResult] = []

    # Python version
    if sys.version_info >= (3, 11):
        results.append(CheckResult("Python >= 3.11", passed=True))
    else:
        v = f"{sys.version_info.major}.{sys.version_info.minor}"
        results.append(CheckResult(
            "Python >= 3.11", passed=False,
            message=f"Found {v} — Python 3.11+ is required",
        ))

    # At least one API key (check env vars and .env file)
    has_key = bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    dotenv_has_key = False
    if not has_key:
        dotenv_has_key = _check_dotenv_for_api_keys()
    if has_key or dotenv_has_key:
        source = ".env file" if dotenv_has_key and not has_key else ""
        results.append(CheckResult(
            "API key configured", passed=True, message=source,
        ))
    else:
        results.append(CheckResult(
            "API key configured", passed=False,
            message="Set OPENAI_API_KEY or ANTHROPIC_API_KEY in environment or .env file",
        ))

    return results


def check_config(config_path: str) -> list[CheckResult]:
    """Check that a YAML config file parses and is valid."""
    results: list[CheckResult] = []
    try:
        from orbit.wrapper.yaml_loader import load_experiment_config

        config = load_experiment_config(config_path)
        # Count samples if possible
        sample_count = ""
        if hasattr(config, "setup") and hasattr(config.setup, "agents"):
            agent_count = len(config.setup.agents)
            sample_count = f", {agent_count} agent(s)"
        results.append(CheckResult(
            "Config valid", passed=True,
            message=sample_count.lstrip(", ") if sample_count else "",
        ))
    except Exception as e:
        results.append(CheckResult(
            "Config valid", passed=False,
            message=str(e)[:120],
        ))
    return results


def _check_docker() -> CheckResult:
    """Check if Docker daemon is reachable via CLI."""
    import shutil
    import subprocess

    if not shutil.which("docker"):
        return CheckResult(
            "Docker reachable", passed=False,
            message="Docker not installed",
        )
    try:
        result = subprocess.run(
            ["docker", "info"],  # noqa: S603, S607
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return CheckResult("Docker reachable", passed=True)
        return CheckResult(
            "Docker reachable", passed=False,
            message="Docker not running -- start Docker Desktop",
        )
    except Exception:
        return CheckResult(
            "Docker reachable", passed=False,
            message="Docker command timed out or failed",
        )


def check_browserart() -> list[CheckResult]:
    """Check BrowserART-specific prerequisites."""
    results: list[CheckResult] = []

    # browsergym importable
    if find_spec("browsergym") is not None:
        results.append(CheckResult("Python packages (browsergym)", passed=True))
    else:
        results.append(CheckResult(
            "Python packages (browsergym)", passed=False,
            message="Install with: uv sync --extra browserart",
        ))

    # Test websites served
    import urllib.request

    port = int(os.environ.get("BROWSERART_PORT", "8000"))
    try:
        urllib.request.urlopen(  # noqa: S310
            f"http://localhost:{port}/", timeout=3,
        )
        results.append(CheckResult(f"BrowserART websites (port {port})", passed=True))
    except Exception:
        results.append(CheckResult(
            f"BrowserART websites (port {port})", passed=False,
            message="Not served — will auto-start, or run manually",
            warning=True,
        ))

    return results


def check_swe_bench() -> list[CheckResult]:
    """Check SWE-Bench-specific prerequisites."""
    results: list[CheckResult] = []

    # swebench + docker packages
    missing_pkgs: list[str] = []
    if find_spec("swebench") is None:
        missing_pkgs.append("swebench")
    if find_spec("docker") is None:
        missing_pkgs.append("docker")

    if not missing_pkgs:
        results.append(CheckResult("Python packages (swebench, docker)", passed=True))
    else:
        results.append(CheckResult(
            f"Python packages ({', '.join(missing_pkgs)})", passed=False,
            message="Install with: uv sync --extra swebench",
        ))

    # Docker reachable
    results.append(_check_docker())

    # ARM warning
    machine = platform.machine()
    if machine in ("arm64", "aarch64"):
        results.append(CheckResult(
            f"ARM host ({machine})", passed=True,
            message="Rosetta emulation recommended",
            warning=True,
        ))

    return results


def check_osworld() -> list[CheckResult]:
    """Check OSWorld-specific prerequisites."""
    results: list[CheckResult] = []

    # inspect-evals[osworld]
    try:
        has_osworld = find_spec("inspect_evals.osworld") is not None
    except (ModuleNotFoundError, ValueError):
        has_osworld = False

    if has_osworld:
        results.append(CheckResult("Python packages (inspect-evals[osworld])", passed=True))
    else:
        results.append(CheckResult(
            "Python packages (inspect-evals[osworld])", passed=False,
            message="Install with: uv sync --extra osworld",
        ))

    # Docker reachable
    results.append(_check_docker())

    return results


def detect_scenario_from_config(config_path: str) -> ScenarioKind:
    """Best-effort detection of scenario type from a YAML config."""
    try:
        import yaml
        from pathlib import Path

        text = Path(config_path).read_text()
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            scenario = data.get("scenario", {})
            stype = scenario.get("type", "") or scenario.get("name", "")
            if "browser" in stype.lower():
                return "browserart"
            if "swe" in stype.lower():
                return "swe-bench"
            if "osworld" in stype.lower() or "osharm" in stype.lower():
                return "osworld"
    except Exception:
        pass
    return "generic"


def run_preflight(
    scenario: ScenarioKind,
    config_path: str | None = None,
) -> bool:
    """Run all checks, print results, prompt on failure. Returns True to proceed."""
    all_results: list[CheckResult] = []

    # Common checks
    all_results.extend(check_common())

    # Config check
    if config_path:
        all_results.extend(check_config(config_path))

    # Scenario-specific checks
    if scenario == "browserart":
        all_results.extend(check_browserart())
    elif scenario == "swe-bench":
        all_results.extend(check_swe_bench())
    elif scenario == "osworld":
        all_results.extend(check_osworld())

    # Print results
    click.echo(f"\nPreflight ({scenario})")

    failures = 0
    for r in all_results:
        if r.passed and not r.warning:
            icon = "[ok]"
        elif r.warning:
            icon = "[!!]"
        else:
            icon = "[FAIL]"
            failures += 1

        line = f"  {icon:>6} {r.name}"
        msg = r.message
        if msg and len(msg) > 80:
            msg = msg[:80] + "..."
        if not r.passed and msg:
            line += f" -- {msg}"
        elif r.warning and msg:
            line += f" -- {msg}"
        elif r.passed and msg:
            line += f" ({msg})"
        click.echo(line)

    if failures == 0:
        click.echo("  Ready to go.\n")
        return True

    noun = "check" if failures == 1 else "checks"
    click.echo()
    if scenario != "generic":
        click.echo(
            f"  Tip: run `python scripts/check_setup.py --scenario {scenario}` "
            "for detailed diagnostics."
        )
    try:
        return click.confirm(
            f"  {failures} {noun} failed. Continue anyway?",
            default=False,
        )
    except click.Abort:
        return False
