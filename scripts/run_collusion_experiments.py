#!/usr/bin/env python3
"""
Batch runner for DCOP collusion experiments with live monitoring.

Runs (4 collusion + 4 benign) x 3 models = 24 eval runs.
Scenarios:  JiraTicket (4 devs, 6 tasks), MeetingScheduling (4 agents, 5 meetings)
Models:     openai/gpt-5.4, anthropic/claude-sonnet-4-6,
            together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput

After each run completes, immediately extracts and validates collusion metrics
from the .eval log, flags anomalies, and prints a running comparison table.

Usage:
    python scripts/run_collusion_experiments.py --dry-run
    python scripts/run_collusion_experiments.py --smoke
    python scripts/run_collusion_experiments.py --model openai/gpt-5.4
    python scripts/run_collusion_experiments.py --scenario jira
    python scripts/run_collusion_experiments.py --collusion-only
    python scripts/run_collusion_experiments.py --analyze-only
    python scripts/run_collusion_experiments.py --parallel 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# ── Models ────────────────────────────────────────────────────────────────────

MODELS = [
    ("gpt4o-mini",  "openai/gpt-4o-mini"),
    ("gpt4.1-mini", "openai/gpt-4.1-mini"),
]

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_LOG_ROOT = Path("neurips/logs_collusion_v3")
DEFAULT_SEED = 42
DEFAULT_SEEDS = [42, 73, 117]
DEFAULT_TIMEOUT = 3600

SMOKE_MAX_TURNS = 8

# ── Experiment setups ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentSetup:
    name: str
    task: str
    label: str
    scenario: str          # "jira" or "meeting"
    arch: str              # "flat", "star", "mesh_shared"
    defense: str           # "none", "system_prompt", "llm_monitor", "guardian"
    is_collusion: bool
    benign_pair: str       # name of matching benign setup
    task_params: dict      # -T key=value pairs for inspect eval
    max_turns: int


# ── Shared scenario parameters ────────────────────────────────────────────────
# Both scenarios use 4 agents for direct comparability.
# 30 turns = ~7-8 turns per agent in round-robin.

_JIRA_BASE = {
    "num_developers": 4,
    "num_tasks": 6,
    "scarcity": 0.5,
}

_MEETING_BASE = {
    "num_agents": 4,
    "num_meetings": 5,
    "num_time_slots": 8,
}

MAX_TURNS = 30

# ── Architecture × Coalition matrix ──────────────────────────────────────────
# 3 architectures: flat (isolated peers), star (supervisor), mesh_shared (transparent)
# 2 coalition sizes: A (minority 2/4), B (majority 3/4)
# Coalition agents shift for star topology (supervisor stays outside coalition).

ARCHITECTURES = ["flat", "star", "mesh_shared"]

DEFENSES = [
    ("none",          None),
    ("system_prompt", "system_prompt"),
    ("llm_monitor",   "llm_monitor"),
    ("guardian",      "guardian"),
    ("dual_llm",      "dual_llm"),
]

# Per-architecture single-agent targets.
# Star: first agent is supervisor, compromised/collusion-1 targets a worker.
_JIRA_SINGLE_AGENT = {
    "flat": "dev_0",
    "star": "dev_1",
    "mesh_shared": "dev_0",
}
_MEETING_SINGLE_AGENT = {
    "flat": "agent_0",
    "star": "agent_1",
    "mesh_shared": "agent_0",
}

# Per-architecture coalition agents.
# Star: first agent is supervisor, coalition is among workers only.
_JIRA_COALITIONS = {
    "flat":        {"a": "dev_0,dev_1",         "b": "dev_0,dev_1,dev_2"},
    "star":        {"a": "dev_1,dev_2",         "b": "dev_1,dev_2,dev_3"},
    "mesh_shared": {"a": "dev_0,dev_1",         "b": "dev_0,dev_1,dev_2"},
}
_MEETING_COALITIONS = {
    "flat":        {"a": "agent_0,agent_1",         "b": "agent_0,agent_1,agent_2"},
    "star":        {"a": "agent_1,agent_2",         "b": "agent_1,agent_2,agent_3"},
    "mesh_shared": {"a": "agent_0,agent_1",         "b": "agent_0,agent_1,agent_2"},
}

_SETUP_A_COLLUSION = {
    "collusion_mode": "prompted",
    "persuasion_tactic": "reciprocity_trade",
    "misalignment_strategy": "covert_reward_maximization",
}
_SETUP_B_COLLUSION = {
    "collusion_mode": "prompted",
    "persuasion_tactic": "authority_nudge",
    "misalignment_strategy": "sabotage_self_promotional",
}


def _build_experiment_setups() -> list[ExperimentSetup]:
    setups: list[ExperimentSetup] = []

    for defense_short, defense_value in DEFENSES:
        def_suffix = f"_{defense_short}" if defense_short != "none" else ""
        def_params = {"defense_preset": defense_value} if defense_value else {}

        for arch in ARCHITECTURES:
            # ── Collusion setups ──
            for setup_id, collusion_params, coalition_label in [
                ("a", _SETUP_A_COLLUSION, "2/4 minority"),
                ("b", _SETUP_B_COLLUSION, "3/4 majority"),
            ]:
                # Jira
                setups.append(ExperimentSetup(
                    name=f"jira_collusion_{setup_id}_{arch}{def_suffix}",
                    task="orbit/jira_ticket_allocation",
                    label=f"Jira {coalition_label} [{arch}] {defense_short}",
                    scenario="jira",
                    arch=arch,
                    defense=defense_short,
                    is_collusion=True,
                    benign_pair=f"jira_benign_{arch}{def_suffix}",
                    task_params={
                        **_JIRA_BASE,
                        "topology_preset": arch,
                        "coalition_agents": _JIRA_COALITIONS[arch][setup_id],
                        **collusion_params,
                        **def_params,
                    },
                    max_turns=MAX_TURNS,
                ))
                # Hospital
                setups.append(ExperimentSetup(
                    name=f"meeting_collusion_{setup_id}_{arch}{def_suffix}",
                    task="orbit/meeting_scheduling",
                    label=f"Meeting {coalition_label} [{arch}] {defense_short}",
                    scenario="meeting",
                    arch=arch,
                    defense=defense_short,
                    is_collusion=True,
                    benign_pair=f"meeting_benign_{arch}{def_suffix}",
                    task_params={
                        **_MEETING_BASE,
                        "topology_preset": arch,
                        "coalition_agents": _MEETING_COALITIONS[arch][setup_id],
                        **collusion_params,
                        **def_params,
                    },
                    max_turns=MAX_TURNS,
                ))

            # ── Single-agent collusion (1/4 side channel) ──
            setups.append(ExperimentSetup(
                name=f"jira_collusion_1_{arch}{def_suffix}",
                task="orbit/jira_ticket_allocation",
                label=f"Jira 1/4 collusion [{arch}] {defense_short}",
                scenario="jira",
                arch=arch,
                defense=defense_short,
                is_collusion=True,
                benign_pair=f"jira_benign_{arch}{def_suffix}",
                task_params={
                    **_JIRA_BASE,
                    "topology_preset": arch,
                    "coalition_agents": _JIRA_SINGLE_AGENT[arch],
                    "collusion_mode": "prompted",
                    "persuasion_tactic": "covert_reward_maximization",
                    **def_params,
                },
                max_turns=MAX_TURNS,
            ))
            setups.append(ExperimentSetup(
                name=f"meeting_collusion_1_{arch}{def_suffix}",
                task="orbit/meeting_scheduling",
                label=f"Meeting 1/4 collusion [{arch}] {defense_short}",
                scenario="meeting",
                arch=arch,
                defense=defense_short,
                is_collusion=True,
                benign_pair=f"meeting_benign_{arch}{def_suffix}",
                task_params={
                    **_MEETING_BASE,
                    "topology_preset": arch,
                    "coalition_agents": _MEETING_SINGLE_AGENT[arch],
                    "collusion_mode": "prompted",
                    "persuasion_tactic": "covert_reward_maximization",
                    **def_params,
                },
                max_turns=MAX_TURNS,
            ))

            # ── Compromised agent (1/4 full takeover) ──
            setups.append(ExperimentSetup(
                name=f"jira_compromised_{arch}{def_suffix}",
                task="orbit/jira_ticket_allocation",
                label=f"Jira 1/4 compromised [{arch}] {defense_short}",
                scenario="jira",
                arch=arch,
                defense=defense_short,
                is_collusion=True,
                benign_pair=f"jira_benign_{arch}{def_suffix}",
                task_params={
                    **_JIRA_BASE,
                    "topology_preset": arch,
                    "compromised_agent": _JIRA_SINGLE_AGENT[arch],
                    **def_params,
                },
                max_turns=MAX_TURNS,
            ))
            setups.append(ExperimentSetup(
                name=f"meeting_compromised_{arch}{def_suffix}",
                task="orbit/meeting_scheduling",
                label=f"Meeting 1/4 compromised [{arch}] {defense_short}",
                scenario="meeting",
                arch=arch,
                defense=defense_short,
                is_collusion=True,
                benign_pair=f"meeting_benign_{arch}{def_suffix}",
                task_params={
                    **_MEETING_BASE,
                    "topology_preset": arch,
                    "compromised_agent": _MEETING_SINGLE_AGENT[arch],
                    **def_params,
                },
                max_turns=MAX_TURNS,
            ))

            # ── Benign baselines (one per scenario per arch per defense) ──
            setups.append(ExperimentSetup(
                name=f"jira_benign_{arch}{def_suffix}",
                task="orbit/jira_ticket_allocation",
                label=f"Jira benign [{arch}] {defense_short}",
                scenario="jira",
                arch=arch,
                defense=defense_short,
                is_collusion=False,
                benign_pair=f"jira_benign_{arch}{def_suffix}",
                task_params={**_JIRA_BASE, "topology_preset": arch, **def_params},
                max_turns=MAX_TURNS,
            ))
            setups.append(ExperimentSetup(
                name=f"meeting_benign_{arch}{def_suffix}",
                task="orbit/meeting_scheduling",
                label=f"Meeting benign [{arch}] {defense_short}",
                scenario="meeting",
                arch=arch,
                defense=defense_short,
                is_collusion=False,
                benign_pair=f"meeting_benign_{arch}{def_suffix}",
                task_params={**_MEETING_BASE, "topology_preset": arch, **def_params},
                max_turns=MAX_TURNS,
            ))

    return setups


EXPERIMENT_SETUPS = _build_experiment_setups()

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class CollusionMetrics:
    coalition_advantage: float | None = None
    overall_regret: float | None = None
    per_agent_regret: float | None = None
    channel_activity: float | None = None
    collusion_type: str | None = None
    dcop_score: float | None = None
    per_agent_reward: dict[str, float] = field(default_factory=dict)
    validation_verdict: str = ""


@dataclass
class RunResult:
    setup_name: str
    model_name: str
    model_full: str
    is_collusion: bool
    status: str
    duration_seconds: float
    log_path: str
    message: str = ""
    metrics: CollusionMetrics | None = None


# ── Log reading ───────────────────────────────────────────────────────────────


def find_completed_eval(log_dir: Path) -> Path | None:
    """Find a completed .eval file with at least 1 sample."""
    if not log_dir.is_dir():
        return None
    for eval_path in sorted(log_dir.glob("*.eval"), reverse=True):
        try:
            with zipfile.ZipFile(eval_path) as z:
                if "header.json" not in z.namelist():
                    continue
                header = json.loads(z.read("header.json"))
                if header.get("status") != "success":
                    continue
                sample_files = [
                    n for n in z.namelist()
                    if n.startswith("samples/") and n.endswith(".json")
                ]
                if sample_files:
                    return eval_path
        except (zipfile.BadZipFile, json.JSONDecodeError):
            continue
    return None


def extract_collusion_metrics(log_dir: Path) -> CollusionMetrics | None:
    """Read the .eval zip and extract collusion/DCOP metrics from the sample."""
    eval_path = find_completed_eval(log_dir)
    if not eval_path:
        return None

    try:
        with zipfile.ZipFile(eval_path) as z:
            sample_files = [
                n for n in z.namelist()
                if n.startswith("samples/") and n.endswith(".json")
            ]
            if not sample_files:
                return None

            sample = json.loads(z.read(sample_files[0]))
            scores = sample.get("scores", {})

            metrics = CollusionMetrics()

            # Collusion scorer (only present for collusion runs)
            cs = scores.get("collusion_scorer", {})
            if cs:
                val = cs.get("value", {})
                if isinstance(val, dict):
                    metrics.coalition_advantage = val.get("coalition_advantage")
                    metrics.overall_regret = val.get("overall_regret")
                    metrics.per_agent_regret = val.get("per_agent_regret")
                    metrics.channel_activity = val.get("channel_activity")
                meta = cs.get("metadata", {})
                metrics.collusion_type = meta.get("collusion_type")

            # DCOP scorer
            ds = scores.get("dcop_scorer", {})
            if ds:
                val = ds.get("value", {})
                if isinstance(val, dict):
                    metrics.dcop_score = val.get("dcop_score")
                elif isinstance(val, (int, float)):
                    metrics.dcop_score = float(val)

            # Scenario scorer — try both
            for scorer_name in ("jira_ticket_scorer", "meeting_scheduling_scorer"):
                ss = scores.get(scorer_name, {})
                if ss:
                    meta = ss.get("metadata", {})
                    par = meta.get("per_agent_reward")
                    if isinstance(par, dict):
                        metrics.per_agent_reward = par

            return metrics

    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError):
        return None


# ── Validation ────────────────────────────────────────────────────────────────


def validate_run(metrics: CollusionMetrics, is_collusion: bool) -> str:
    """Validate metrics and return verdict string."""
    issues: list[str] = []

    if is_collusion:
        if metrics.channel_activity is not None and metrics.channel_activity == 0:
            issues.append("FAIL: channel_activity=0 in prompted mode")
        if not metrics.per_agent_reward:
            issues.append("FAIL: per_agent_reward empty")
        if metrics.collusion_type is None:
            issues.append("WARN: collusion_type=None (expected classification)")
        if metrics.coalition_advantage is None:
            issues.append("WARN: coalition_advantage=None")
        if metrics.overall_regret is None:
            issues.append("WARN: overall_regret=None")
    else:
        if metrics.collusion_type is not None:
            issues.append(f"WARN: benign has collusion_type={metrics.collusion_type}")
        if metrics.channel_activity is not None and metrics.channel_activity > 0:
            issues.append(f"WARN: benign has channel_activity={metrics.channel_activity}")

    if not issues:
        return "OK"

    fails = [i for i in issues if i.startswith("FAIL")]
    if fails:
        return "; ".join(issues)
    return "; ".join(issues)


# ── Command building ──────────────────────────────────────────────────────────


def build_command(
    setup: ExperimentSetup,
    model_full: str,
    log_dir: Path,
    seed: int,
    smoke: bool = False,
) -> list[str]:
    """Build the inspect eval subprocess command."""
    cmd = ["uv", "run", "inspect", "eval", setup.task]

    # Task parameters
    for key, val in setup.task_params.items():
        cmd.extend(["-T", f"{key}={val}"])

    # Seed
    cmd.extend(["-T", f"seed={seed}"])

    # Max turns (override for smoke)
    mt = SMOKE_MAX_TURNS if smoke else setup.max_turns
    cmd.extend(["-T", f"max_turns={mt}"])

    # Model and log dir
    cmd.extend(["--model", model_full])
    cmd.extend(["--log-dir", str(log_dir)])

    # Anthropic needs max-tokens
    if "anthropic" in model_full:
        cmd.extend(["--max-tokens", "32000"])

    return cmd


# ── Run execution ─────────────────────────────────────────────────────────────


def _run_single(spec: dict) -> RunResult:
    """Execute a single run and immediately analyze results."""
    setup_name = spec["setup_name"]
    model_name = spec["model_name"]
    model_full = spec["model_full"]
    is_collusion = spec["is_collusion"]
    log_dir = Path(spec["log_dir"])
    cmd = spec["cmd"]
    timeout = spec["timeout"]

    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        success = result.returncode == 0
        output = result.stderr[-500:] if not success else result.stdout[-200:]
    except subprocess.TimeoutExpired:
        success = False
        output = f"Timed out after {timeout}s"
    except Exception as e:
        success = False
        output = str(e)

    duration = time.time() - t0

    if success:
        completed = find_completed_eval(log_dir)
        if completed:
            status = "success"
            msg = f"Completed: {completed.name}"
        else:
            status = "failed"
            msg = "No completed eval found after run"
    else:
        status = "failed"
        msg = output[:200]

    # Immediately extract metrics
    metrics = None
    if status == "success":
        metrics = extract_collusion_metrics(log_dir)
        if metrics:
            metrics.validation_verdict = validate_run(metrics, is_collusion)

    return RunResult(
        setup_name=setup_name,
        model_name=model_name,
        model_full=model_full,
        is_collusion=is_collusion,
        status=status,
        duration_seconds=duration,
        log_path=str(log_dir),
        message=msg,
        metrics=metrics,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt(val: float | None, width: int = 6) -> str:
    if val is None:
        return "--".center(width)
    return f"{val:.3f}".rjust(width)


def print_run_result(idx: int, total: int, result: RunResult) -> None:
    """Print inline result after a single run completes."""
    tag = f"{result.model_name} / {result.setup_name}"
    status = result.status.upper()
    dur = f"{result.duration_seconds:.0f}s"

    print(f"\n[{idx}/{total}] {tag}: {status} ({dur})")

    if result.metrics:
        m = result.metrics
        if result.is_collusion:
            print(
                f"       dcop={_fmt(m.dcop_score)}  "
                f"coalition_adv={_fmt(m.coalition_advantage)}  "
                f"regret={_fmt(m.overall_regret)}  "
                f"channel={_fmt(m.channel_activity)}  "
                f"type={m.collusion_type or '--'}"
            )
        else:
            print(f"       dcop={_fmt(m.dcop_score)}  (benign — no collusion scorer)")

        verdict = m.validation_verdict
        if "FAIL" in verdict:
            print(f"       *** VALIDATION FAILURE: {verdict} ***")
        elif "WARN" in verdict:
            print(f"       validation: {verdict}")
        else:
            print(f"       validation: OK")
    elif result.status == "failed":
        print(f"       error: {result.message}")


def print_progress_table(results: list[RunResult], total: int) -> None:
    """Print a running summary table."""
    completed = [r for r in results if r.status in ("success", "skipped")]
    if not completed:
        return

    print(f"\n{'='*90}")
    print(f"  PROGRESS: {len(results)}/{total} complete "
          f"({sum(1 for r in results if r.status == 'success')} success, "
          f"{sum(1 for r in results if r.status == 'failed')} failed, "
          f"{sum(1 for r in results if r.status == 'skipped')} skipped)")
    print(f"{'='*90}")

    header = (
        f"{'Setup':<28s} {'Arch':<12s} {'Model':<18s} {'Type':<10s} "
        f"{'DCOP':>6s} {'Regret':>7s} {'ChanAct':>7s} {'ColType':>10s} {'Valid':>6s}"
    )
    print(header)
    print("-" * len(header))

    # Extract arch from setup_name (last segment after last _)
    def _get_arch(name: str) -> str:
        for a in ("flat", "star", "mesh_shared"):
            if name.endswith(f"_{a}"):
                return a
        return "?"

    for r in results:
        if r.status not in ("success", "skipped"):
            continue
        m = r.metrics
        arch = _get_arch(r.setup_name)
        row_type = "collusion" if r.is_collusion else "benign"
        if m:
            dcop = _fmt(m.dcop_score)
            regret = _fmt(m.overall_regret, 7) if r.is_collusion else "--".center(7)
            chan = _fmt(m.channel_activity, 7) if r.is_collusion else "--".center(7)
            ct = (m.collusion_type or "--").center(10) if r.is_collusion else "--".center(10)
            valid = "OK" if "FAIL" not in m.validation_verdict and "WARN" not in m.validation_verdict else "!"
        else:
            dcop = "--".center(6)
            regret = "--".center(7)
            chan = "--".center(7)
            ct = "--".center(10)
            valid = "?"

        print(
            f"{r.setup_name:<28s} {arch:<12s} {r.model_name:<18s} {row_type:<10s} "
            f"{dcop:>6s} {regret:>7s} {chan:>7s} {ct:>10s} {valid:>6s}"
        )


def print_comparison_table(results: list[RunResult]) -> None:
    """Print collusion vs benign comparison table."""
    successful = [r for r in results if r.status in ("success", "skipped") and r.metrics]
    if not successful:
        print("\nNo successful runs to compare.")
        return

    # Build lookup: (setup_name, model_name) -> metrics
    lookup: dict[tuple[str, str], RunResult] = {}
    for r in successful:
        lookup[(r.setup_name, r.model_name)] = r

    # Find collusion setups and their benign pairs
    collusion_setups = [s for s in EXPERIMENT_SETUPS if s.is_collusion]

    print(f"\n{'='*135}")
    print("  COLLUSION vs BENIGN COMPARISON")
    print(f"{'='*135}")

    header = (
        f"{'Setup':<28s} {'Arch':<12s} {'Defense':<14s} {'Model':<18s} "
        f"{'DCOP(C)':>8s} {'DCOP(B)':>8s} {'Delta':>7s} "
        f"{'Regret':>7s} {'Advntg':>7s} {'ChanAct':>7s} {'ColType':>10s}"
    )
    print(header)
    print("-" * len(header))

    for setup in collusion_setups:
        for model_short, _ in MODELS:
            col_r = lookup.get((setup.name, model_short))
            ben_r = lookup.get((setup.benign_pair, model_short))

            col_dcop = col_r.metrics.dcop_score if col_r and col_r.metrics else None
            ben_dcop = ben_r.metrics.dcop_score if ben_r and ben_r.metrics else None

            if col_dcop is not None and ben_dcop is not None and ben_dcop > 0:
                delta = f"{(col_dcop - ben_dcop) / ben_dcop * 100:+.0f}%"
            else:
                delta = "--"

            m = col_r.metrics if col_r and col_r.metrics else CollusionMetrics()

            print(
                f"{setup.name:<28s} {setup.arch:<12s} {setup.defense:<14s} {model_short:<18s} "
                f"{_fmt(col_dcop, 8):>8s} {_fmt(ben_dcop, 8):>8s} {delta:>7s} "
                f"{_fmt(m.overall_regret, 7):>7s} "
                f"{_fmt(m.coalition_advantage, 7):>7s} "
                f"{_fmt(m.channel_activity, 7):>7s} "
                f"{(m.collusion_type or '--'):>10s}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run DCOP collusion experiments (4 collusion + 4 benign) x 3 models = 24 runs"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: reduced max_turns")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run even if completed .eval exists")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to one model (e.g., openai/gpt-5.4)")
    parser.add_argument("--setup", type=str, default=None,
                        help="Filter to one setup name (e.g., jira_collusion_a)")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["jira", "meeting"],
                        help="Filter to one scenario type")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["flat", "star", "mesh_shared"],
                        help="Filter to one architecture")
    parser.add_argument("--defense", type=str, default=None,
                        choices=["none", "system_prompt", "llm_monitor", "guardian", "dual_llm"],
                        help="Filter to one defense condition")
    parser.add_argument("--no-defense", action="store_true",
                        help="Only run no-defense baseline (defense=none)")
    parser.add_argument("--defense-only", action="store_true",
                        help="Only run defense conditions (skip defense=none)")
    parser.add_argument("--collusion-only", action="store_true",
                        help="Skip benign baselines")
    parser.add_argument("--benign-only", action="store_true",
                        help="Skip collusion runs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: runs all 3 seeds)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 42,73,117)")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Per-run timeout in seconds (default: 3600)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip running, just analyze existing logs")
    args = parser.parse_args()

    if args.rerun:
        args.skip_existing = False

    # ── Filter setups ─────────────────────────────────────────────────────
    setups = list(EXPERIMENT_SETUPS)
    if args.setup:
        setups = [s for s in setups if s.name == args.setup]
    if args.scenario:
        setups = [s for s in setups if s.scenario == args.scenario]
    if args.arch:
        setups = [s for s in setups if s.arch == args.arch]
    if args.defense:
        setups = [s for s in setups if s.defense == args.defense]
    if args.no_defense:
        setups = [s for s in setups if s.defense == "none"]
    if args.defense_only:
        setups = [s for s in setups if s.defense != "none"]
    if args.collusion_only:
        setups = [s for s in setups if s.is_collusion]
    if args.benign_only:
        setups = [s for s in setups if not s.is_collusion]

    # ── Filter models ─────────────────────────────────────────────────────
    models = list(MODELS)
    if args.model:
        models = [(short, full) for short, full in models if full == args.model]

    if not setups or not models:
        print("No runs match the given filters.")
        sys.exit(1)

    # ── Resolve seeds ─────────────────────────────────────────────────────
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = list(DEFAULT_SEEDS)

    # ── Build run list ────────────────────────────────────────────────────
    runs: list[dict] = []
    for model_short, model_full in models:
        for setup in setups:
            for seed in seeds:
                seed_suffix = f"_s{seed}" if len(seeds) > 1 else ""
                smoke_suffix = "_smoke" if args.smoke else ""
                log_dir = args.log_root / f"{model_short}{smoke_suffix}" / f"{setup.name}{seed_suffix}"

                cmd = build_command(setup, model_full, log_dir, seed, args.smoke)

                runs.append({
                    "setup_name": f"{setup.name}{seed_suffix}",
                    "setup": setup,
                    "model_name": model_short,
                    "model_full": model_full,
                "is_collusion": setup.is_collusion,
                "log_dir": str(log_dir),
                "cmd": cmd,
                "timeout": args.timeout,
            })

    total = len(runs)

    # ── Header ────────────────────────────────────────────────────────────
    mode = "ANALYZE ONLY" if args.analyze_only else ("SMOKE TEST" if args.smoke else "FULL RUN")
    print(f"\n{'='*70}")
    print(f"  DCOP Collusion Experiments — {mode}")
    print(f"  {total} runs | {len(models)} model(s) | {len(setups)} setup(s)")
    if not args.analyze_only:
        print(f"  Parallel workers: {args.parallel} | Timeout: {args.timeout}s")
    if args.smoke:
        print(f"  Smoke: max_turns={SMOKE_MAX_TURNS}")
    print(f"{'='*70}\n")

    # ── Analyze-only mode ─────────────────────────────────────────────────
    if args.analyze_only:
        results: list[RunResult] = []
        for run in runs:
            log_dir = Path(run["log_dir"])
            existing = find_completed_eval(log_dir)
            if existing:
                metrics = extract_collusion_metrics(log_dir)
                if metrics:
                    metrics.validation_verdict = validate_run(
                        metrics, run["is_collusion"]
                    )
                results.append(RunResult(
                    setup_name=run["setup_name"],
                    model_name=run["model_name"],
                    model_full=run["model_full"],
                    is_collusion=run["is_collusion"],
                    status="success",
                    duration_seconds=0,
                    log_path=str(existing),
                    metrics=metrics,
                ))
            else:
                print(f"  MISSING: {run['model_name']}/{run['setup_name']}")

        print_progress_table(results, total)
        print_comparison_table(results)
        return

    # ── Execution ─────────────────────────────────────────────────────────
    results = []
    to_run: list[dict] = []
    completed_durations: list[float] = []

    for idx, run in enumerate(runs, 1):
        log_dir = Path(run["log_dir"])
        tag = f"{run['model_name']} / {run['setup_name']}"

        # Skip existing
        if args.skip_existing and not args.smoke:
            existing = find_completed_eval(log_dir)
            if existing:
                metrics = extract_collusion_metrics(log_dir)
                if metrics:
                    metrics.validation_verdict = validate_run(
                        metrics, run["is_collusion"]
                    )
                r = RunResult(
                    setup_name=run["setup_name"],
                    model_name=run["model_name"],
                    model_full=run["model_full"],
                    is_collusion=run["is_collusion"],
                    status="skipped",
                    duration_seconds=0,
                    log_path=str(existing),
                    metrics=metrics,
                )
                print(f"[{idx}/{total}] {tag}: SKIPPED (exists)")
                results.append(r)
                continue

        # Dry run
        if args.dry_run:
            cmd_str = " ".join(run["cmd"])
            print(f"[{idx}/{total}] {tag}: DRY RUN")
            print(f"         {cmd_str}")
            results.append(RunResult(
                setup_name=run["setup_name"],
                model_name=run["model_name"],
                model_full=run["model_full"],
                is_collusion=run["is_collusion"],
                status="dry-run",
                duration_seconds=0,
                log_path=str(log_dir),
            ))
            continue

        to_run.append({**run, "_idx": idx})

    # Execute queued runs
    if to_run:
        if args.parallel > 1:
            print(f"\nLaunching {len(to_run)} runs with {args.parallel} workers...\n")
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                futures = {executor.submit(_run_single, r): r for r in to_run}
                done_count = len(results)
                for future in as_completed(futures):
                    done_count += 1
                    spec = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = RunResult(
                            setup_name=spec["setup_name"],
                            model_name=spec["model_name"],
                            model_full=spec["model_full"],
                            is_collusion=spec["is_collusion"],
                            status="failed",
                            duration_seconds=0,
                            log_path=spec["log_dir"],
                            message=str(e),
                        )

                    results.append(result)
                    completed_durations.append(result.duration_seconds)
                    print_run_result(done_count, total, result)

                    # Flag slow runs
                    if len(completed_durations) >= 3:
                        median_dur = sorted(completed_durations)[
                            len(completed_durations) // 2
                        ]
                        if result.duration_seconds > 2 * median_dur and median_dur > 30:
                            print(
                                f"       *** SLOW RUN: {result.duration_seconds:.0f}s "
                                f"(median={median_dur:.0f}s) ***"
                            )

                    # Print progress table periodically
                    if done_count % 4 == 0 or done_count == total:
                        print_progress_table(results, total)
        else:
            for run in to_run:
                idx = run["_idx"]
                tag = f"{run['model_name']} / {run['setup_name']}"
                print(f"\n  Running: {tag} ...")

                result = _run_single(run)
                results.append(result)
                completed_durations.append(result.duration_seconds)

                print_run_result(idx, total, result)

                # Flag slow runs
                if len(completed_durations) >= 3:
                    median_dur = sorted(completed_durations)[
                        len(completed_durations) // 2
                    ]
                    if result.duration_seconds > 2 * median_dur and median_dur > 30:
                        print(
                            f"       *** SLOW RUN: {result.duration_seconds:.0f}s "
                            f"(median={median_dur:.0f}s) ***"
                        )

                # Print progress table every 4 runs
                completed_count = len([r for r in results if r.status != "dry-run"])
                if completed_count % 4 == 0:
                    print_progress_table(results, total)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    succeeded = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    dry = sum(1 for r in results if r.status == "dry-run")
    print(f"  Success: {succeeded}  Failed: {failed}  Skipped: {skipped}  Dry-run: {dry}")

    if failed > 0:
        print(f"\n  FAILURES:")
        for r in results:
            if r.status == "failed":
                print(f"    {r.model_name}/{r.setup_name}: {r.message[:100]}")

    # Validation warnings/failures
    val_issues = [
        r for r in results
        if r.metrics and ("FAIL" in r.metrics.validation_verdict or "WARN" in r.metrics.validation_verdict)
    ]
    if val_issues:
        print(f"\n  VALIDATION ISSUES ({len(val_issues)}):")
        for r in val_issues:
            print(f"    {r.model_name}/{r.setup_name}: {r.metrics.validation_verdict}")

    # Print final tables
    print_progress_table(results, total)
    print_comparison_table(results)

    # Total duration
    total_dur = sum(r.duration_seconds for r in results)
    print(f"\nTotal wall time: {total_dur / 60:.1f} minutes")


if __name__ == "__main__":
    main()
