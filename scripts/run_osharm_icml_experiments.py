#!/usr/bin/env python3
"""
Batch runner for ICML OS-Harm experiments (paper-matching conditions).

Runs 13 conditions x 1 dataset (osharm_misuse) = 13 eval runs.
Supports parallel execution, skip-existing, and multi-seed runs.

Paper: "Architecture Matters for Multi-Agent Security"
  - Table 1: Role Distribution (7 conditions)
  - Table 2: Communication Topology (4 conditions, 2 shared with T1)
  - Table 3: Memory & State Visibility (6 conditions, 2 shared with T2)
  - 13 unique conditions total

Usage:
    python scripts/run_osharm_icml_experiments.py --dry-run
    python scripts/run_osharm_icml_experiments.py --smoke --model openai/gpt-4o
    python scripts/run_osharm_icml_experiments.py --model openai/gpt-4o
    python scripts/run_osharm_icml_experiments.py --condition single_agent
    python scripts/run_osharm_icml_experiments.py --model openai/gpt-4o --parallel 3 --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# ── Experiment matrix (13 conditions matching ICML paper) ─────────────────────

EXPERIMENTS = [
    # (dir_name, condition_code, paper_name, tables)
    # All specialist conditions use tool-action specialists (click, type, key,
    # scroll) to match BrowserART's action-type decomposition.
    #
    # Table 1: Role Distribution
    ("single_agent",              "single_agent",              "Standalone Agent",                    [1, 2]),
    ("star_batch_relaxed",        "star_batch_relaxed",         "Star + Single Executor",              [1]),
    ("star_batch",                "star_batch",                 "Star + Batch Executor",               [1]),
    ("star_step",                 "star_step",                  "Star + Stepwise",                     [1]),
    ("star_tool_2_specialist",    "star_tool_2_specialist",     "Star + 2 Specialist",                 [1]),
    ("star_tool_3_specialist",    "star_tool_3_specialist",     "Star + 3 Specialist",                 [1]),
    ("star_tool_specialist",      "star_tool_specialist",       "Star + Specialized",                  [1, 2, 3]),
    # Table 2: Communication Topology
    ("mesh_tool_round_robin",     "mesh_tool_round_robin",      "Chain + Specialized",                 [2]),
    ("mesh_tool_delegation",      "mesh_tool_delegation",       "Mesh + Specialized",                  [2, 3]),
    # Table 3: Memory & State Visibility
    ("tool_memory_own_reasoning", "tool_memory_own_reasoning",  "Star + Specialized + CoT",            [3]),
    ("tool_memory_full",          "tool_memory_full",           "Star + Specialized + Shared Memory",  [3]),
    ("mesh_tool_delegation_cot",  "mesh_tool_delegation_cot",   "Mesh + Specialized + CoT",            [3]),
    ("mesh_tool_delegation_full", "mesh_tool_delegation_full",  "Mesh + Specialized + Shared Memory",  [3]),
]

# OS-Harm datasets.  osharm_misuse is the primary harmful dataset (48 tasks
# after default chrome exclusion).  Add more entries here if you also want
# benign OSWorld benchmark runs.
DATASETS = [
    ("osharm_misuse", "misuse", 44),
]

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_JUDGE_MODEL = "openai/gpt-4.1"
DEFAULT_SEED = 42
DEFAULT_MAX_TURNS = 100
DEFAULT_MAX_TIME = 1800      # seconds per task (30 min)
DEFAULT_MAX_SAMPLES = None   # Inspect parallel samples (None = Inspect default)
DEFAULT_RETRY_ON_ERROR = 1
DEFAULT_LOG_ROOT = Path("logs/osharm_icml")

SMOKE_LIMIT = 3
SMOKE_MAX_TURNS = 15
SMOKE_MAX_TIME = 120

# Per-condition subprocess timeout.  44 samples × 1800s max_time = 22h
# theoretical max, but most samples finish well under the limit; 4h is generous.
RUN_TIMEOUT_SECONDS = 14400


@dataclass
class RunResult:
    experiment: str
    condition: str
    dataset: str
    dataset_label: str
    status: str  # success, failed, skipped, dry-run
    sample_count: int
    duration_seconds: float
    log_path: str
    message: str = ""
    seed: int = 42


def find_completed_eval(log_dir: Path, expected_samples: int) -> Path | None:
    """Check if log_dir contains a completed eval with expected sample count."""
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
                if len(sample_files) >= expected_samples:
                    return eval_path
        except (zipfile.BadZipFile, json.JSONDecodeError):
            continue
    return None


def validate_eval(eval_path: Path, dataset: str, expected_samples: int) -> tuple[bool, str]:
    """Validate a completed eval log. Returns (ok, message)."""
    try:
        with zipfile.ZipFile(eval_path) as z:
            header = json.loads(z.read("header.json"))
            if header.get("status") != "success":
                return False, f"status={header.get('status')}"

            sample_files = [
                n for n in z.namelist()
                if n.startswith("samples/") and n.endswith(".json")
            ]
            if len(sample_files) < expected_samples:
                return False, f"only {len(sample_files)}/{expected_samples} samples"

            # Spot-check a few samples for scorer results
            errors = []
            for sf in sample_files[:5]:
                sample = json.loads(z.read(sf))
                scores = sample.get("scores", {})
                if "osworld_scorer" not in scores:
                    errors.append(f"{sf}: missing osworld_scorer")

            if errors:
                return False, "; ".join(errors[:3])

            return True, f"{len(sample_files)} samples, status=success"
    except Exception as e:
        return False, str(e)


def run_experiment(
    condition: str,
    dataset: str,
    log_dir: Path,
    model: str,
    judge_model: str,
    seed: int,
    max_turns: int,
    max_time: float,
    retry_on_error: int,
    limit: int | None = None,
    max_samples: int | None = None,
) -> tuple[bool, str]:
    """Run a single experiment via subprocess. Returns (success, output)."""
    # --time-limit is Inspect's per-sample wall-clock timeout (seconds).
    # Set it above max_time to allow for scoring overhead after the agent
    # finishes, but low enough to kill truly hung samples.
    sample_timeout = int(max_time) + 300
    cmd = [
        "uv", "run", "orbit", "osworld",
        "--model", model,
        "--dataset", dataset,
        "--condition", condition,
        "--seed", str(seed),
        "--max-turns", str(max_turns),
        "--max-time", str(max_time),
        "--judge-model", judge_model,
        "--retry-on-error", str(retry_on_error),
        "--time-limit", str(sample_timeout),
        "--log-dir", str(log_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return False, f"Exit code {result.returncode}\n{result.stderr[-500:]}"
        return True, result.stdout[-200:]
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {RUN_TIMEOUT_SECONDS // 3600}h"
    except Exception as e:
        return False, str(e)


def _run_single(run_spec: dict) -> RunResult:
    """Execute a single run (used by both sequential and parallel modes)."""
    exp_name = run_spec["exp_name"]
    condition = run_spec["condition"]
    dataset = run_spec["dataset"]
    label = run_spec["label"]
    expected_count = run_spec["expected_count"]
    log_dir = Path(run_spec["log_dir"])
    model = run_spec["model"]
    judge_model = run_spec["judge_model"]
    seed = run_spec["seed"]
    max_turns = run_spec["max_turns"]
    max_time = run_spec["max_time"]
    retry_on_error = run_spec["retry_on_error"]
    limit = run_spec.get("limit")
    max_samples = run_spec.get("max_samples")
    effective_expected = limit if limit else expected_count

    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    success, output = run_experiment(
        condition=condition,
        dataset=dataset,
        log_dir=log_dir,
        model=model,
        judge_model=judge_model,
        seed=seed,
        max_turns=max_turns,
        max_time=max_time,
        retry_on_error=retry_on_error,
        limit=limit,
        max_samples=max_samples,
    )
    duration = time.time() - t0

    if success:
        completed = find_completed_eval(log_dir, effective_expected)
        if completed:
            ok, msg = validate_eval(completed, dataset, effective_expected)
            status = "success" if ok else "failed"
        else:
            status = "failed"
            msg = "No completed eval found after run"
    else:
        status = "failed"
        msg = output[:200]

    return RunResult(
        experiment=exp_name, condition=condition,
        dataset=dataset, dataset_label=label,
        status=status, sample_count=effective_expected,
        duration_seconds=duration,
        log_path=str(log_dir),
        message=msg,
        seed=seed,
    )


def print_progress(idx: int, total: int, exp_name: str, dataset_label: str,
                   condition: str, expected: int, action: str):
    """Print progress line."""
    print(
        f"[{idx}/{total}] {exp_name}/{dataset_label}: "
        f"{condition} ({expected} samples) — {action}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run ICML OS-Harm experiments (13 conditions, paper-matching)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Smoke test: {SMOKE_LIMIT} samples per condition, "
             f"max-turns={SMOKE_MAX_TURNS}, max-time={SMOKE_MAX_TIME}s",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip runs with existing completed evals (default: True)",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run even if completed eval exists",
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Run only this condition (e.g., single_agent)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=[d[0] for d in DATASETS],
        help="Run only this dataset",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model for safety scorer (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[DEFAULT_SEED],
        help=f"Seed(s) to run (default: {DEFAULT_SEED}). "
             "Pass multiple for mean±std: --seed 42 43 44",
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help=f"Max agent turns per task (default: {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument(
        "--max-time", type=float, default=DEFAULT_MAX_TIME,
        help=f"Max wall-clock seconds per task (default: {DEFAULT_MAX_TIME})",
    )
    parser.add_argument(
        "--retry-on-error", type=int, default=DEFAULT_RETRY_ON_ERROR,
    )
    parser.add_argument(
        "--log-root", type=Path, default=DEFAULT_LOG_ROOT,
        help=f"Root log directory (default: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Max parallel condition workers (default: 1 = sequential). "
             "Use with --max-samples to control per-condition parallelism, "
             "e.g. --parallel 3 --max-samples 10 runs 3 conditions at once "
             "with 10 Docker containers each.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help="Max parallel samples per condition (passed to Inspect's "
             "--max-samples). Controls Docker container concurrency within "
             "each eval run. Default: None (Inspect default).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples per run (overrides --smoke limit)",
    )
    args = parser.parse_args()

    if args.rerun:
        args.skip_existing = False

    seeds = args.seed  # list of ints

    # Build run list
    runs: list[dict] = []
    for dir_name, condition, paper_name, tables in EXPERIMENTS:
        if args.condition and condition != args.condition:
            continue
        for dataset, label, count in DATASETS:
            if args.dataset and dataset != args.dataset:
                continue
            for seed in seeds:
                runs.append({
                    "exp_name": dir_name,
                    "condition": condition,
                    "paper_name": paper_name,
                    "dataset": dataset,
                    "label": label,
                    "expected_count": count,
                    "seed": seed,
                })

    if not runs:
        print("No runs match the given filters.")
        sys.exit(1)

    # Determine params
    limit = args.limit if args.limit is not None else (SMOKE_LIMIT if args.smoke else None)
    max_turns = SMOKE_MAX_TURNS if args.smoke else args.max_turns
    max_time = SMOKE_MAX_TIME if args.smoke else args.max_time
    smoke_suffix = "_smoke" if args.smoke else ""
    log_root = args.log_root.parent / f"{args.log_root.name}{smoke_suffix}" if args.smoke else args.log_root

    # Compute log dirs and populate run specs
    for run in runs:
        seed_suffix = f"seed_{run['seed']}" if len(seeds) > 1 else ""
        if seed_suffix:
            run["log_dir"] = str(log_root / run["exp_name"] / run["label"] / seed_suffix)
        else:
            run["log_dir"] = str(log_root / run["exp_name"] / run["label"])
        run["model"] = args.model
        run["judge_model"] = args.judge_model
        run["max_turns"] = max_turns
        run["max_time"] = max_time
        run["retry_on_error"] = args.retry_on_error
        run["limit"] = limit
        run["max_samples"] = args.max_samples

    total = len(runs)
    results: list[RunResult] = []

    print(f"\n{'='*60}")
    mode = "SMOKE TEST" if args.smoke else "FULL RUN"
    seed_str = ", ".join(str(s) for s in seeds)
    print(f"  ICML OS-Harm Experiments — {mode}")
    print(f"  {total} runs, model={args.model}, seed(s)={seed_str}")
    print(f"  Judge: {args.judge_model}")
    print(f"  Max turns: {max_turns}, max time: {max_time}s/task")
    if args.max_samples:
        print(f"  Max parallel samples per condition: {args.max_samples}")
    print(f"  Parallel workers: {args.parallel}")
    if args.smoke:
        print(f"  Smoke: {SMOKE_LIMIT} samples/run, "
              f"max_turns={SMOKE_MAX_TURNS}, max_time={SMOKE_MAX_TIME}s")
    print(f"  Log root: {log_root}")
    print(f"{'='*60}\n")

    # Check for skips first
    to_run: list[dict] = []
    for idx, run in enumerate(runs, 1):
        effective_expected = limit if limit else run["expected_count"]
        log_dir = Path(run["log_dir"])

        if args.skip_existing and not args.smoke:
            existing = find_completed_eval(log_dir, effective_expected)
            if existing:
                ok, msg = validate_eval(existing, run["dataset"], effective_expected)
                if ok:
                    print_progress(idx, total, run["exp_name"], run["label"],
                                   run["condition"], effective_expected,
                                   f"SKIPPED (exists: {existing.name})")
                    results.append(RunResult(
                        experiment=run["exp_name"], condition=run["condition"],
                        dataset=run["dataset"], dataset_label=run["label"],
                        status="skipped", sample_count=effective_expected,
                        duration_seconds=0, log_path=str(existing),
                        message=msg, seed=run["seed"],
                    ))
                    continue

        if args.dry_run:
            print_progress(idx, total, run["exp_name"], run["label"],
                           run["condition"], effective_expected,
                           "DRY RUN (would execute)")
            results.append(RunResult(
                experiment=run["exp_name"], condition=run["condition"],
                dataset=run["dataset"], dataset_label=run["label"],
                status="dry-run", sample_count=effective_expected,
                duration_seconds=0, log_path=str(log_dir),
                seed=run["seed"],
            ))
            continue

        to_run.append(run)

    # Execute runs (parallel or sequential)
    if to_run:
        if args.parallel > 1:
            print(f"\nLaunching {len(to_run)} runs with {args.parallel} parallel workers...\n")
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                future_to_run = {
                    executor.submit(_run_single, run): run for run in to_run
                }
                for future in as_completed(future_to_run):
                    run = future_to_run[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = RunResult(
                            experiment=run["exp_name"], condition=run["condition"],
                            dataset=run["dataset"], dataset_label=run["label"],
                            status="failed", sample_count=run["expected_count"],
                            duration_seconds=0, log_path=run["log_dir"],
                            message=str(e), seed=run["seed"],
                        )
                    status_icon = "OK" if result.status == "success" else "FAIL"
                    print(f"  [{status_icon}] {result.experiment}/{result.dataset_label} "
                          f"(seed={result.seed}, {result.duration_seconds:.0f}s): {result.message[:80]}")
                    results.append(result)
        else:
            for run in to_run:
                effective_expected = limit if limit else run["expected_count"]
                print(f"  Running {run['exp_name']}/{run['label']} "
                      f"(seed={run['seed']}, {effective_expected} samples)...")
                result = _run_single(run)
                status_icon = "OK" if result.status == "success" else "FAIL"
                print(f"    -> [{status_icon}] ({result.duration_seconds:.0f}s): {result.message[:80]}")
                results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    succeeded = [r for r in results if r.status == "success"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    dry_runs = [r for r in results if r.status == "dry-run"]

    print(f"Total:     {len(results)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Skipped:   {len(skipped)}")
    print(f"Failed:    {len(failed)}")
    if dry_runs:
        print(f"Dry-run:   {len(dry_runs)}")

    total_time = sum(r.duration_seconds for r in results)
    print(f"Total time: {total_time/60:.1f} min")

    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  {r.experiment}/{r.dataset_label} (seed={r.seed}): {r.message[:100]}")

    # Print table
    print(f"\n{'Experiment':<25} {'Dataset':<10} {'Seed':<6} {'Status':<10} {'Samples':<8} {'Time':<8}")
    print("-" * 70)
    for r in results:
        t = f"{r.duration_seconds:.0f}s" if r.duration_seconds > 0 else "-"
        print(f"{r.experiment:<25} {r.dataset_label:<10} {r.seed:<6} {r.status:<10} {r.sample_count:<8} {t:<8}")

    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
