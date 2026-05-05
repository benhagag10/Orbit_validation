#!/usr/bin/env python3
"""
Batch runner for ICML RedCode-Gen experiments (13 conditions).

Runs 13 conditions x 160 tasks (all malware categories) = 13 eval runs.
Supports parallel execution, skip-existing, and multi-seed runs.

Usage:
    python scripts/run_redcode_gen_icml_experiments.py --dry-run
    python scripts/run_redcode_gen_icml_experiments.py --smoke --model openai/gpt-4o
    python scripts/run_redcode_gen_icml_experiments.py --model openai/gpt-4o
    python scripts/run_redcode_gen_icml_experiments.py --condition single_agent --model openai/gpt-4o
    python scripts/run_redcode_gen_icml_experiments.py --model openai/gpt-4o --parallel 3
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
    # Table 1: Role Distribution
    ("single_agent",          "single_agent",          "Standalone Agent",                   [1, 2]),
    ("star_batch_relaxed",    "star_batch_relaxed",    "Star + Single Executor",             [1]),
    ("star_batch",            "star_batch",            "Star + Batch Executor",              [1]),
    ("star_step",             "star_step",             "Star + Stepwise",                    [1]),
    ("star_2_specialist",     "star_2_specialist",     "Star + 2 Specialist",                [1]),
    ("star_3_specialist",     "star_3_specialist",     "Star + 3 Specialist",                [1]),
    ("star_specialist",       "star_specialist",       "Star + Specialized",                 [1, 2, 3]),
    # Table 2: Communication Topology
    ("mesh_round_robin",      "mesh_round_robin",      "Chain + Specialized",                [2]),
    ("mesh_delegation",       "mesh_delegation",       "Mesh + Specialized",                 [2, 3]),
    # Table 3: Memory & State Visibility
    ("memory_own_reasoning",  "memory_own_reasoning",  "Star + Specialized + CoT",           [3]),
    ("memory_full",           "memory_full",           "Star + Specialized + Shared Memory", [3]),
    ("mesh_delegation_cot",   "mesh_delegation_cot",   "Mesh + Specialized + CoT",           [3]),
    ("mesh_delegation_full",  "mesh_delegation_full",  "Mesh + Specialized + Shared Memory", [3]),
]

# All 160 tasks (8 categories x 20)
TOTAL_TASKS = 160

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_JUDGE_MODEL = "openai/gpt-4.1"
DEFAULT_SEED = 42
DEFAULT_MAX_TURNS = 100
DEFAULT_MAX_TIME = 300.0   # seconds per task (5 min)
DEFAULT_MAX_SAMPLES = 8    # parallel samples per condition
DEFAULT_RETRY_ON_ERROR = 1
DEFAULT_LOG_ROOT = Path("logs/redcode_gen_icml")

SMOKE_LIMIT = 3
SMOKE_MAX_TURNS = 10
SMOKE_MAX_TIME = 120.0

# Per-condition subprocess timeout.
RUN_TIMEOUT_SECONDS = 7200  # 2h generous


@dataclass
class RunResult:
    experiment: str
    condition: str
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


def validate_eval(eval_path: Path, expected_samples: int) -> tuple[bool, str]:
    """Validate a completed eval log."""
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
                if "redcode_gen_scorer" not in scores:
                    errors.append(f"{sf}: missing redcode_gen_scorer")

            if errors:
                return False, "; ".join(errors[:3])

            return True, f"{len(sample_files)} samples, status=success"
    except Exception as e:
        return False, str(e)


def run_experiment(
    condition: str,
    log_dir: Path,
    model: str,
    judge_model: str,
    seed: int,
    max_turns: int,
    max_time: float,
    retry_on_error: int,
    limit: int | None = None,
    max_samples: int | None = None,
    categories: str | None = None,
) -> tuple[bool, str]:
    """Run a single experiment via subprocess."""
    cmd = [
        "uv", "run", "orbit", "redcode-gen",
        "--model", model,
        "--condition", condition,
        "--seed", str(seed),
        "--max-turns", str(max_turns),
        "--max-time", str(max_time),
        "--judge-model", judge_model,
        "--retry-on-error", str(retry_on_error),
        "--log-dir", str(log_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if categories is not None:
        cmd.extend(["--categories", categories])

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
    """Execute a single run."""
    exp_name = run_spec["exp_name"]
    condition = run_spec["condition"]
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
    categories = run_spec.get("categories")
    effective_expected = limit if limit else expected_count

    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    success, output = run_experiment(
        condition=condition,
        log_dir=log_dir,
        model=model,
        judge_model=judge_model,
        seed=seed,
        max_turns=max_turns,
        max_time=max_time,
        retry_on_error=retry_on_error,
        limit=limit,
        max_samples=max_samples,
        categories=categories,
    )
    duration = time.time() - t0

    if success:
        completed = find_completed_eval(log_dir, effective_expected)
        if completed:
            ok, msg = validate_eval(completed, effective_expected)
            status = "success" if ok else "failed"
        else:
            status = "failed"
            msg = "No completed eval found after run"
    else:
        status = "failed"
        msg = output[:200]

    return RunResult(
        experiment=exp_name, condition=condition,
        status=status, sample_count=effective_expected,
        duration_seconds=duration,
        log_path=str(log_dir),
        message=msg,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run ICML RedCode-Gen experiments (13 conditions)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Smoke test: {SMOKE_LIMIT} samples per condition",
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
        "--categories", type=str, default=None,
        help="Comma-separated malware categories (e.g. spyware,ransomware)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[DEFAULT_SEED],
        help=f"Seed(s) (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help=f"Max agent turns per task (default: {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument(
        "--max-time", type=float, default=DEFAULT_MAX_TIME,
        help=f"Max seconds per task (default: {DEFAULT_MAX_TIME})",
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
        help="Max parallel condition workers (default: 1)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help="Max parallel samples per condition.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples per run",
    )
    args = parser.parse_args()

    if args.rerun:
        args.skip_existing = False

    seeds = args.seed

    # Build run list
    runs: list[dict] = []
    for dir_name, condition, paper_name, tables in EXPERIMENTS:
        if args.condition and condition != args.condition:
            continue
        for seed in seeds:
            runs.append({
                "exp_name": dir_name,
                "condition": condition,
                "paper_name": paper_name,
                "expected_count": TOTAL_TASKS,
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

    # Populate run specs
    for run in runs:
        seed_suffix = f"seed_{run['seed']}" if len(seeds) > 1 else ""
        if seed_suffix:
            run["log_dir"] = str(log_root / run["exp_name"] / seed_suffix)
        else:
            run["log_dir"] = str(log_root / run["exp_name"])
        run["model"] = args.model
        run["judge_model"] = args.judge_model
        run["max_turns"] = max_turns
        run["max_time"] = max_time
        run["retry_on_error"] = args.retry_on_error
        run["limit"] = limit
        run["max_samples"] = args.max_samples
        run["categories"] = args.categories

    total = len(runs)
    results: list[RunResult] = []

    print(f"\n{'='*60}")
    mode = "SMOKE TEST" if args.smoke else "FULL RUN"
    seed_str = ", ".join(str(s) for s in seeds)
    print(f"  ICML RedCode-Gen Experiments — {mode}")
    print(f"  {total} runs, model={args.model}, seed(s)={seed_str}")
    print(f"  Judge: {args.judge_model}")
    print(f"  Max turns: {max_turns}, max time: {max_time}s/task")
    print(f"  Parallel workers: {args.parallel}")
    if args.smoke:
        print(f"  Smoke: {SMOKE_LIMIT} samples/run")
    print(f"  Log root: {log_root}")
    print(f"{'='*60}\n")

    # Check for skips
    to_run: list[dict] = []
    for idx, run in enumerate(runs, 1):
        effective_expected = limit if limit else run["expected_count"]
        log_dir = Path(run["log_dir"])

        if args.skip_existing and not args.smoke:
            existing = find_completed_eval(log_dir, effective_expected)
            if existing:
                ok, msg = validate_eval(existing, effective_expected)
                if ok:
                    print(f"[{idx}/{total}] {run['exp_name']}: SKIPPED (exists)")
                    results.append(RunResult(
                        experiment=run["exp_name"], condition=run["condition"],
                        status="skipped", sample_count=effective_expected,
                        duration_seconds=0, log_path=str(existing),
                        message=msg, seed=run["seed"],
                    ))
                    continue

        if args.dry_run:
            print(f"[{idx}/{total}] {run['exp_name']}: DRY RUN")
            results.append(RunResult(
                experiment=run["exp_name"], condition=run["condition"],
                status="dry-run", sample_count=effective_expected,
                duration_seconds=0, log_path=str(log_dir),
                seed=run["seed"],
            ))
            continue

        to_run.append(run)

    # Execute runs
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
                            status="failed", sample_count=run["expected_count"],
                            duration_seconds=0, log_path=run["log_dir"],
                            message=str(e), seed=run["seed"],
                        )
                    status_icon = "OK" if result.status == "success" else "FAIL"
                    print(f"  [{status_icon}] {result.experiment} "
                          f"(seed={result.seed}, {result.duration_seconds:.0f}s): "
                          f"{result.message[:80]}")
                    results.append(result)
        else:
            for run in to_run:
                effective_expected = limit if limit else run["expected_count"]
                print(f"  Running {run['exp_name']} "
                      f"(seed={run['seed']}, {effective_expected} samples)...")
                result = _run_single(run)
                status_icon = "OK" if result.status == "success" else "FAIL"
                print(f"    -> [{status_icon}] ({result.duration_seconds:.0f}s): "
                      f"{result.message[:80]}")
                results.append(result)

    # ─��� Summary ─��────────────────────────────────────────────────────────
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
            print(f"  {r.experiment} (seed={r.seed}): {r.message[:100]}")

    # Print table
    print(f"\n{'Experiment':<25} {'Seed':<6} {'Status':<10} {'Samples':<8} {'Time':<8}")
    print("-" * 60)
    for r in results:
        t = f"{r.duration_seconds:.0f}s" if r.duration_seconds > 0 else "-"
        print(f"{r.experiment:<25} {r.seed:<6} {r.status:<10} {r.sample_count:<8} {t:<8}")

    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
