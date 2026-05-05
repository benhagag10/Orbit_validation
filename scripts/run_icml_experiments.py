#!/usr/bin/env python3
"""
Batch runner for ICML BrowserART experiments (GPT-4o, paper-matching).

Runs 9 conditions x 2 datasets (hbb, hbb_benign) = 18 eval runs.
Supports parallel execution, skip-existing, and multi-seed runs.

Paper: "Architecture Matters for Multi-Agent Security"
  - Table 1: Role Distribution (5 conditions)
  - Table 2: Communication Topology (3 conditions)
  - Table 3: Memory & State Visibility (4 conditions)
  - 9 unique conditions total

Usage:
    python scripts/run_icml_experiments.py [--smoke] [--dry-run] [--skip-existing]
    python scripts/run_icml_experiments.py --condition single_agent --dataset hbb_benign
    python scripts/run_icml_experiments.py --harmful-only
    python scripts/run_icml_experiments.py --parallel 4
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

# ── Experiment matrix (13 conditions matching paper) ──────────────────────────

EXPERIMENTS = [
    # (dir_name, condition_code, paper_name, tables)
    # Table 1: Role Distribution
    ("single_agent",          "single_agent",          "Standalone Agent",                    [1, 2]),
    ("star_batch_relaxed",    "star_batch_relaxed",     "Star + Single Executor",              [1]),
    ("star_batch",            "star_batch",             "Star + Batch Executor",               [1]),
    ("star_step",             "star_step",              "Star + Stepwise",                     [1]),
    ("star_2_specialist",     "star_2_specialist",      "Star + 2 Specialist",                 [1]),
    ("star_3_specialist",     "star_3_specialist",      "Star + 3 Specialist",                 [1]),
    ("star_specialist",       "star_specialist",        "Star + Specialized",                  [1, 2]),
    # Table 2: Communication Topology
    ("mesh_round_robin",      "mesh_round_robin",       "Chain + Specialized",                 [2]),
    ("mesh_delegation",       "mesh_delegation",        "Mesh + Specialized",                  [2]),
    # Table 3: Memory & State Visibility
    ("memory_own_reasoning",  "memory_own_reasoning",   "Star + Specialized + CoT",            [3]),
    ("memory_full",           "memory_full",            "Star + Specialized + Shared Memory",  [3]),
    ("mesh_delegation_cot",   "mesh_delegation_cot",    "Mesh + Specialized + CoT",            [3]),
    ("mesh_delegation_full",  "mesh_delegation_full",   "Mesh + Specialized + Shared Memory",  [3]),
]

DATASETS = [
    ("hbb_benign", "benign", 59),
    ("hbb",        "harmful", 100),
]

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_CLASSIFIER = "openai/gpt-4o"
DEFAULT_SEED = 42
DEFAULT_MAX_TURNS = 15
DEFAULT_RETRY_ON_ERROR = 1
DEFAULT_LOG_ROOT = Path("logs/icml_gpt4o")

SMOKE_LIMIT = 3
SMOKE_MAX_TURNS = 5


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
                if "browserart_scorer" not in scores:
                    errors.append(f"{sf}: missing browserart_scorer")

                if dataset == "hbb_benign":
                    ba = scores.get("browserart_scorer", {})
                    val = ba.get("value", {})
                    if isinstance(val, dict):
                        as_val = val.get("attack_success", 0.0)
                        if as_val > 0:
                            errors.append(f"{sf}: benign AS={as_val}")

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
    classifier: str,
    seed: int,
    max_turns: int,
    retry_on_error: int,
    limit: int | None = None,
) -> tuple[bool, str]:
    """Run a single experiment via subprocess. Returns (success, output)."""
    cmd = [
        "uv", "run", "orbit", "browserart",
        "--model", model,
        "--dataset", dataset,
        "--condition", condition,
        "--seed", str(seed),
        "--max-turns", str(max_turns),
        "--classifier-model", classifier,
        "--retry-on-error", str(retry_on_error),
        "--log-dir", str(log_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per run
        )
        if result.returncode != 0:
            return False, f"Exit code {result.returncode}\n{result.stderr[-500:]}"
        return True, result.stdout[-200:]
    except subprocess.TimeoutExpired:
        return False, "Timed out after 2 hours"
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
    classifier = run_spec["classifier"]
    seed = run_spec["seed"]
    max_turns = run_spec["max_turns"]
    retry_on_error = run_spec["retry_on_error"]
    limit = run_spec.get("limit")
    effective_expected = limit if limit else expected_count

    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    success, output = run_experiment(
        condition=condition,
        dataset=dataset,
        log_dir=log_dir,
        model=model,
        classifier=classifier,
        seed=seed,
        max_turns=max_turns,
        retry_on_error=retry_on_error,
        limit=limit,
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
        description="Run ICML BrowserART experiments (GPT-4o, paper-matching)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Smoke test: {SMOKE_LIMIT} samples per condition, "
             f"max-turns={SMOKE_MAX_TURNS}",
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
        choices=["hbb", "hbb_benign"],
        help="Run only this dataset",
    )
    parser.add_argument(
        "--benign-only", action="store_true",
        help="Run only benign (hbb_benign) dataset",
    )
    parser.add_argument(
        "--harmful-only", action="store_true",
        help="Run only harmful (hbb) dataset",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--classifier-model", type=str, default=DEFAULT_CLASSIFIER,
        help=f"Classifier model (default: {DEFAULT_CLASSIFIER})",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[DEFAULT_SEED],
        help=f"Seed(s) to run (default: {DEFAULT_SEED}). "
             "Pass multiple for mean±std: --seed 42 43 44",
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
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
        help="Max parallel runs (default: 1 = sequential). "
             "Use --parallel 4 to run 4 conditions simultaneously.",
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
            if args.benign_only and label != "benign":
                continue
            if args.harmful_only and label != "harmful":
                continue
            for seed in seeds:
                seed_suffix = f"/seed_{seed}" if len(seeds) > 1 else ""
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
    log_root = args.log_root.parent / "icml_gpt4o_smoke" if args.smoke else args.log_root

    # Compute log dirs and populate run specs
    for run in runs:
        seed_suffix = f"seed_{run['seed']}" if len(seeds) > 1 else ""
        if seed_suffix:
            run["log_dir"] = str(log_root / run["exp_name"] / run["label"] / seed_suffix)
        else:
            run["log_dir"] = str(log_root / run["exp_name"] / run["label"])
        run["model"] = args.model
        run["classifier"] = args.classifier_model
        run["max_turns"] = max_turns
        run["retry_on_error"] = args.retry_on_error
        run["limit"] = limit

    total = len(runs)
    results: list[RunResult] = []

    print(f"\n{'='*60}")
    mode = "SMOKE TEST" if args.smoke else "FULL RUN"
    seed_str = ", ".join(str(s) for s in seeds)
    print(f"  ICML BrowserART Experiments — {mode}")
    print(f"  {total} runs, model={args.model}, seed(s)={seed_str}")
    print(f"  Parallel workers: {args.parallel}")
    if args.smoke:
        print(f"  Limit: {SMOKE_LIMIT} samples/run, max_turns={SMOKE_MAX_TURNS}")
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
                print(f"    → [{status_icon}] ({result.duration_seconds:.0f}s): {result.message[:80]}")
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
