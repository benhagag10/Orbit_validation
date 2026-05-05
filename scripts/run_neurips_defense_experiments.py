#!/usr/bin/env python3
"""
Batch runner for NeurIPS defense experiments.

Runs 4 conditions × 3 models × 2 datasets × 3 defenses = 72 eval runs.
Conditions: single_agent (Baseline), star_specialist (Star),
            mesh_delegation (Mesh), memory_full (Star + Shared Memory)
Models:     openai/gpt-5.4, anthropic/claude-sonnet-4-6,
            together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput
Datasets:   hbb (harmful → ASR), hbb_benign (benign → Benign TC)
Defenses:   system_prompt, llm_monitor, guardian

Usage:
    python scripts/run_neurips_defense_experiments.py --smoke
    python scripts/run_neurips_defense_experiments.py --dry-run
    python scripts/run_neurips_defense_experiments.py --model openai/gpt-5.4
    python scripts/run_neurips_defense_experiments.py --defense system_prompt
    python scripts/run_neurips_defense_experiments.py --condition single_agent
    python scripts/run_neurips_defense_experiments.py --parallel 4
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

# ── Experiment matrix ──────────────────────────────────────────────────

CONDITIONS = [
    # (dir_name, condition_code, paper_name)
    # dir_names match analyze_icml_experiments.py expectations
    ("single_agent",    "single_agent",    "Baseline"),
    ("star_specialist", "star_specialist",  "Star"),
    ("mesh_delegation", "mesh_delegation",  "Mesh"),
    ("memory_full",     "memory_full",      "Star + Shared Memory"),
]

MODELS = [
    ("gpt5.4",           "openai/gpt-5.4"),
    ("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
    ("qwen",             "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
]

DATASETS = [
    ("hbb",        "harmful", 100),
    ("hbb_benign", "benign",   42),
]

DEFENSES = [
    ("system_prompt", "D1: System Prompt"),
    ("llm_monitor",   "D2: LLM Monitor"),
    ("guardian",      "D3: Guardian Agent"),
    ("dual_llm",     "D4: Dual-LLM"),
]

# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_CLASSIFIER = "openai/gpt-4.1"
DEFAULT_SEED = 42
DEFAULT_MAX_TURNS = 15
DEFAULT_LOG_ROOT = Path("neurips/logs_defenses")

SMOKE_LIMIT = 3
SMOKE_MAX_TURNS = 5


@dataclass
class RunResult:
    condition: str
    model_name: str
    dataset: str
    dataset_label: str
    defense: str
    status: str
    sample_count: int
    duration_seconds: float
    log_path: str
    message: str = ""


def find_completed_eval(log_dir: Path, expected_samples: int) -> Path | None:
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


def run_experiment(
    condition: str,
    dataset: str,
    defense: str,
    log_dir: Path,
    model: str,
    classifier: str,
    seed: int,
    max_turns: int,
    limit: int | None = None,
) -> tuple[bool, str]:
    cmd = [
        "uv", "run", "orbit", "--skip-preflight", "browserart",
        "--model", model,
        "--dataset", dataset,
        "--condition", condition,
        "--seed", str(seed),
        "--max-turns", str(max_turns),
        "--classifier-model", classifier,
        "--retry-on-error", "1",
        "--log-dir", str(log_dir),
        "-T", f"defense_preset={defense}",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if "anthropic" in model:
        cmd.extend(["--max-tokens", "32000"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
        )
        if result.returncode != 0:
            return False, f"Exit code {result.returncode}\n{result.stderr[-500:]}"
        return True, result.stdout[-200:]
    except subprocess.TimeoutExpired:
        return False, "Timed out after 2 hours"
    except Exception as e:
        return False, str(e)


def _run_single(spec: dict) -> RunResult:
    log_dir = Path(spec["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    effective_expected = spec.get("limit") or spec["expected_count"]

    t0 = time.time()
    success, output = run_experiment(
        condition=spec["condition"],
        dataset=spec["dataset"],
        defense=spec["defense"],
        log_dir=log_dir,
        model=spec["model"],
        classifier=spec["classifier"],
        seed=spec["seed"],
        max_turns=spec["max_turns"],
        limit=spec.get("limit"),
    )
    duration = time.time() - t0

    if success:
        completed = find_completed_eval(log_dir, effective_expected)
        if completed:
            status = "success"
            msg = f"Completed: {completed.name}"
        else:
            status = "failed"
            msg = "No completed eval found after run"
    else:
        status = "failed"
        msg = output[:200]

    return RunResult(
        condition=spec["condition"],
        model_name=spec["model_name"],
        dataset=spec["dataset"],
        dataset_label=spec["label"],
        defense=spec["defense"],
        status=status,
        sample_count=effective_expected,
        duration_seconds=duration,
        log_path=str(log_dir),
        message=msg,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run NeurIPS defense experiments (3 defenses × 4 conditions × 3 models × 2 datasets)"
    )
    parser.add_argument("--smoke", action="store_true",
                        help=f"Smoke test: {SMOKE_LIMIT} samples, max-turns={SMOKE_MAX_TURNS}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--condition", type=str, default=None,
                        help="Filter to condition code (e.g., single_agent)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to model (e.g., openai/gpt-5.4)")
    parser.add_argument("--defense", type=str, default=None,
                        choices=["system_prompt", "llm_monitor", "guardian", "dual_llm"],
                        help="Filter to one defense preset")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["hbb", "hbb_benign"])
    parser.add_argument("--classifier-model", type=str, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.rerun:
        args.skip_existing = False

    # Build run list
    runs: list[dict] = []
    for dir_name, condition, paper_name in CONDITIONS:
        if args.condition and condition != args.condition:
            continue
        for model_short, model_full in MODELS:
            if args.model and model_full != args.model:
                continue
            for defense, defense_label in DEFENSES:
                if args.defense and defense != args.defense:
                    continue
                for dataset, label, count in DATASETS:
                    if args.dataset and dataset != args.dataset:
                        continue
                    # Log structure: logs_defenses/<model>/<defense>/<condition>/<harmful|benign>/
                    log_root = args.log_root / f"{model_short}_smoke" if args.smoke else args.log_root / model_short
                    log_dir = log_root / defense / dir_name / label
                    runs.append({
                        "dir_name": dir_name,
                        "condition": condition,
                        "paper_name": paper_name,
                        "model": model_full,
                        "model_name": model_short,
                        "defense": defense,
                        "defense_label": defense_label,
                        "dataset": dataset,
                        "label": label,
                        "expected_count": count,
                        "log_dir": str(log_dir),
                        "classifier": args.classifier_model,
                        "seed": args.seed,
                        "max_turns": SMOKE_MAX_TURNS if args.smoke else args.max_turns,
                        "limit": args.limit if args.limit is not None else (SMOKE_LIMIT if args.smoke else None),
                    })

    if not runs:
        print("No runs match the given filters.")
        sys.exit(1)

    total = len(runs)
    print(f"\n{'='*70}")
    mode = "SMOKE TEST" if args.smoke else "FULL RUN"
    print(f"  NeurIPS Defense Experiments — {mode}")
    print(f"  {total} runs")
    print(f"  Parallel workers: {args.parallel}")
    if args.smoke:
        print(f"  Limit: {SMOKE_LIMIT} samples/run, max_turns={SMOKE_MAX_TURNS}")
    print(f"{'='*70}\n")

    to_run: list[dict] = []
    results: list[RunResult] = []

    for idx, run in enumerate(runs, 1):
        effective_expected = run.get("limit") or run["expected_count"]
        log_dir = Path(run["log_dir"])
        tag = f"{run['model_name']}/{run['defense']}/{run['dir_name']}/{run['label']}"

        if args.skip_existing and not args.smoke:
            existing = find_completed_eval(log_dir, effective_expected)
            if existing:
                print(f"[{idx}/{total}] {tag}: SKIPPED (exists)")
                results.append(RunResult(
                    condition=run["condition"], model_name=run["model_name"],
                    dataset=run["dataset"], dataset_label=run["label"],
                    defense=run["defense"], status="skipped",
                    sample_count=effective_expected, duration_seconds=0,
                    log_path=str(existing), message="exists",
                ))
                continue

        if args.dry_run:
            cmd_preview = (
                f"orbit browserart --model {run['model']} --dataset {run['dataset']} "
                f"--condition {run['condition']} -T defense_preset={run['defense']}"
            )
            print(f"[{idx}/{total}] {tag}: DRY RUN")
            print(f"          cmd: {cmd_preview}")
            results.append(RunResult(
                condition=run["condition"], model_name=run["model_name"],
                dataset=run["dataset"], dataset_label=run["label"],
                defense=run["defense"], status="dry-run",
                sample_count=effective_expected, duration_seconds=0,
                log_path=str(log_dir),
            ))
            continue

        print(f"[{idx}/{total}] {tag}: QUEUED ({effective_expected} samples)")
        to_run.append(run)

    # Execute
    if to_run:
        if args.parallel > 1:
            print(f"\nLaunching {len(to_run)} runs with {args.parallel} workers...\n")
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                futures = {executor.submit(_run_single, r): r for r in to_run}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        r = futures[future]
                        result = RunResult(
                            condition=r["condition"], model_name=r["model_name"],
                            dataset=r["dataset"], dataset_label=r["label"],
                            defense=r["defense"], status="failed",
                            sample_count=r["expected_count"], duration_seconds=0,
                            log_path=r["log_dir"], message=str(e),
                        )
                    tag = f"{result.model_name}/{result.defense}/{result.condition}/{result.dataset_label}"
                    print(f"  {result.status.upper()}: {tag} ({result.duration_seconds:.0f}s)")
                    results.append(result)
        else:
            for run in to_run:
                tag = f"{run['model_name']}/{run['defense']}/{run['dir_name']}/{run['label']}"
                print(f"\n  Running: {tag} ...")
                result = _run_single(run)
                print(f"  {result.status.upper()} ({result.duration_seconds:.0f}s): {result.message}")
                results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    succeeded = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    print(f"  Success: {succeeded}  Failed: {failed}  Skipped: {skipped}  Total: {len(results)}")

    if failed > 0:
        print(f"\n  FAILURES:")
        for r in results:
            if r.status == "failed":
                print(f"    {r.model_name}/{r.defense}/{r.condition}/{r.dataset_label}: {r.message[:100]}")

    print()


if __name__ == "__main__":
    main()
