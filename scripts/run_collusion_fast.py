#!/usr/bin/env python3
"""
Fast parallel runner for DCOP collusion experiments.

Generates all commands, runs them with high parallelism, tracks progress
and errors in real-time. Much faster than run_collusion_experiments.py.

Usage:
    python scripts/run_collusion_fast.py                    # all 1350 runs
    python scripts/run_collusion_fast.py --no-defense       # 270 no-defense runs
    python scripts/run_collusion_fast.py --workers 20       # 20 parallel workers
    python scripts/run_collusion_fast.py --dry-run           # just print commands
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_collusion_experiments import (
    MODELS,
    _build_experiment_setups,
    build_command,
    find_completed_eval,
)

DEFAULT_SEEDS = [42, 73, 117]
LOG_ROOT = Path("neurips/logs_collusion_v3")


def generate_runs(
    seeds: list[int],
    model_filter: str | None = None,
    defense_filter: str | None = None,
    no_defense: bool = False,
    arch_filter: str | None = None,
    scenario_filter: str | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    """Generate all run specs, skipping existing."""
    setups = _build_experiment_setups()
    runs = []

    for model_short, model_full in MODELS:
        if model_filter and model_full != model_filter:
            continue
        for setup in setups:
            if defense_filter and setup.defense != defense_filter:
                continue
            if no_defense and setup.defense != "none":
                continue
            if arch_filter and setup.arch != arch_filter:
                continue
            if scenario_filter and setup.scenario != scenario_filter:
                continue

            for seed in seeds:
                seed_suffix = f"_s{seed}" if len(seeds) > 1 else ""
                log_dir = LOG_ROOT / model_short / f"{setup.name}{seed_suffix}"

                if skip_existing and find_completed_eval(log_dir):
                    continue

                cmd = build_command(setup, model_full, log_dir, seed, False)
                runs.append({
                    "cmd": cmd,
                    "tag": f"{model_short}/{setup.name}{seed_suffix}",
                    "log_dir": str(log_dir),
                    "model": model_short,
                    "defense": setup.defense,
                })

    return runs


def run_one(spec: dict) -> dict:
    """Execute a single run. Returns result dict."""
    t0 = time.time()
    try:
        r = subprocess.run(
            spec["cmd"], capture_output=True, text=True, timeout=600,
        )
        duration = time.time() - t0
        ok = r.returncode == 0
        if not ok:
            err = r.stderr[-300:] if r.stderr else "unknown error"
        else:
            err = ""
        return {"tag": spec["tag"], "ok": ok, "duration": duration, "error": err}
    except subprocess.TimeoutExpired:
        return {"tag": spec["tag"], "ok": False, "duration": 600, "error": "TIMEOUT"}
    except Exception as e:
        return {"tag": spec["tag"], "ok": False, "duration": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Fast parallel collusion runner")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--defense", type=str, default=None)
    parser.add_argument("--no-defense", action="store_true")
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed else DEFAULT_SEEDS

    runs = generate_runs(
        seeds=seeds,
        model_filter=args.model,
        defense_filter=args.defense,
        no_defense=args.no_defense,
        arch_filter=args.arch,
        scenario_filter=args.scenario,
        skip_existing=not args.no_skip,
    )

    if not runs:
        print("No runs to execute (all exist or no matches).")
        return

    print(f"{'='*60}")
    print(f"  {len(runs)} runs | {args.workers} workers | seeds={seeds}")
    print(f"  Log root: {LOG_ROOT}")
    est_minutes = len(runs) * 2 / args.workers
    print(f"  Estimated: ~{est_minutes:.0f} minutes")
    print(f"{'='*60}")

    if args.dry_run:
        for r in runs[:10]:
            print(f"  {r['tag']}")
        if len(runs) > 10:
            print(f"  ... and {len(runs)-10} more")
        return

    t0 = time.time()
    done = 0
    failed = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, r): r for r in runs}
        for f in as_completed(futures):
            result = f.result()
            done += 1
            if not result["ok"]:
                failed += 1
                errors.append(f"{result['tag']}: {result['error'][:100]}")

            elapsed = time.time() - t0
            rate = done / elapsed * 60 if elapsed > 0 else 0
            remaining = (len(runs) - done) / rate if rate > 0 else 0

            if done % 10 == 0 or not result["ok"]:
                status = "OK" if result["ok"] else "FAIL"
                print(
                    f"  [{done}/{len(runs)}] {status} {result['tag']} "
                    f"({result['duration']:.0f}s) | "
                    f"{rate:.0f}/min | ~{remaining:.0f}min left | "
                    f"{failed} errors"
                )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE: {done-failed} success, {failed} failed, {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")


if __name__ == "__main__":
    main()
