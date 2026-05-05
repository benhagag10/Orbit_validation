#!/usr/bin/env python3
"""
Validate defense experiment eval files.

Checks each eval for: valid zip, correct sample count, low browser/API
error rate, and real agent generations. Used by the GCP monitor for
auto-restart on bad evals, and for the final local audit.

Usage:
    python scripts/validate_defense_evals.py <path>              # single file
    python scripts/validate_defense_evals.py <dir>               # all evals in dir
    python scripts/validate_defense_evals.py <dir> --expected 96 # assert count
    python scripts/validate_defense_evals.py <dir> --json        # JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EvalReport:
    path: str
    valid: bool
    total_samples: int = 0
    browser_errors: int = 0
    api_errors: int = 0
    empty_generations: int = 0
    good_samples: int = 0
    error: str = ""


def validate_eval(path: Path, min_samples: int = 1, max_error_rate: float = 0.3) -> EvalReport:
    """Validate a single eval file. Returns a report."""
    report = EvalReport(path=str(path), valid=False)

    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        report.error = f"Bad file: {e}"
        return report

    try:
        header = json.loads(z.read("header.json"))
    except (KeyError, json.JSONDecodeError) as e:
        report.error = f"Bad header: {e}"
        return report

    samps = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
    report.total_samples = len(samps)

    if len(samps) < min_samples:
        report.error = f"Too few samples: {len(samps)} < {min_samples}"
        return report

    for sf in samps:
        try:
            s = json.loads(z.read(sf))
        except json.JSONDecodeError:
            report.browser_errors += 1
            continue

        meta = s.get("metadata", {})
        err_str = meta.get("error", "")
        msgs = s.get("messages", [])
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]

        if meta.get("browser_error") or "task_id=" in err_str:
            report.browser_errors += 1
        elif "credit_limit" in err_str or "Connection error" in err_str or "402" in err_str:
            report.api_errors += 1
        elif len(assistant_msgs) == 0 and len(msgs) <= 2:
            report.empty_generations += 1
        else:
            report.good_samples += 1

    total_errors = report.browser_errors + report.api_errors + report.empty_generations
    error_rate = total_errors / max(report.total_samples, 1)

    if error_rate > max_error_rate:
        report.error = (
            f"Error rate {error_rate:.0%}: "
            f"{report.browser_errors} browser, "
            f"{report.api_errors} api, "
            f"{report.empty_generations} empty"
        )
        return report

    report.valid = True
    return report


def validate_directory(
    directory: Path,
    expected: int | None = None,
) -> tuple[list[EvalReport], list[EvalReport]]:
    """Validate all evals in a directory tree. Returns (good, bad)."""
    good = []
    bad = []

    for eval_path in sorted(directory.rglob("*.eval")):
        if "_smoke" in str(eval_path):
            continue
        report = validate_eval(eval_path)
        if report.valid:
            good.append(report)
        else:
            bad.append(report)

    return good, bad


def main():
    parser = argparse.ArgumentParser(description="Validate defense eval files")
    parser.add_argument("path", type=Path, help="Eval file or directory")
    parser.add_argument("--expected", type=int, default=None,
                        help="Expected number of good evals (exit 1 if not met)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--delete-bad", action="store_true",
                        help="Delete bad eval files")
    args = parser.parse_args()

    if args.path.is_file():
        report = validate_eval(args.path)
        if args.json:
            print(json.dumps(asdict(report), indent=2))
        else:
            status = "GOOD" if report.valid else "BAD"
            print(f"{status}: {report.path}")
            if report.error:
                print(f"  {report.error}")
            print(f"  {report.good_samples}/{report.total_samples} good samples")
        sys.exit(0 if report.valid else 1)

    good, bad = validate_directory(args.path)

    if args.delete_bad:
        for r in bad:
            Path(r.path).unlink(missing_ok=True)

    if args.json:
        print(json.dumps({
            "good": [asdict(r) for r in good],
            "bad": [asdict(r) for r in bad],
            "summary": {
                "good_count": len(good),
                "bad_count": len(bad),
                "total": len(good) + len(bad),
            },
        }, indent=2))
    else:
        print(f"GOOD: {len(good)}  BAD: {len(bad)}  TOTAL: {len(good) + len(bad)}")
        if bad:
            print(f"\nBAD EVALS:")
            for r in bad:
                tag = str(r.path).replace("neurips/logs_defenses/", "")
                print(f"  {tag}: {r.error}")
        if args.delete_bad and bad:
            print(f"\nDeleted {len(bad)} bad evals")

    if args.expected is not None and len(good) < args.expected:
        print(f"\nFAIL: {len(good)} good < {args.expected} expected")
        sys.exit(1)

    sys.exit(0 if not bad else 2)


if __name__ == "__main__":
    main()
