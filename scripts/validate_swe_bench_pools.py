#!/usr/bin/env python3
"""
Generate validated_pools.json by running Docker-based validation.

Developer tool — runs each pool's tests in Docker to verify that
fail_to_pass tests actually FAIL and pass_to_pass tests actually PASS
at the environment_setup_commit. Results are saved to the shipped data
file so users never need Docker at runtime.

Usage:
    # Dry run — show what would be validated
    python scripts/validate_swe_bench_pools.py --dry-run

    # Validate django pools only
    python scripts/validate_swe_bench_pools.py --repos django/django --max-workers 2

    # Full validation (all repos, requires Docker + many images)
    python scripts/validate_swe_bench_pools.py --max-workers 4

    # Re-run only issues that previously errored (patch failures, image pulls)
    python scripts/validate_swe_bench_pools.py --revalidate-errors --max-workers 4

    # Force re-validate all issues in all cached pools
    python scripts/validate_swe_bench_pools.py --repos django/django --force-revalidate
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orbit.scenarios.coding.swe_bench.configs import SWEBenchScenarioConfig
from orbit.scenarios.coding.swe_bench.dataset_builder import (
    build_version_pools,
    get_pool_statistics,
    load_swe_bench_instances,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "orbit" / "scenarios" / "swe_bench" / "data" / "validated_pools.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SWE-Bench version pools (requires Docker)"
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="Filter to specific repos (e.g. django/django)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel Docker containers (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force-revalidate",
        action="store_true",
        help="Re-validate all issues in all cached pools",
    )
    parser.add_argument(
        "--revalidate-errors",
        action="store_true",
        help="Re-run only issues that previously errored (patch failures, "
        "image pull failures). Pools with no errors are skipped entirely.",
    )
    parser.add_argument(
        "--timeout-per-issue",
        type=int,
        default=300,
        help="Timeout per issue in seconds (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be validated without running Docker",
    )
    args = parser.parse_args()

    # Load instances
    config = SWEBenchScenarioConfig(repos=args.repos)
    issues = load_swe_bench_instances(config)

    # Build pools without validation data (we're generating it)
    pools = build_version_pools(issues, validation_cache=None)

    # Import validator (requires docker package) — needed for cache loading too
    try:
        from orbit.scenarios.coding.swe_bench.validator import (
            load_validation_cache,
            validate_all_pools,
        )
    except ImportError as e:
        if not args.dry_run:
            print(f"\nError: {e}")
            print("Install docker package: pip install docker")
            sys.exit(1)
        load_validation_cache = None  # type: ignore[assignment]

    # Load existing cache if any
    existing_cache = None
    if load_validation_cache and args.output.exists():
        existing_cache = load_validation_cache(args.output)

    # Print summary
    stats = get_pool_statistics(pools)
    print("\n" + "=" * 70)
    print("VERSION POOL SUMMARY")
    print("=" * 70)
    print(f"\nTotal pools: {stats['total_pools']}")
    print(f"Total issues: {stats['total_issues']}")

    if existing_cache:
        total_cached = sum(len(v) for v in existing_cache.results.values())
        total_valid = sum(
            1 for v in existing_cache.results.values() for r in v.values() if r.valid
        )
        total_errors = sum(
            1 for v in existing_cache.results.values() for r in v.values() if r.error_message
        )
        print(f"\nExisting cache: {len(existing_cache.results)} pools, "
              f"{total_cached} issues ({total_valid} valid, {total_errors} errors)")

    if args.revalidate_errors and existing_cache:
        # Show what will be re-validated
        errored_pools = 0
        errored_issues = 0
        for pool in pools:
            if pool.pool_id in existing_cache.results:
                n_errors = sum(
                    1 for r in existing_cache.results[pool.pool_id].values()
                    if r.error_message
                )
                if n_errors > 0:
                    errored_pools += 1
                    errored_issues += n_errors
        print(f"\nWill re-validate: {errored_pools} pools, {errored_issues} errored issues")
        print("(pools with no errors are skipped)")
    else:
        print(f"\nPools by repo:")
        for repo, count in stats["pools_by_repo"].items():
            issue_count = stats["issues_by_repo"][repo]
            print(f"  {repo:40s}  {count:3d} pools  {issue_count:4d} issues")

    if args.dry_run:
        if args.revalidate_errors and existing_cache:
            print(f"\n[DRY RUN] Pools with errors to re-validate:")
            for pool in pools:
                if pool.pool_id in existing_cache.results:
                    errors = [
                        (iid, r.error_message)
                        for iid, r in existing_cache.results[pool.pool_id].items()
                        if r.error_message
                    ]
                    if errors:
                        print(f"\n  {pool.pool_id} ({len(errors)} errors):")
                        for iid, msg in errors[:5]:
                            print(f"    {iid}: {msg[:60]}")
                        if len(errors) > 5:
                            print(f"    ... and {len(errors) - 5} more")
        else:
            print(f"\n[DRY RUN] Would validate {len(pools)} pools. Exiting.")
        return

    # Run validation (saves after each pool completes for crash resilience)
    print(f"\nStarting validation with {args.max_workers} workers...")
    print(f"Progress saved incrementally to: {args.output}")
    cache = validate_all_pools(
        pools,
        max_workers=args.max_workers,
        existing_cache=existing_cache,
        force_revalidate=args.force_revalidate,
        revalidate_errors=args.revalidate_errors,
        save_path=args.output,
    )

    # Print results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    total_valid = 0
    total_invalid = 0
    total_error = 0

    for pool_id, pool_results in sorted(cache.results.items()):
        valid = sum(1 for r in pool_results.values() if r.valid)
        invalid = sum(1 for r in pool_results.values() if not r.valid and not r.error_message)
        error = sum(1 for r in pool_results.values() if r.error_message)
        total_valid += valid
        total_invalid += invalid
        total_error += error
        print(f"  {pool_id:50s}  valid={valid:3d}  invalid={invalid:3d}  error={error:3d}")

    total = total_valid + total_invalid + total_error
    print(f"\nTotal: {total_valid}/{total} valid ({100*total_valid/total:.1f}%)")
    if total_error:
        print(f"Errors: {total_error}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
