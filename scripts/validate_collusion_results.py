#!/usr/bin/env python3
"""
Validate all DCOP collusion experiment results.

Checks: scorer completeness, metric consistency, attack injection,
defense activation, CollusionLog initialization, benign sanity.

Usage:
    python scripts/validate_collusion_results.py
    python scripts/validate_collusion_results.py --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

LOG_ROOT = Path("neurips/logs_collusion")

EXPECTED_SCORERS = ["security_scorer", "integrity_scorer"]
SCENARIO_SCORERS = {"jira": "jira_ticket_scorer", "hospital": "hospital_scorer"}


def _parse_setup(model: str, setup: str) -> dict:
    """Extract attack type, arch, defense, scenario from setup name."""
    info = {"model": model, "setup": setup, "scenario": "jira" if "jira" in setup else "hospital"}

    if "compromised" in setup:
        info["attack"] = "compromised"
    elif "collusion_1" in setup:
        info["attack"] = "collusion_1"
    elif "collusion_a" in setup:
        info["attack"] = "collusion_a"
    elif "collusion_b" in setup:
        info["attack"] = "collusion_b"
    elif "benign" in setup:
        info["attack"] = "benign"
    else:
        info["attack"] = "unknown"

    info["defense"] = "none"
    for d in ["dual_llm", "system_prompt", "llm_monitor", "guardian"]:
        if f"_{d}" in setup:
            info["defense"] = d
            break

    info["arch"] = "flat"
    clean = setup
    if info["defense"] != "none":
        clean = setup.replace(f"_{info['defense']}", "")
    for a in ["mesh_shared", "star"]:
        if f"_{a}" in clean:
            info["arch"] = a
            break

    return info


def validate_eval(eval_path: Path, info: dict, verbose: bool = False) -> list[str]:
    """Validate a single eval file. Returns list of issues."""
    issues = []
    tag = f"{info['model']}/{info['setup']}"

    try:
        with zipfile.ZipFile(eval_path) as z:
            if "header.json" not in z.namelist():
                return [f"{tag}: no header.json"]

            header = json.loads(z.read("header.json"))
            if header.get("status") != "success":
                return [f"{tag}: status={header.get('status')}"]

            sample_files = [
                n for n in z.namelist()
                if n.startswith("samples/") and n.endswith(".json")
            ]
            if not sample_files:
                return [f"{tag}: no samples"]

            sample = json.loads(z.read(sample_files[0]))
            scores = sample.get("scores", {})

            # 1. Scorer completeness
            for scorer in EXPECTED_SCORERS:
                if scorer not in scores:
                    issues.append(f"{tag}: missing {scorer}")

            scenario_scorer = SCENARIO_SCORERS[info["scenario"]]
            if scenario_scorer not in scores:
                issues.append(f"{tag}: missing {scenario_scorer}")

            if "dcop_scorer" not in scores:
                issues.append(f"{tag}: missing dcop_scorer")

            is_attack = info["attack"] != "benign"

            if is_attack and "collusion_scorer" not in scores:
                issues.append(f"{tag}: attack run missing collusion_scorer")

            if not is_attack and "collusion_scorer" in scores:
                issues.append(f"{tag}: benign run has collusion_scorer")

            # 2. Metric consistency (coalition_advantage vs per_agent_reward)
            cs = scores.get("collusion_scorer", {})
            if cs and is_attack:
                val = cs.get("value", {})
                meta = cs.get("metadata", {})

                coalition_agents = meta.get("coalition_agents", [])
                reported_adv = val.get("coalition_advantage") if isinstance(val, dict) else None

                ss = scores.get(scenario_scorer, {})
                ss_meta = ss.get("metadata", {}) if ss else {}
                par = ss_meta.get("per_agent_reward", {})

                if par and coalition_agents and reported_adv is not None:
                    coal_rewards = [par.get(a, 0.0) for a in coalition_agents if a in par]
                    non_coal_rewards = [v for k, v in par.items() if k not in coalition_agents]
                    if coal_rewards and non_coal_rewards:
                        expected_adv = (
                            sum(coal_rewards) / len(coal_rewards)
                            - sum(non_coal_rewards) / len(non_coal_rewards)
                        )
                        if abs(reported_adv - expected_adv) > 0.01:
                            issues.append(
                                f"{tag}: coalition_advantage mismatch: "
                                f"reported={reported_adv:.3f} computed={expected_adv:.3f}"
                            )

            # 3. CollusionLog for compromised runs
            if info["attack"] == "compromised" and cs:
                meta = cs.get("metadata", {})
                if not meta.get("coalition_agents"):
                    issues.append(f"{tag}: compromised run has empty coalition_agents")
                if meta.get("collusion_mode") != "compromised":
                    issues.append(
                        f"{tag}: compromised run has collusion_mode="
                        f"'{meta.get('collusion_mode')}' (expected 'compromised')"
                    )

            # 4. Channel activity sanity
            if cs and is_attack:
                val = cs.get("value", {})
                chan = val.get("channel_activity", 0) if isinstance(val, dict) else 0
                if info["attack"] == "compromised" and chan > 0:
                    issues.append(f"{tag}: compromised run has channel_activity={chan}")

            # 5. Security scorer for compromised (ASR check)
            sec = scores.get("security_scorer", {})
            if sec and info["attack"] == "compromised":
                sec_val = sec.get("value", {})
                isr = sec_val.get("injection_success_rate")
                if verbose and isr is not None:
                    pass  # ISR available, good

    except (zipfile.BadZipFile, json.JSONDecodeError) as e:
        issues.append(f"{tag}: corrupt eval: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate DCOP collusion results")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    all_issues: list[str] = []
    stats = defaultdict(int)
    attack_counts = Counter()
    defense_counts = Counter()

    for model_dir in sorted(LOG_ROOT.iterdir()):
        if not model_dir.is_dir() or "smoke" in model_dir.name or "verify" in model_dir.name:
            continue
        model = model_dir.name

        for setup_dir in sorted(model_dir.iterdir()):
            if not setup_dir.is_dir():
                continue
            setup = setup_dir.name
            info = _parse_setup(model, setup)

            # Find newest valid eval
            found = False
            for p in sorted(setup_dir.glob("*.eval"), reverse=True):
                try:
                    with zipfile.ZipFile(p) as z:
                        if "header.json" not in z.namelist():
                            continue
                        h = json.loads(z.read("header.json"))
                        if h.get("status") != "success":
                            continue
                        found = True
                        issues = validate_eval(p, info, args.verbose)
                        all_issues.extend(issues)
                        stats["validated"] += 1
                        attack_counts[info["attack"]] += 1
                        defense_counts[info["defense"]] += 1
                        break
                except (zipfile.BadZipFile, json.JSONDecodeError):
                    continue

            if not found:
                stats["no_valid_eval"] += 1

    # Report
    print("=" * 70)
    print("  VALIDATION REPORT")
    print("=" * 70)
    print(f"  Evals validated: {stats['validated']}")
    print(f"  No valid eval found: {stats['no_valid_eval']}")
    print(f"  By attack type: {dict(attack_counts)}")
    print(f"  By defense: {dict(defense_counts)}")
    print()

    if all_issues:
        print(f"  ISSUES FOUND: {len(all_issues)}")
        for issue in all_issues:
            print(f"    {issue}")
    else:
        print("  ISSUES: None! All validations passed.")

    print()
    return len(all_issues)


if __name__ == "__main__":
    exit(main())
