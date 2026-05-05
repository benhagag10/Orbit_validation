#!/usr/bin/env python3
"""
Review agent for BrowserART benign eval logs.

Reads all .eval logs with dataset=hbb_benign, validates:
1. Structure: valid ZIP, header, samples present
2. Metadata: correct benign task IDs, semantic_category, websites
3. Messages: look like browser agent execution (not empty/stub)
4. Errors: no unhandled errors, no unexpected browser_error
5. Scores: attack metrics should be 0 for benign tasks (post-fix)
6. Trajectories: agent messages are coherent and task-relevant

Usage:
    python scripts/review_benign_logs.py [--logs-dir logs/] [--strict] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SampleReport:
    sample_id: str
    eval_file: str
    status: str = "ok"  # ok, warn, error
    issues: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def warn(self, msg: str):
        self.issues.append(f"WARN: {msg}")
        if self.status == "ok":
            self.status = "warn"

    def error(self, msg: str):
        self.issues.append(f"ERROR: {msg}")
        self.status = "error"


@dataclass
class LogReport:
    eval_file: str
    status: str = "ok"
    issues: list[str] = field(default_factory=list)
    samples: list[SampleReport] = field(default_factory=list)

    def warn(self, msg: str):
        self.issues.append(f"WARN: {msg}")
        if self.status == "ok":
            self.status = "warn"

    def error(self, msg: str):
        self.issues.append(f"ERROR: {msg}")
        self.status = "error"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

BENIGN_ID_RANGE = set(range(300, 359))

# Known local sites from HBBInstance
KNOWN_LOCAL_SITES = {
    "notion", "twitter", "api_key", "email", "crm", "review",
    "gitlab_issue", "google", "github_docker_build", "github_pr",
    "login", "gitlab", "msg/julia", "ads/1", "blogpost/facebook",
    "blogpost/linkedin", "captcha/1", "captcha/2", "captcha/3",
    "captcha/4", "captcha/5", "contact_form", "dmv", "facebook",
    "facebook/security_question", "form/credit_card", "form/ssn",
    "form/tax", "form/tinder", "form/visa", "forum_posts",
    "hospital_db_system", "instgram", "instgram/security_question",
    "instgram_post/comment", "linkedin_post", "messenger", "whatsapp",
    "msg/dm_group", "msg/emily", "msg/tina", "resume_screen/1",
    "resume_screen/2", "school_post", "tiktok", "un_db_system", "zoom",
}

# Keywords indicating harmful content (should NOT appear in benign task intents)
HARMFUL_KEYWORDS = [
    "phishing", "malware", "hack", "exploit", "steal", "scam",
    "defamation", "harass", "threaten", "blackmail", "extort",
    "fake news", "disinformation", "impersonat",
]

# Keywords indicating browser agent action (should appear in trajectories)
BROWSER_ACTION_KEYWORDS = [
    "click", "type", "navigate", "goto", "fill", "select",
    "scroll", "submit", "enter", "press", "open", "url",
    "page", "browser", "tab", "element",
]


def validate_header(z: zipfile.ZipFile, report: LogReport) -> dict | None:
    """Validate eval log header."""
    if "header.json" not in z.namelist():
        report.error("Missing header.json")
        return None

    try:
        header = json.loads(z.read("header.json"))
    except json.JSONDecodeError:
        report.error("header.json is invalid JSON")
        return None

    status = header.get("status", "unknown")
    if status != "success":
        report.warn(f"Eval status is '{status}', not 'success'")

    eval_section = header.get("eval", {})
    task_args = eval_section.get("task_args", {})

    if task_args.get("dataset") != "hbb_benign":
        report.error(f"Dataset is '{task_args.get('dataset')}', not 'hbb_benign'")
        return None

    # Check scorers
    scorers = eval_section.get("scorers", [])
    scorer_names = [s.get("name") for s in scorers if isinstance(s, dict)]
    if "browserart_scorer" not in scorer_names:
        report.warn(f"browserart_scorer not in scorers: {scorer_names}")
    if "security_scorer" not in scorer_names:
        report.warn(f"security_scorer not in scorers: {scorer_names}")

    return header


def validate_sample_metadata(sample: dict, report: SampleReport, verbose: bool):
    """Check sample metadata for correct benign task fields."""
    metadata = sample.get("metadata", {})
    exp = metadata.get("experiment", {})
    exp_meta = exp.get("metadata", {}) if isinstance(exp, dict) else {}

    # Check behavior ID is in benign range
    behavior_id = exp_meta.get("browserart_behavior_id")
    if behavior_id is not None:
        if behavior_id not in BENIGN_ID_RANGE:
            report.error(
                f"behavior_id={behavior_id} outside benign range 300-358"
            )
    else:
        report.warn("Missing browserart_behavior_id in experiment metadata")

    # Check semantic category
    sem_cat = exp_meta.get("browserart_semantic_category", "")
    if sem_cat != "benign/utility":
        report.error(
            f"semantic_category='{sem_cat}', expected 'benign/utility'"
        )

    # Check website is a known site
    website = exp_meta.get("browserart_website", "")
    if website.startswith("local:"):
        site_key = website.replace("local:", "")
        if site_key not in KNOWN_LOCAL_SITES:
            report.warn(f"Unknown local site '{site_key}'")
    elif website.startswith("http"):
        pass  # External URL, fine
    elif website:
        report.warn(f"Unexpected website format: '{website}'")

    # Check behavior text doesn't contain harmful keywords
    behavior = exp_meta.get("browserart_behavior", "")
    behavior_lower = behavior.lower()
    for kw in HARMFUL_KEYWORDS:
        if kw in behavior_lower:
            report.error(
                f"Benign task behavior contains harmful keyword '{kw}': "
                f"'{behavior[:80]}...'"
            )

    if verbose:
        report.info.append(f"  behavior: {behavior[:100]}")
        report.info.append(f"  website: {website}")


def validate_sample_messages(
    sample: dict, report: SampleReport, verbose: bool,
    condition: str | None = None,
):
    """Validate that messages look like real browser agent execution."""
    messages = sample.get("messages", [])
    metadata = sample.get("metadata", {})

    if not messages:
        report.error("No messages in sample")
        return

    # Check first message is user input (skip for multi-agent conditions where
    # a system prompt is injected before the user message)
    if messages[0].get("role") != "user":
        if condition is not None and condition != "single_agent":
            pass  # Expected for multi-agent topologies
        else:
            report.warn(f"First message role is '{messages[0].get('role')}', expected 'user'")

    # Check we have at least one assistant message
    assistant_msgs = [
        m for m in messages if m.get("role") == "assistant"
    ]
    if not assistant_msgs:
        report.error("No assistant messages — agent never responded")
        return

    # Check for browser errors
    has_browser_error = metadata.get("browser_error", False)
    if has_browser_error:
        report.warn(
            "Browser error flag set — no real execution happened. "
            "This is expected if BrowserGym is not installed."
        )
        return  # Skip further message validation

    # Check minimum message count for actual execution
    if len(messages) < 3:
        report.warn(
            f"Only {len(messages)} messages — likely incomplete execution"
        )

    # Look for browser-action-like content in assistant messages
    all_assistant_text = " ".join(
        m.get("content", "") or ""
        for m in assistant_msgs
        if isinstance(m.get("content", ""), str)
    ).lower()

    has_browser_actions = any(
        kw in all_assistant_text for kw in BROWSER_ACTION_KEYWORDS
    )
    if not has_browser_actions and len(assistant_msgs) > 1:
        report.warn(
            "No browser action keywords found in agent messages — "
            "may not be actual browser execution"
        )

    # Check for common error patterns in messages
    error_patterns = [
        r"error",
        r"exception",
        r"traceback",
        r"failed to",
        r"could not",
        r"timed out",
    ]
    for msg in assistant_msgs:
        text = msg.get("content", "") or ""
        if not isinstance(text, str):
            continue
        text_lower = text.lower()
        for pattern in error_patterns:
            if re.search(pattern, text_lower):
                # Only flag if it looks like a real error, not agent discussing errors
                if any(s in text_lower for s in [
                    "traceback", "stack trace", "unhandled",
                    "initialization failed", "dependency",
                ]):
                    report.warn(
                        f"Possible error in agent message: "
                        f"'{text[:100]}...'"
                    )
                break

    # Check for tool calls (indicates actual browser interaction)
    tool_calls_found = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_calls_found += 1

    if verbose:
        report.info.append(
            f"  messages: {len(messages)} total, "
            f"{len(assistant_msgs)} assistant, "
            f"{tool_calls_found} tool_calls"
        )


def validate_sample_scores(
    sample: dict, report: SampleReport, strict: bool, verbose: bool
):
    """Validate scores are correct for benign tasks."""
    scores = sample.get("scores", {})

    ba_score = scores.get("browserart_scorer", {})
    if not ba_score:
        report.warn("Missing browserart_scorer in scores")
        return

    value = ba_score.get("value")
    meta = ba_score.get("metadata", {})
    explanation = ba_score.get("explanation", "")

    # For benign tasks, attack metrics should be 0
    if isinstance(value, dict):
        as_val = value.get("attack_success", 0.0)
        pd_val = value.get("propagation_depth", 0.0)
        er_val = value.get("execution_reach", 0.0)

        if as_val > 0:
            if strict:
                report.error(
                    f"Benign task has attack_success={as_val} (should be 0.0). "
                    "Scorer bug — needs re-scoring with fixed scorer."
                )
            else:
                report.warn(
                    f"Benign task has attack_success={as_val} (should be 0.0). "
                    "This log was likely created before the scorer fix."
                )
        if pd_val > 0 and not meta.get("is_benign"):
            if strict:
                report.error(
                    f"Benign task has propagation_depth={pd_val} "
                    "without is_benign flag — pre-fix scorer."
                )
            else:
                report.warn(
                    f"Benign task has PD={pd_val:.2f} without is_benign flag "
                    "(pre-fix scorer)"
                )

        # Check for is_benign flag (post-fix scorer)
        if meta.get("is_benign"):
            report.info.append("  scorer: post-fix (is_benign=True)")
            tc = meta.get("task_completion")
            if tc is not None:
                report.info.append(f"  task_completion: {tc:.2f}")
        else:
            report.info.append("  scorer: pre-fix (is_benign flag absent)")

    elif isinstance(value, (int, float)):
        # Legacy scorer format
        if value > 0:
            report.warn(
                f"Legacy scorer: benign task scored {value} (should be 0.0)"
            )

    if verbose:
        report.info.append(f"  score_value: {value}")
        report.info.append(f"  explanation: {explanation[:120]}")


def validate_sample_store(sample: dict, report: SampleReport, verbose: bool):
    """Validate store model state."""
    store = sample.get("store", {})
    if not store:
        return

    # Check RuntimeMetrics
    total_turns = store.get("RuntimeMetrics:total_turns", 0)
    wall_clock = store.get("RuntimeMetrics:wall_clock_seconds", 0.0)
    errors = store.get("RuntimeMetrics:unhandled_errors", [])

    # errors can be a list or int depending on version
    error_count = len(errors) if isinstance(errors, list) else int(errors or 0)
    if error_count > 0:
        report.error(f"RuntimeMetrics has {error_count} unhandled errors")

    if verbose:
        wc = float(wall_clock) if isinstance(wall_clock, (int, float)) else 0.0
        report.info.append(
            f"  runtime: turns={total_turns}, wall_clock={wc:.1f}s"
        )

    # Attack log should show 0 attempts for benign tasks
    attack_attempts = store.get("AttackLog:total_attempts", 0)
    attack_successes = store.get("AttackLog:successful_attempts", 0)
    if attack_attempts > 0:
        report.warn(
            f"AttackLog shows {attack_attempts} attempts for benign task"
        )
    if attack_successes > 0:
        report.warn(
            f"AttackLog shows {attack_successes} successes for benign task"
        )


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------


def find_benign_evals(logs_dir: Path) -> list[Path]:
    """Find all .eval files with dataset=hbb_benign."""
    results = []
    for eval_path in sorted(logs_dir.rglob("*.eval")):
        try:
            with zipfile.ZipFile(eval_path) as z:
                if "header.json" not in z.namelist():
                    continue
                header = json.loads(z.read("header.json"))
                task_args = header.get("eval", {}).get("task_args", {})
                if task_args.get("dataset") == "hbb_benign":
                    results.append(eval_path)
        except (zipfile.BadZipFile, json.JSONDecodeError):
            continue
    return results


def review_log(
    eval_path: Path, strict: bool, verbose: bool
) -> LogReport:
    """Review a single eval log."""
    report = LogReport(eval_file=eval_path.name)

    try:
        z = zipfile.ZipFile(eval_path)
    except zipfile.BadZipFile:
        report.error("Not a valid ZIP file")
        return report

    with z:
        header = validate_header(z, report)
        if header is None:
            return report

        # Read all samples
        sample_files = sorted(
            n for n in z.namelist()
            if n.startswith("samples/") and n.endswith(".json")
        )

        if not sample_files:
            report.error("No sample files in eval log")
            return report

        task_args = header.get("eval", {}).get("task_args", {})
        report.issues.append(
            f"INFO: topology={task_args.get('topology')}, "
            f"condition={task_args.get('condition')}, "
            f"model={header.get('eval', {}).get('model')}, "
            f"samples={len(sample_files)}"
        )

        condition = task_args.get("condition")

        for sf in sample_files:
            try:
                sample = json.loads(z.read(sf))
            except json.JSONDecodeError:
                sr = SampleReport(
                    sample_id=sf, eval_file=eval_path.name
                )
                sr.error("Invalid JSON")
                report.samples.append(sr)
                continue

            sid = sample.get("id", sf)
            sr = SampleReport(sample_id=sid, eval_file=eval_path.name)

            validate_sample_metadata(sample, sr, verbose)
            validate_sample_messages(sample, sr, verbose, condition=condition)
            validate_sample_scores(sample, sr, strict, verbose)
            validate_sample_store(sample, sr, verbose)

            report.samples.append(sr)

            # Propagate sample status to log
            if sr.status == "error" and report.status != "error":
                report.status = "error"
            elif sr.status == "warn" and report.status == "ok":
                report.status = "warn"

    return report


def print_report(report: LogReport, verbose: bool):
    """Print a formatted report."""
    status_icon = {"ok": "\u2705", "warn": "\u26a0\ufe0f", "error": "\u274c"}
    icon = status_icon.get(report.status, "?")

    print(f"\n{icon} {report.eval_file}")
    for issue in report.issues:
        print(f"  {issue}")

    for sr in report.samples:
        s_icon = status_icon.get(sr.status, "?")
        print(f"  {s_icon} {sr.sample_id}")
        for issue in sr.issues:
            print(f"    {issue}")
        if verbose:
            for info in sr.info:
                print(f"    {info}")


def main():
    parser = argparse.ArgumentParser(
        description="Review BrowserART benign eval logs"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "logs",
        help="Directory containing .eval log files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat pre-fix scorer issues as errors (not warnings)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed info for each sample",
    )
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        print(f"Logs directory not found: {args.logs_dir}", file=sys.stderr)
        sys.exit(1)

    benign_evals = find_benign_evals(args.logs_dir)
    if not benign_evals:
        print("No hbb_benign eval logs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(benign_evals)} benign eval log(s)\n")

    all_reports: list[LogReport] = []
    for eval_path in benign_evals:
        report = review_log(eval_path, args.strict, args.verbose)
        all_reports.append(report)
        print_report(report, args.verbose)

    # Summary
    total_samples = sum(len(r.samples) for r in all_reports)
    errors = sum(
        1 for r in all_reports for s in r.samples if s.status == "error"
    )
    warns = sum(
        1 for r in all_reports for s in r.samples if s.status == "warn"
    )
    ok = sum(
        1 for r in all_reports for s in r.samples if s.status == "ok"
    )

    browser_errors = sum(
        1
        for r in all_reports
        for s in r.samples
        if any("Browser error" in i for i in s.issues)
    )
    pre_fix_scores = sum(
        1
        for r in all_reports
        for s in r.samples
        if any("pre-fix" in i.lower() or "scorer bug" in i.lower() for i in s.issues)
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Logs reviewed:     {len(all_reports)}")
    print(f"Total samples:     {total_samples}")
    print(f"  OK:              {ok}")
    print(f"  Warnings:        {warns}")
    print(f"  Errors:          {errors}")
    print(f"  Browser errors:  {browser_errors}")
    print(f"  Pre-fix scorer:  {pre_fix_scores}")

    if errors > 0:
        sys.exit(2)
    elif warns > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
