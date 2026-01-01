#!/usr/bin/env python3
"""
SWE-bench subset experiment runner.

Evaluates Claude Code on a curated subset of real GitHub issues
from the SWE-bench benchmark.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics.task_completion import TaskResult


@dataclass
class SWEBenchIssue:
    """A SWE-bench issue definition."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    test_patch: str
    patch: str  # Ground truth (for verification only)
    difficulty: str  # easy, medium, hard


def load_issues(issues_file: Path) -> List[SWEBenchIssue]:
    """Load curated SWE-bench issues."""
    if not issues_file.exists():
        return []

    with open(issues_file) as f:
        data = json.load(f)

    return [SWEBenchIssue(**issue) for issue in data]


def setup_repo(issue: SWEBenchIssue, work_dir: Path) -> Path:
    """Clone and checkout repo at base commit."""
    repo_dir = work_dir / issue.instance_id

    if repo_dir.exists():
        # Reset to base commit
        subprocess.run(
            ["git", "checkout", issue.base_commit, "--force"],
            cwd=repo_dir,
            capture_output=True
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=repo_dir,
            capture_output=True
        )
    else:
        # Clone fresh
        subprocess.run(
            ["git", "clone", f"https://github.com/{issue.repo}.git", str(repo_dir)],
            capture_output=True
        )
        subprocess.run(
            ["git", "checkout", issue.base_commit],
            cwd=repo_dir,
            capture_output=True
        )

    return repo_dir


def run_tests(repo_dir: Path, test_patch: str) -> bool:
    """
    Apply test patch and run tests.

    Returns True if all tests pass.
    """
    # Apply test patch
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(test_patch)
        patch_file = f.name

    try:
        result = subprocess.run(
            ["git", "apply", patch_file],
            cwd=repo_dir,
            capture_output=True
        )

        if result.returncode != 0:
            return False

        # Run pytest
        result = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-q"],
            cwd=repo_dir,
            capture_output=True,
            timeout=120
        )

        return result.returncode == 0

    except Exception:
        return False
    finally:
        Path(patch_file).unlink(missing_ok=True)


def run_issue(
    issue: SWEBenchIssue,
    config: dict,
    run_id: int,
    work_dir: Path
) -> TaskResult:
    """
    Attempt to solve a SWE-bench issue with Claude Code.
    """
    start_time = time.time()

    try:
        repo_dir = setup_repo(issue, work_dir)
    except Exception as e:
        return TaskResult(
            task_id=issue.instance_id,
            condition=config["name"],
            run_id=run_id,
            success=False,
            errors=[f"Setup failed: {e}"]
        )

    prompt = f"""Fix the following GitHub issue in this repository:

## Issue
{issue.problem_statement}

## Hints
{issue.hints_text}

## Instructions
1. Understand the issue and locate the relevant code
2. Implement a fix
3. Verify your fix works

Make the minimal necessary changes to fix the issue.
"""

    env = dict(config.get("environment", {}))
    cmd = ["claude", "--print", "--dangerously-skip-permissions", prompt]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300,
            env={**dict(__import__('os').environ), **env}
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        # Check if tests pass
        success = run_tests(repo_dir, issue.test_patch)

        # Count approximate steps
        steps = output.count("```") // 2 + output.count("Edit") + output.count("Write")

        # Rough token estimate
        tokens = len(output.split()) * 2

        return TaskResult(
            task_id=issue.instance_id,
            condition=config["name"],
            run_id=run_id,
            success=success,
            steps=max(1, steps),
            tokens=tokens,
            duration_seconds=duration
        )

    except subprocess.TimeoutExpired:
        return TaskResult(
            task_id=issue.instance_id,
            condition=config["name"],
            run_id=run_id,
            success=False,
            duration_seconds=300,
            errors=["TIMEOUT"]
        )
    except Exception as e:
        return TaskResult(
            task_id=issue.instance_id,
            condition=config["name"],
            run_id=run_id,
            success=False,
            duration_seconds=time.time() - start_time,
            errors=[str(e)]
        )


def run_experiment(
    condition: str,
    runs: int = 5,
    issues_subset: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
):
    """Run SWE-bench subset experiment."""
    from experiments.long_horizon.runner import load_harness_config

    issues_file = Path(__file__).parent / "selected_issues.json"
    issues = load_issues(issues_file)

    if not issues:
        print("No issues found. Add issues to experiments/swe-bench/selected_issues.json")
        print("\nExpected format:")
        print(json.dumps({
            "instance_id": "repo__issue__commit",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "problem_statement": "Description of the bug...",
            "hints_text": "Helpful context...",
            "test_patch": "diff --git a/test...",
            "patch": "diff --git a/src...",
            "difficulty": "easy"
        }, indent=2))
        return

    if issues_subset:
        issues = [i for i in issues if i.instance_id in issues_subset]

    config = load_harness_config(condition)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(__file__).parent.parent.parent / "workspaces" / "swe-bench"
    work_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"swe-bench_{condition}_{timestamp}.jsonl"

    print(f"Running {len(issues)} issues × {runs} runs for condition: {condition}")

    all_results = []

    for issue in issues:
        print(f"\n=== Issue: {issue.instance_id} ({issue.difficulty}) ===")

        for run_id in range(1, runs + 1):
            print(f"  Run {run_id}/{runs}...", end=" ", flush=True)

            result = run_issue(issue, config, run_id, work_dir)
            all_results.append(result)

            status = "PASS" if result.success else "FAIL"
            print(f"{status} ({result.duration_seconds:.1f}s)")

            with open(results_file, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")

    # Summary
    print(f"\n=== Summary for {condition} ===")
    from metrics.task_completion import completion_rate

    by_difficulty = {}
    for issue in issues:
        if issue.difficulty not in by_difficulty:
            by_difficulty[issue.difficulty] = []
        by_difficulty[issue.difficulty].extend(
            [r for r in all_results if r.task_id == issue.instance_id]
        )

    print(f"Overall: {completion_rate(all_results):.1%}")
    for diff, results in sorted(by_difficulty.items()):
        print(f"  {diff}: {completion_rate(results):.1%}")

    print(f"\nResults written to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench subset experiments")
    parser.add_argument("--condition", required=True,
                       choices=["baseline", "idle-full", "idle-no-alice", "idle-sonnet-alice"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--issues", nargs="+", help="Specific issue IDs")
    parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()

    run_experiment(
        condition=args.condition,
        runs=args.runs,
        issues_subset=args.issues,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
