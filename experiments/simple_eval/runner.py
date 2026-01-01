#!/usr/bin/env python3
"""
Simple evaluation runner for idle comparison.

This runner executes a defined task using Claude Code and measures:
- Success/failure
- Token usage (from output)
- Steps taken
- Time elapsed

To compare conditions:
- baseline: Run with IDLE_DISABLED=1 environment variable
- idle-full: Run normally (idle hooks active)
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class EvalResult:
    """Result of a single evaluation run."""
    task_id: str
    condition: str
    run_id: int
    success: bool
    duration_seconds: float
    output_lines: int
    error: Optional[str] = None


# Simple, verifiable tasks
TASKS = {
    "fizzbuzz": {
        "prompt": """Create a file called fizzbuzz.py that:
1. Prints numbers 1-100
2. For multiples of 3, print "Fizz" instead
3. For multiples of 5, print "Buzz" instead
4. For multiples of both 3 and 5, print "FizzBuzz"

Then run the file and show the output.""",
        "verify": lambda d: (d / "fizzbuzz.py").exists(),
    },
    "calculator": {
        "prompt": """Create a file called calc.py with a Calculator class that has:
- add(a, b) method
- subtract(a, b) method
- multiply(a, b) method
- divide(a, b) method with zero division handling

Then create test_calc.py with pytest tests for each method.
Run the tests and show they pass.""",
        "verify": lambda d: (d / "calc.py").exists() and (d / "test_calc.py").exists(),
    },
    "api_mock": {
        "prompt": """Create a simple Flask API in app.py with:
- GET /health returning {"status": "ok"}
- GET /users returning a list of 3 mock users
- POST /users that accepts JSON and returns the created user

Include basic error handling. Don't actually run the server.""",
        "verify": lambda d: (d / "app.py").exists(),
    },
}


def run_task(
    task_id: str,
    condition: str,
    run_id: int,
    work_dir: Path,
    timeout: int = 300
) -> EvalResult:
    """Execute a task with Claude Code."""

    task = TASKS.get(task_id)
    if not task:
        return EvalResult(
            task_id=task_id,
            condition=condition,
            run_id=run_id,
            success=False,
            duration_seconds=0,
            output_lines=0,
            error=f"Unknown task: {task_id}"
        )

    # Create task-specific directory
    task_dir = work_dir / f"{task_id}_{condition}_run{run_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Set environment based on condition
    env = os.environ.copy()
    if condition == "baseline":
        env["IDLE_DISABLED"] = "1"
        # Also create .idle-disabled file
        (task_dir / ".idle-disabled").touch()
    elif condition == "idle-no-alice":
        # idle enabled but alice review disabled
        (task_dir / ".idle-disabled").touch()  # Disables stop hook review requirement

    start_time = time.time()

    try:
        result = subprocess.run(
            ["claude", "--print", "--dangerously-skip-permissions", task["prompt"]],
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        duration = time.time() - start_time
        output = result.stdout + result.stderr
        output_lines = len(output.strip().split("\n"))

        # Verify task completion
        success = task["verify"](task_dir)

        return EvalResult(
            task_id=task_id,
            condition=condition,
            run_id=run_id,
            success=success,
            duration_seconds=duration,
            output_lines=output_lines
        )

    except subprocess.TimeoutExpired:
        return EvalResult(
            task_id=task_id,
            condition=condition,
            run_id=run_id,
            success=False,
            duration_seconds=timeout,
            output_lines=0,
            error="TIMEOUT"
        )
    except Exception as e:
        return EvalResult(
            task_id=task_id,
            condition=condition,
            run_id=run_id,
            success=False,
            duration_seconds=time.time() - start_time,
            output_lines=0,
            error=str(e)
        )


def run_experiment(
    task_id: str,
    condition: str,
    runs: int = 3,
    output_dir: Optional[Path] = None
):
    """Run experiment for a single task/condition combination."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(__file__).parent.parent.parent / "workspaces" / "simple_eval"
    work_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"simple_{task_id}_{condition}_{timestamp}.jsonl"

    print(f"Running {task_id} × {runs} runs for condition: {condition}")

    results = []
    for run_id in range(1, runs + 1):
        print(f"  Run {run_id}/{runs}...", end=" ", flush=True)

        result = run_task(task_id, condition, run_id, work_dir)
        results.append(result)

        status = "PASS" if result.success else "FAIL"
        print(f"{status} ({result.duration_seconds:.1f}s)")

        with open(results_file, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    # Summary
    successes = sum(1 for r in results if r.success)
    print(f"\nSummary: {successes}/{runs} succeeded ({successes/runs*100:.0f}%)")
    print(f"Results: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run simple evaluation experiment")
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()),
                       help="Task to run")
    parser.add_argument("--condition", required=True,
                       choices=["baseline", "idle-full", "idle-no-alice"],
                       help="Evaluation condition")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory")

    args = parser.parse_args()

    run_experiment(
        task_id=args.task,
        condition=args.condition,
        runs=args.runs,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
