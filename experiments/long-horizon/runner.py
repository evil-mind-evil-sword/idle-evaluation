#!/usr/bin/env python3
"""
Long-horizon task experiment runner.

Inspired by τ-Bench: measures reliability of multi-step conversational
coding tasks with pass^k metric.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add metrics to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics.task_completion import TaskResult
from metrics.pass_k import pass_power_k, aggregate_pass_power_k


@dataclass
class LongHorizonTask:
    """Definition of a long-horizon task."""
    id: str
    name: str
    description: str
    prompt: str
    expected_steps: int
    success_criteria: List[str]
    verification_script: Optional[str] = None


def load_tasks(tasks_dir: Path) -> List[LongHorizonTask]:
    """Load task definitions from JSON files."""
    tasks = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        with open(task_file) as f:
            data = json.load(f)
            tasks.append(LongHorizonTask(**data))
    return tasks


def load_harness_config(condition: str) -> dict:
    """Load harness configuration for a condition."""
    config_path = Path(__file__).parent.parent.parent / "harnesses" / condition / "config.json"

    # Handle condition name mapping
    condition_dirs = {
        "baseline": "baseline",
        "idle-full": "idle",
        "idle-no-alice": "idle-no-review",
        "idle-sonnet-alice": "idle-no-consensus",
    }

    if condition in condition_dirs:
        config_path = Path(__file__).parent.parent.parent / "harnesses" / condition_dirs[condition] / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Harness config not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def run_task(
    task: LongHorizonTask,
    config: dict,
    run_id: int,
    work_dir: Path,
    timeout: int = 600
) -> TaskResult:
    """
    Execute a single task run with Claude Code.

    Args:
        task: Task definition
        config: Harness configuration
        run_id: Run number for this task
        work_dir: Working directory for task execution
        timeout: Maximum execution time in seconds

    Returns:
        TaskResult with execution details
    """
    start_time = time.time()

    # Set up environment
    env = dict(config.get("environment", {}))

    # Create task-specific working directory
    task_work_dir = work_dir / f"{task.id}_run{run_id}"
    task_work_dir.mkdir(parents=True, exist_ok=True)

    # Build Claude Code command
    cmd = ["claude", "--print", "--dangerously-skip-permissions"]
    cmd.extend(config.get("claude_code_args", []))
    cmd.append(task.prompt)

    try:
        result = subprocess.run(
            cmd,
            cwd=task_work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**dict(__import__('os').environ), **env}
        )

        output = result.stdout + result.stderr
        duration = time.time() - start_time

        # Verify success using criteria or script
        success = verify_task_success(task, task_work_dir, output)

        # Count steps (approximate from conversation)
        steps = count_conversation_steps(output)

        # Estimate tokens (approximate)
        tokens = len(output.split()) * 2  # Rough approximation

        return TaskResult(
            task_id=task.id,
            condition=config["name"],
            run_id=run_id,
            success=success,
            steps=steps,
            tokens=tokens,
            duration_seconds=duration,
            errors=extract_errors(output)
        )

    except subprocess.TimeoutExpired:
        return TaskResult(
            task_id=task.id,
            condition=config["name"],
            run_id=run_id,
            success=False,
            steps=0,
            tokens=0,
            duration_seconds=timeout,
            errors=["TIMEOUT"]
        )
    except Exception as e:
        return TaskResult(
            task_id=task.id,
            condition=config["name"],
            run_id=run_id,
            success=False,
            steps=0,
            tokens=0,
            duration_seconds=time.time() - start_time,
            errors=[str(e)]
        )


def verify_task_success(task: LongHorizonTask, work_dir: Path, output: str) -> bool:
    """Verify if task completed successfully."""
    # If verification script provided, run it
    if task.verification_script:
        script_path = Path(__file__).parent / "tasks" / task.verification_script
        if script_path.exists():
            try:
                result = subprocess.run(
                    ["bash", str(script_path)],
                    cwd=work_dir,
                    capture_output=True,
                    timeout=30
                )
                return result.returncode == 0
            except Exception:
                return False

    # Otherwise check success criteria in output
    for criterion in task.success_criteria:
        if criterion.startswith("file:"):
            # Check file exists
            file_path = work_dir / criterion[5:]
            if not file_path.exists():
                return False
        elif criterion.startswith("contains:"):
            # Check output contains string
            if criterion[9:] not in output:
                return False
        elif criterion.startswith("exit:"):
            # Already handled by subprocess returncode
            pass

    return True


def count_conversation_steps(output: str) -> int:
    """Estimate number of agent steps from output."""
    # Count tool calls or message boundaries
    markers = ["<tool>", "```", "Human:", "Assistant:"]
    count = sum(output.count(m) for m in markers)
    return max(1, count // 2)


def extract_errors(output: str) -> List[str]:
    """Extract error messages from output."""
    errors = []
    error_markers = ["Error:", "error:", "Exception:", "FAILED", "TypeError", "ValueError"]

    for line in output.split("\n"):
        for marker in error_markers:
            if marker in line:
                errors.append(line.strip()[:200])
                break

    return errors[:10]  # Limit to 10 errors


def run_experiment(
    condition: str,
    runs: int = 5,
    tasks_subset: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
):
    """
    Run full experiment for a condition.

    Args:
        condition: Condition name (baseline, idle-full, etc.)
        runs: Number of runs per task
        tasks_subset: Optional list of task IDs to run
        output_dir: Directory for results
    """
    tasks_dir = Path(__file__).parent / "tasks"
    tasks = load_tasks(tasks_dir)

    if tasks_subset:
        tasks = [t for t in tasks if t.id in tasks_subset]

    if not tasks:
        print("No tasks found. Add task definitions to experiments/long-horizon/tasks/")
        return

    config = load_harness_config(condition)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(__file__).parent.parent.parent / "workspaces" / "long-horizon"
    work_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"long-horizon_{condition}_{timestamp}.jsonl"

    print(f"Running {len(tasks)} tasks × {runs} runs for condition: {condition}")

    all_results = []
    task_results = {}  # For pass^k calculation

    for task in tasks:
        print(f"\n=== Task: {task.id} ({task.name}) ===")
        task_results[task.id] = []

        for run_id in range(1, runs + 1):
            print(f"  Run {run_id}/{runs}...", end=" ", flush=True)

            result = run_task(task, config, run_id, work_dir)
            all_results.append(result)
            task_results[task.id].append(result.success)

            status = "PASS" if result.success else "FAIL"
            print(f"{status} ({result.steps} steps, {result.duration_seconds:.1f}s)")

            # Write result immediately
            with open(results_file, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")

    # Calculate summary metrics
    print(f"\n=== Summary for {condition} ===")
    from metrics.task_completion import completion_rate
    from metrics.pass_k import aggregate_pass_power_k

    overall_rate = completion_rate(all_results)
    print(f"Overall completion rate: {overall_rate:.1%}")

    pass_k_results = [task_results[t.id] for t in tasks]
    for k in [1, 3, 5]:
        if runs >= k:
            pk = aggregate_pass_power_k(pass_k_results, k)
            print(f"pass^{k}: {pk:.1%}")

    print(f"\nResults written to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run long-horizon reliability experiments")
    parser.add_argument("--condition", required=True,
                       choices=["baseline", "idle-full", "idle-no-alice", "idle-sonnet-alice"],
                       help="Evaluation condition")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of runs per task")
    parser.add_argument("--tasks", nargs="+",
                       help="Specific task IDs to run (default: all)")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for results")

    args = parser.parse_args()

    run_experiment(
        condition=args.condition,
        runs=args.runs,
        tasks_subset=args.tasks,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
