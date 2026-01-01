#!/usr/bin/env python3
"""
τ-bench integration for idle-evaluation.

τ-bench tests long-horizon, tool-enabled conversational workflows with
realistic human-in-the-loop conditions. Key metrics include pass^k reliability.

Source: https://github.com/sierra-research/tau-bench
Paper: https://arxiv.org/abs/2406.12045

Prerequisites:
    pip install git+https://github.com/sierra-research/tau-bench
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_tau_bench_installed() -> bool:
    """Check if tau-bench is installed."""
    try:
        import tau_bench
        return True
    except ImportError:
        return False


def install_tau_bench():
    """Install tau-bench from GitHub."""
    print("Installing tau-bench...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "git+https://github.com/sierra-research/tau-bench"],
        check=True
    )


def run_tau_bench(
    env: str,
    condition: str,
    model: str = "claude-sonnet-4-20250514",
    user_model: str = "gpt-4o",
    task_ids: Optional[List[int]] = None,
    max_concurrency: int = 5,
    num_trials: int = 1,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run τ-bench evaluation.

    Args:
        env: Environment to test (retail, airline)
        condition: Evaluation condition (baseline, idle-full, etc.)
        model: Agent model to use
        user_model: User simulator model
        task_ids: Specific task IDs to run (None = all)
        max_concurrency: Max parallel tasks
        num_trials: Number of trials per task
        output_dir: Directory for results

    Returns:
        Dict with results summary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment for condition
    env_vars = os.environ.copy()
    if condition == "baseline":
        env_vars["IDLE_DISABLED"] = "1"

    # Build command
    cmd = [
        sys.executable, "-m", "tau_bench.run",
        "--agent-strategy", "tool-calling",
        "--env", env,
        "--model", model,
        "--model-provider", "anthropic",
        "--user-model", user_model,
        "--user-model-provider", "openai",
        "--user-strategy", "llm",
        "--max-concurrency", str(max_concurrency),
        "--num-trials", str(num_trials),
    ]

    if task_ids:
        cmd.extend(["--task-ids"] + [str(t) for t in task_ids])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"tau_bench_{env}_{condition}_{timestamp}.json"

    print(f"Running τ-bench {env} environment")
    print(f"Condition: {condition}")
    print(f"Model: {model}")
    print(f"Trials per task: {num_trials}")
    print()

    try:
        result = subprocess.run(
            cmd,
            env=env_vars,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        output = result.stdout + result.stderr

        # Parse results from output
        # τ-bench outputs JSON results
        results = {
            "env": env,
            "condition": condition,
            "model": model,
            "num_trials": num_trials,
            "timestamp": timestamp,
            "raw_output": output,
            "returncode": result.returncode,
        }

        # Try to extract metrics from output
        try:
            # τ-bench typically outputs metrics in specific format
            if "pass@1" in output.lower():
                # Extract pass@k values
                pass
        except Exception:
            pass

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        return results

    except subprocess.TimeoutExpired:
        return {
            "env": env,
            "condition": condition,
            "error": "TIMEOUT",
            "timestamp": timestamp,
        }
    except Exception as e:
        return {
            "env": env,
            "condition": condition,
            "error": str(e),
            "timestamp": timestamp,
        }


def run_comparison(
    env: str = "retail",
    task_ids: Optional[List[int]] = None,
    num_trials: int = 3,
    output_dir: Optional[Path] = None,
):
    """
    Run τ-bench comparison between baseline and idle-full.

    This is the main entry point for idle evaluation.
    """
    print("=" * 60)
    print("τ-bench Comparison: baseline vs idle-full")
    print("=" * 60)

    conditions = ["baseline", "idle-full"]
    all_results = {}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Running condition: {condition}")
        print("=" * 60)

        results = run_tau_bench(
            env=env,
            condition=condition,
            task_ids=task_ids,
            num_trials=num_trials,
            output_dir=output_dir,
        )
        all_results[condition] = results

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for condition, results in all_results.items():
        if "error" in results:
            print(f"{condition}: ERROR - {results['error']}")
        else:
            print(f"{condition}: Completed (see results file)")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run τ-bench evaluation for idle comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run retail domain comparison
    python runner.py --env retail --trials 3

    # Run specific tasks
    python runner.py --env retail --task-ids 1 2 3 --trials 5

    # Run single condition
    python runner.py --env airline --condition baseline --trials 3
        """
    )

    parser.add_argument("--env", default="retail", choices=["retail", "airline"],
                       help="τ-bench environment")
    parser.add_argument("--condition", choices=["baseline", "idle-full", "comparison"],
                       default="comparison",
                       help="Run single condition or comparison")
    parser.add_argument("--task-ids", type=int, nargs="+",
                       help="Specific task IDs to run")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials per task (for pass^k)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                       help="Agent model")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for results")
    parser.add_argument("--install", action="store_true",
                       help="Install tau-bench if not present")

    args = parser.parse_args()

    # Check/install tau-bench
    if not check_tau_bench_installed():
        if args.install:
            install_tau_bench()
        else:
            print("τ-bench not installed. Run with --install or:")
            print("  pip install git+https://github.com/sierra-research/tau-bench")
            sys.exit(1)

    # Run evaluation
    if args.condition == "comparison":
        run_comparison(
            env=args.env,
            task_ids=args.task_ids,
            num_trials=args.trials,
            output_dir=args.output_dir,
        )
    else:
        run_tau_bench(
            env=args.env,
            condition=args.condition,
            model=args.model,
            task_ids=args.task_ids,
            num_trials=args.trials,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
