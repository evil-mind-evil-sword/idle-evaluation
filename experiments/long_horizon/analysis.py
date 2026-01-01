#!/usr/bin/env python3
"""
Analysis utilities for long-horizon experiments.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics.pass_k import aggregate_pass_power_k, reliability_curve
from metrics.task_completion import completion_rate, completion_by_condition, improvement_matrix


def load_results(results_dir: Path, prefix: str = "long-horizon") -> List[dict]:
    """Load long-horizon results from JSONL files."""
    results = []

    for jsonl_file in results_dir.glob(f"{prefix}*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    return results


def analyze_experiment(results: List[dict]) -> Dict[str, Any]:
    """Generate comprehensive analysis of experiment results."""
    if not results:
        return {"error": "No results to analyze"}

    # Group by condition and task
    by_condition = {}
    by_task = {}

    for r in results:
        cond = r.get("condition", "unknown")
        task = r.get("task_id", "unknown")

        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)

        key = (cond, task)
        if key not in by_task:
            by_task[key] = []
        by_task[key].append(r.get("success", False))

    analysis = {
        "total_runs": len(results),
        "conditions": list(by_condition.keys()),
        "tasks": list(set(r.get("task_id") for r in results)),
    }

    # Completion rates by condition
    analysis["completion_rates"] = {}
    for cond, runs in by_condition.items():
        successes = sum(1 for r in runs if r.get("success", False))
        analysis["completion_rates"][cond] = {
            "rate": successes / len(runs) if runs else 0,
            "successes": successes,
            "total": len(runs)
        }

    # pass^k analysis
    analysis["pass_k"] = {}
    for cond in by_condition.keys():
        task_results = [
            by_task[(cond, task)]
            for task in analysis["tasks"]
            if (cond, task) in by_task
        ]

        if task_results:
            curve = reliability_curve(task_results, max_k=5)
            analysis["pass_k"][cond] = {
                f"pass^{k+1}": curve[k] if k < len(curve) else None
                for k in range(5)
            }

    # Efficiency metrics
    analysis["efficiency"] = {}
    for cond, runs in by_condition.items():
        steps = [r.get("steps", 0) for r in runs]
        tokens = [r.get("tokens", 0) for r in runs]
        durations = [r.get("duration_seconds", 0) for r in runs]

        analysis["efficiency"][cond] = {
            "avg_steps": sum(steps) / len(steps) if steps else 0,
            "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
        }

    # Improvement over baseline
    if "baseline" in by_condition:
        baseline_rate = analysis["completion_rates"]["baseline"]["rate"]
        analysis["improvement_over_baseline"] = {}

        for cond in by_condition.keys():
            if cond != "baseline":
                treatment_rate = analysis["completion_rates"][cond]["rate"]
                analysis["improvement_over_baseline"][cond] = {
                    "absolute": treatment_rate - baseline_rate,
                    "relative": (treatment_rate - baseline_rate) / baseline_rate if baseline_rate > 0 else None
                }

    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Pretty print analysis results."""
    print("\n" + "=" * 60)
    print("LONG-HORIZON EXPERIMENT ANALYSIS")
    print("=" * 60)

    print(f"\nTotal runs: {analysis.get('total_runs', 0)}")
    print(f"Conditions: {', '.join(analysis.get('conditions', []))}")
    print(f"Tasks: {len(analysis.get('tasks', []))}")

    print("\n--- Completion Rates ---")
    for cond, data in analysis.get("completion_rates", {}).items():
        rate = data["rate"] * 100
        print(f"  {cond:20} {rate:5.1f}% ({data['successes']}/{data['total']})")

    print("\n--- pass^k Reliability ---")
    for cond, data in analysis.get("pass_k", {}).items():
        values = " | ".join(
            f"k={k}: {v*100:.1f}%" if v is not None else f"k={k}: N/A"
            for k, v in sorted((int(k.split('^')[1]), v) for k, v in data.items())
        )
        print(f"  {cond:20} {values}")

    print("\n--- Efficiency ---")
    for cond, data in analysis.get("efficiency", {}).items():
        print(f"  {cond:20} steps={data['avg_steps']:.1f} tokens={data['avg_tokens']:.0f} duration={data['avg_duration']:.1f}s")

    if "improvement_over_baseline" in analysis:
        print("\n--- Improvement over Baseline ---")
        for cond, data in analysis["improvement_over_baseline"].items():
            abs_imp = data["absolute"] * 100
            rel_imp = data["relative"] * 100 if data["relative"] is not None else 0
            print(f"  {cond:20} +{abs_imp:.1f}pp ({rel_imp:+.1f}% relative)")

    print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze long-horizon experiment results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, help="Save analysis to JSON file")

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    results_dir = repo_root / args.results_dir

    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    analysis = analyze_experiment(results)
    print_analysis(analysis)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()
