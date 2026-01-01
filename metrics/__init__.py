"""
idle-evaluation metrics library.

Implements evaluation metrics from:
- τ-Bench: pass^k reliability
- MAKER: per-step error rate
- Standard: task completion, cost efficiency
"""

from .pass_k import pass_at_k, pass_power_k
from .error_rate import per_step_error_rate, first_error_step
from .task_completion import completion_rate, graded_completion
from .cost_efficiency import cost_per_success, tokens_per_task

__all__ = [
    "pass_at_k",
    "pass_power_k",
    "per_step_error_rate",
    "first_error_step",
    "completion_rate",
    "graded_completion",
    "cost_per_success",
    "tokens_per_task",
]


def summarize(results_dir: str) -> dict:
    """Generate summary statistics from experiment results."""
    import json
    from pathlib import Path

    results_path = Path(results_dir)
    all_results = []

    for jsonl_file in results_path.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    if not all_results:
        return {"error": "No results found"}

    # Group by condition
    by_condition = {}
    for r in all_results:
        cond = r.get("condition", "unknown")
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)

    summary = {}
    for cond, runs in by_condition.items():
        successes = [r for r in runs if r.get("success", False)]
        summary[cond] = {
            "total_runs": len(runs),
            "successes": len(successes),
            "completion_rate": len(successes) / len(runs) if runs else 0,
            "avg_steps": sum(r.get("steps", 0) for r in runs) / len(runs) if runs else 0,
            "avg_tokens": sum(r.get("tokens", 0) for r in runs) / len(runs) if runs else 0,
        }

    print(json.dumps(summary, indent=2))
    return summary
