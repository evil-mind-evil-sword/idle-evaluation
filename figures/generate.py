#!/usr/bin/env python3
"""
Figure generation for idle-evaluation.

Produces publication-quality figures comparing evaluation conditions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "baseline": "#E24A33",      # Red
    "idle-full": "#348ABD",     # Blue
    "idle-no-alice": "#988ED5", # Purple
    "idle-sonnet-alice": "#777777",  # Gray
}
CONDITION_LABELS = {
    "baseline": "Baseline",
    "idle-full": "idle (full)",
    "idle-no-alice": "idle (no alice)",
    "idle-sonnet-alice": "idle (Sonnet alice)",
}


def load_results(results_dir: Path) -> Dict[str, List[dict]]:
    """Load all results grouped by experiment type."""
    results = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        experiment_type = jsonl_file.stem.split("_")[0]

        if experiment_type not in results:
            results[experiment_type] = []

        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    results[experiment_type].append(json.loads(line))

    return results


def plot_completion_rates(results: List[dict], output_path: Path):
    """
    Bar chart comparing completion rates across conditions.
    """
    from metrics.task_completion import completion_by_condition

    # Convert to TaskResult-like dicts
    rates = {}
    for r in results:
        cond = r.get("condition", "unknown")
        if cond not in rates:
            rates[cond] = {"successes": 0, "total": 0}
        rates[cond]["total"] += 1
        if r.get("success", False):
            rates[cond]["successes"] += 1

    conditions = list(rates.keys())
    values = [rates[c]["successes"] / rates[c]["total"] * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        range(len(conditions)),
        values,
        color=[COLORS.get(c, "#333333") for c in conditions],
        edgecolor="white",
        linewidth=1.5
    )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=11)
    ax.set_ylabel("Completion Rate (%)", fontsize=12)
    ax.set_title("Task Completion Rate by Condition", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)

    # Add value labels (inside bars if at 100%)
    for bar, val in zip(bars, values):
        y_pos = bar.get_height() - 5 if val >= 95 else bar.get_height() + 2
        va = "top" if val >= 95 else "bottom"
        color = "white" if val >= 95 else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:.1f}%",
            ha="center",
            va=va,
            fontsize=10,
            color=color,
            fontweight="bold" if val >= 95 else "normal"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_pass_k_curve(results: List[dict], output_path: Path, max_k: int = 5):
    """
    Line chart showing pass^k reliability degradation.
    """
    from metrics.pass_k import aggregate_pass_power_k

    # Group by condition and task
    by_condition = {}
    for r in results:
        cond = r.get("condition", "unknown")
        task_id = r.get("task_id", "unknown")

        if cond not in by_condition:
            by_condition[cond] = {}
        if task_id not in by_condition[cond]:
            by_condition[cond][task_id] = []

        by_condition[cond][task_id].append(r.get("success", False))

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond, tasks in by_condition.items():
        task_results = list(tasks.values())

        # Calculate pass^k for k=1 to max_k
        ks = range(1, max_k + 1)
        pk_values = []

        for k in ks:
            # Filter tasks with enough runs
            valid_tasks = [t for t in task_results if len(t) >= k]
            if valid_tasks:
                pk = aggregate_pass_power_k(valid_tasks, k)
                pk_values.append(pk * 100)
            else:
                pk_values.append(0)

        ax.plot(
            ks,
            pk_values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=CONDITION_LABELS.get(cond, cond),
            color=COLORS.get(cond, "#333333")
        )

    ax.set_xlabel("k (required successes)", fontsize=12)
    ax.set_ylabel("pass^k (%)", fontsize=12)
    ax.set_title("Reliability Degradation (pass^k)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(1, max_k + 1))
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_accumulation(results: List[dict], output_path: Path):
    """
    Line chart showing error rate over steps (for error-correction experiments).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        cond = r.get("condition", "unknown")
        errors = r.get("errors", [])

        if not errors:
            continue

        # Build cumulative error curve
        steps = [e.get("step_id", i) for i, e in enumerate(errors)]
        cumulative = np.arange(1, len(errors) + 1)

        ax.scatter(
            steps,
            cumulative,
            alpha=0.3,
            color=COLORS.get(cond, "#333333"),
            s=20
        )

    # Add legend
    handles = [
        mpatches.Patch(color=COLORS.get(c, "#333333"), label=CONDITION_LABELS.get(c, c))
        for c in set(r.get("condition") for r in results)
    ]
    ax.legend(handles=handles, loc="upper left")

    ax.set_xlabel("Step Number", fontsize=12)
    ax.set_ylabel("Cumulative Errors", fontsize=12)
    ax.set_title("Error Accumulation Over Steps", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_rate_comparison(results: List[dict], output_path: Path):
    """
    Box plot comparing per-step error rates.
    """
    from metrics.error_rate import per_step_error_rate

    by_condition = {}
    for r in results:
        cond = r.get("condition", "unknown")
        error_rate = r.get("error_rate", 0)

        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(error_rate * 100)

    conditions = list(by_condition.keys())
    data = [by_condition[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        data,
        labels=[CONDITION_LABELS.get(c, c) for c in conditions],
        patch_artist=True
    )

    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(COLORS.get(cond, "#333333"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Per-Step Error Rate (%)", fontsize=12)
    ax.set_title("Error Rate Distribution by Condition", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_cost_efficiency(results: List[dict], output_path: Path):
    """
    Scatter plot of success rate vs cost.
    """
    by_condition = {}
    for r in results:
        cond = r.get("condition", "unknown")
        if cond not in by_condition:
            by_condition[cond] = {"successes": 0, "total": 0, "tokens": 0}

        by_condition[cond]["total"] += 1
        by_condition[cond]["tokens"] += r.get("tokens", 0)
        if r.get("success", False):
            by_condition[cond]["successes"] += 1

    fig, ax = plt.subplots(figsize=(8, 6))

    for cond, data in by_condition.items():
        success_rate = data["successes"] / data["total"] * 100 if data["total"] > 0 else 0
        avg_tokens = data["tokens"] / data["total"] / 1000 if data["total"] > 0 else 0  # In K

        ax.scatter(
            avg_tokens,
            success_rate,
            s=200,
            color=COLORS.get(cond, "#333333"),
            label=CONDITION_LABELS.get(cond, cond),
            edgecolors="white",
            linewidth=2
        )

    ax.set_xlabel("Average Tokens per Task (K)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Cost-Efficiency Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate all figures from results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    # Combine all results for overall figures
    all_results = []
    for experiment_results in results.values():
        all_results.extend(experiment_results)

    if all_results:
        plot_completion_rates(all_results, output_dir / "completion_rates.png")
        plot_pass_k_curve(all_results, output_dir / "pass_k_curve.png")
        plot_cost_efficiency(all_results, output_dir / "cost_efficiency.png")

    # Error-specific figures
    if "error-correction" in results:
        plot_error_accumulation(results["error-correction"], output_dir / "error_accumulation.png")
        plot_error_rate_comparison(results["error-correction"], output_dir / "error_rate_box.png")

    print(f"\nAll figures saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/output"))

    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / args.results_dir if not args.results_dir.is_absolute() else args.results_dir
    output_dir = repo_root / args.output_dir if not args.output_dir.is_absolute() else args.output_dir

    generate_all_figures(results_dir, output_dir)


if __name__ == "__main__":
    main()
