"""
Cost efficiency metrics.

Measures resource usage per successful task completion.
"""

from typing import List, Optional
from dataclasses import dataclass


# Approximate token costs (as of late 2024, subject to change)
TOKEN_COSTS = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},  # per 1M tokens
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class CostResult:
    """Cost breakdown for a task execution."""
    task_id: str
    condition: str
    success: bool
    input_tokens: int
    output_tokens: int
    model: str = "claude-sonnet-4-20250514"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD."""
        if self.model not in TOKEN_COSTS:
            return 0.0

        costs = TOKEN_COSTS[self.model]
        input_cost = (self.input_tokens / 1_000_000) * costs["input"]
        output_cost = (self.output_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost


def tokens_per_task(results: List[CostResult]) -> float:
    """
    Calculate average tokens per task attempt.

    Args:
        results: List of cost results

    Returns:
        Average total tokens per task
    """
    if not results:
        return 0.0

    total = sum(r.total_tokens for r in results)
    return total / len(results)


def cost_per_success(results: List[CostResult]) -> float:
    """
    Calculate average cost per successful task.

    Args:
        results: List of cost results

    Returns:
        Average cost in USD per success
    """
    successes = [r for r in results if r.success]

    if not successes:
        return float('inf')

    total_cost = sum(r.estimated_cost for r in successes)
    return total_cost / len(successes)


def tokens_per_success(results: List[CostResult]) -> float:
    """
    Calculate average tokens per successful task.

    Args:
        results: List of cost results

    Returns:
        Average tokens per success
    """
    successes = [r for r in results if r.success]

    if not successes:
        return float('inf')

    total_tokens = sum(r.total_tokens for r in successes)
    return total_tokens / len(successes)


def efficiency_ratio(
    baseline_results: List[CostResult],
    treatment_results: List[CostResult]
) -> float:
    """
    Calculate cost-efficiency ratio between conditions.

    Ratio > 1 means treatment is more efficient.

    Args:
        baseline_results: Results from baseline condition
        treatment_results: Results from treatment condition

    Returns:
        Efficiency ratio (baseline_cost / treatment_cost)
    """
    baseline_cost = cost_per_success(baseline_results)
    treatment_cost = cost_per_success(treatment_results)

    if treatment_cost == float('inf'):
        return 0.0
    if baseline_cost == float('inf'):
        return float('inf')

    return baseline_cost / treatment_cost


def cost_breakdown_by_condition(results: List[CostResult]) -> dict:
    """
    Generate cost summary by condition.

    Args:
        results: List of cost results

    Returns:
        Dict with cost metrics per condition
    """
    by_condition = {}
    for r in results:
        if r.condition not in by_condition:
            by_condition[r.condition] = []
        by_condition[r.condition].append(r)

    summary = {}
    for cond, runs in by_condition.items():
        successes = [r for r in runs if r.success]
        summary[cond] = {
            "attempts": len(runs),
            "successes": len(successes),
            "success_rate": len(successes) / len(runs) if runs else 0,
            "total_cost": sum(r.estimated_cost for r in runs),
            "cost_per_attempt": sum(r.estimated_cost for r in runs) / len(runs) if runs else 0,
            "cost_per_success": cost_per_success(runs),
            "tokens_per_success": tokens_per_success(runs),
        }

    return summary
