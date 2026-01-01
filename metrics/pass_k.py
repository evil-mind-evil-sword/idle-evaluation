"""
pass@k and pass^k metrics from τ-Bench.

pass@k: Probability that at least 1 of k samples passes
pass^k: Probability that ALL k samples pass (reliability metric)
"""

from typing import List
import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k: probability of at least one success in k samples.

    From Chen et al. "Evaluating Large Language Models Trained on Code" (2021)

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to draw

    Returns:
        Estimated pass@k probability
    """
    if n - c < k:
        return 1.0

    # Use log to avoid overflow: 1 - C(n-c, k) / C(n, k)
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pass_power_k(results: List[bool], k: int = None) -> float:
    """
    Calculate pass^k: fraction of tasks where ALL k attempts succeed.

    From τ-Bench: Measures reliability, not just capability.
    A system with 90% pass@1 might have much lower pass^5.

    Args:
        results: List of boolean success values for repeated runs
        k: Number of runs to require all passing (default: len(results))

    Returns:
        pass^k score (0.0 to 1.0)

    Example:
        >>> pass_power_k([True, True, True, True, True])  # All 5 pass
        1.0
        >>> pass_power_k([True, True, True, True, False])  # 1 failure
        0.0
        >>> # For multiple tasks:
        >>> task_results = [[True]*5, [True]*5, [True, True, True, True, False]]
        >>> sum(pass_power_k(t) for t in task_results) / len(task_results)
        0.666...
    """
    if k is None:
        k = len(results)

    if len(results) < k:
        raise ValueError(f"Need at least {k} results, got {len(results)}")

    # All k attempts must succeed
    return 1.0 if all(results[:k]) else 0.0


def aggregate_pass_power_k(task_results: List[List[bool]], k: int = None) -> float:
    """
    Aggregate pass^k across multiple tasks.

    Args:
        task_results: List of result lists, one per task
        k: Number of runs required (default: min length across tasks)

    Returns:
        Fraction of tasks achieving pass^k
    """
    if not task_results:
        return 0.0

    if k is None:
        k = min(len(r) for r in task_results)

    passing = sum(1 for results in task_results if pass_power_k(results, k) == 1.0)
    return passing / len(task_results)


def reliability_curve(task_results: List[List[bool]], max_k: int = None) -> List[float]:
    """
    Generate pass^k curve for k=1 to max_k.

    Useful for visualizing reliability degradation.

    Args:
        task_results: List of result lists, one per task
        max_k: Maximum k to compute (default: min length across tasks)

    Returns:
        List of pass^k values for k=1, 2, ..., max_k
    """
    if not task_results:
        return []

    if max_k is None:
        max_k = min(len(r) for r in task_results)

    return [aggregate_pass_power_k(task_results, k) for k in range(1, max_k + 1)]
