"""
Error rate metrics from MAKER framework.

Measures per-step error accumulation for long-horizon tasks.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StepResult:
    """Result of a single step in a multi-step task."""
    step_id: int
    expected: str
    actual: str
    correct: bool
    error_type: Optional[str] = None


def per_step_error_rate(steps: List[StepResult]) -> float:
    """
    Calculate per-step error rate: ε = errors / total_steps.

    From MAKER: A 1% per-step error rate means expected failure
    after ~100 steps of a million-step task.

    Args:
        steps: List of step results

    Returns:
        Error rate (0.0 to 1.0)
    """
    if not steps:
        return 0.0

    errors = sum(1 for s in steps if not s.correct)
    return errors / len(steps)


def first_error_step(steps: List[StepResult]) -> Optional[int]:
    """
    Find the step index where the first error occurred.

    Useful for understanding derailment patterns.

    Args:
        steps: List of step results

    Returns:
        Index of first error, or None if no errors
    """
    for i, step in enumerate(steps):
        if not step.correct:
            return i
    return None


def error_free_streak(steps: List[StepResult]) -> int:
    """
    Count consecutive correct steps from the start.

    Args:
        steps: List of step results

    Returns:
        Number of consecutive correct steps
    """
    streak = 0
    for step in steps:
        if step.correct:
            streak += 1
        else:
            break
    return streak


def error_distribution(steps: List[StepResult]) -> dict:
    """
    Analyze error type distribution.

    Args:
        steps: List of step results

    Returns:
        Dict mapping error types to counts
    """
    dist = {}
    for step in steps:
        if not step.correct:
            error_type = step.error_type or "unknown"
            dist[error_type] = dist.get(error_type, 0) + 1
    return dist


def expected_steps_to_failure(error_rate: float) -> float:
    """
    Calculate expected number of steps before first failure.

    For geometric distribution with per-step error rate ε,
    expected steps = 1/ε.

    Args:
        error_rate: Per-step error probability

    Returns:
        Expected steps to first failure
    """
    if error_rate <= 0:
        return float('inf')
    return 1.0 / error_rate


def required_error_rate(target_steps: int, target_success_prob: float = 0.5) -> float:
    """
    Calculate required per-step error rate to achieve target.

    For n steps with success probability p:
    (1 - ε)^n = p
    ε = 1 - p^(1/n)

    Args:
        target_steps: Number of steps to complete
        target_success_prob: Desired probability of completing all steps

    Returns:
        Required per-step error rate

    Example:
        >>> required_error_rate(1_000_000, 0.5)  # 50% success over 1M steps
        6.93e-07  # Need < 0.0001% error rate per step
    """
    if target_steps <= 0:
        return 0.0
    return 1.0 - (target_success_prob ** (1.0 / target_steps))


def compare_error_rates(
    baseline_steps: List[StepResult],
    treatment_steps: List[StepResult]
) -> Tuple[float, float, float]:
    """
    Compare error rates between conditions.

    Args:
        baseline_steps: Steps from baseline condition
        treatment_steps: Steps from treatment condition

    Returns:
        Tuple of (baseline_rate, treatment_rate, improvement_ratio)
    """
    baseline_rate = per_step_error_rate(baseline_steps)
    treatment_rate = per_step_error_rate(treatment_steps)

    if baseline_rate > 0:
        improvement = baseline_rate / treatment_rate if treatment_rate > 0 else float('inf')
    else:
        improvement = 1.0

    return baseline_rate, treatment_rate, improvement
