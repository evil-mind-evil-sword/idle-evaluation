"""
Task completion metrics.

Standard success rate metrics with optional graded scoring.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    condition: str
    run_id: int
    success: bool
    partial_score: Optional[float] = None  # 0.0-1.0 for graded
    steps: int = 0
    errors: List[str] = None
    tokens: int = 0
    duration_seconds: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def completion_rate(results: List[TaskResult]) -> float:
    """
    Calculate binary task completion rate.

    Args:
        results: List of task results

    Returns:
        Fraction of successful tasks (0.0 to 1.0)
    """
    if not results:
        return 0.0

    successes = sum(1 for r in results if r.success)
    return successes / len(results)


def graded_completion(results: List[TaskResult]) -> float:
    """
    Calculate graded completion score using partial_score.

    Falls back to binary if partial_score not available.

    Args:
        results: List of task results with optional partial_score

    Returns:
        Average graded score (0.0 to 1.0)
    """
    if not results:
        return 0.0

    total = 0.0
    for r in results:
        if r.partial_score is not None:
            total += r.partial_score
        else:
            total += 1.0 if r.success else 0.0

    return total / len(results)


def completion_by_condition(results: List[TaskResult]) -> dict:
    """
    Group completion rates by condition.

    Args:
        results: List of task results

    Returns:
        Dict mapping condition names to completion rates
    """
    by_condition = {}
    for r in results:
        if r.condition not in by_condition:
            by_condition[r.condition] = []
        by_condition[r.condition].append(r)

    return {cond: completion_rate(runs) for cond, runs in by_condition.items()}


def completion_by_task(results: List[TaskResult]) -> dict:
    """
    Group completion rates by task.

    Useful for identifying which tasks benefit most from treatment.

    Args:
        results: List of task results

    Returns:
        Dict mapping task_ids to completion rates
    """
    by_task = {}
    for r in results:
        if r.task_id not in by_task:
            by_task[r.task_id] = []
        by_task[r.task_id].append(r)

    return {task: completion_rate(runs) for task, runs in by_task.items()}


def improvement_matrix(results: List[TaskResult], baseline_condition: str = "baseline") -> dict:
    """
    Calculate improvement of each condition over baseline, per task.

    Args:
        results: List of task results
        baseline_condition: Name of baseline condition

    Returns:
        Dict of {condition: {task_id: improvement_delta}}
    """
    by_task = {}
    for r in results:
        if r.task_id not in by_task:
            by_task[r.task_id] = {}
        if r.condition not in by_task[r.task_id]:
            by_task[r.task_id][r.condition] = []
        by_task[r.task_id][r.condition].append(r)

    improvements = {}
    for task_id, conditions in by_task.items():
        if baseline_condition not in conditions:
            continue

        baseline_rate = completion_rate(conditions[baseline_condition])

        for cond, runs in conditions.items():
            if cond == baseline_condition:
                continue

            if cond not in improvements:
                improvements[cond] = {}

            treatment_rate = completion_rate(runs)
            improvements[cond][task_id] = treatment_rate - baseline_rate

    return improvements
