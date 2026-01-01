# Preliminary Results: idle Evaluation

## Experiment Setup

- **Tasks**: fizzbuzz, calculator (simple coding tasks)
- **Conditions**: baseline (IDLE_DISABLED=1), idle-full (normal idle)
- **Runs**: 2 per task per condition
- **Date**: 2024-12-31

## Results Summary

### Completion Rates

| Condition | fizzbuzz | calculator | Overall |
|-----------|----------|------------|---------|
| baseline | 2/2 (100%) | 2/2 (100%) | 4/4 (100%) |
| idle-full | 2/2 (100%) | 2/2 (100%) | 4/4 (100%) |

### Duration (seconds)

| Condition | fizzbuzz avg | calculator avg | Overall avg |
|-----------|--------------|----------------|-------------|
| baseline | 51.4s | 55.4s | 53.4s |
| idle-full | 54.7s | 59.0s | 56.9s |

**Overhead**: idle-full adds ~6.5% latency on these simple tasks.

## Analysis

### Key Finding: Simple Tasks Don't Differentiate

Both conditions achieve 100% success. This is expected because:

1. **Tasks are trivial**: fizzbuzz and calculator are well-within Claude's capabilities
2. **No error recovery needed**: The agent doesn't make mistakes requiring review
3. **Single-step verification**: Success is binary (file exists), not quality-graded

### idle's Value Proposition

idle's mandatory review (alice) provides value in scenarios where:

1. **The agent might exit prematurely** before completing all requirements
2. **Subtle errors occur** that self-review misses
3. **Long-horizon tasks** where errors accumulate (MAKER insight)
4. **Quality matters** beyond binary success

### Required: Harder Tasks

To properly evaluate idle, we need:

1. **Multi-step refactoring tasks** (20+ steps)
2. **Ambiguous requirements** where "done" is subjective
3. **Error-prone tasks** with higher baseline failure rates
4. **Quality-graded evaluation** (not just "does file exist")

## Figures

![Completion Rates](../figures/output/completion_rates.png)
![pass^k Curve](../figures/output/pass_k_curve.png)

## Conclusion

Preliminary results show idle adds ~6.5% overhead on trivial tasks with no measurable benefit. This is consistent with idle's design: review gates provide value on complex tasks, not simple ones.

**Next steps**: Design harder evaluation tasks that stress-test idle's review mechanism.
