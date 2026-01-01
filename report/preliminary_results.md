# Preliminary Results: idle Evaluation

## Experiment Setup

- **Tasks**: fizzbuzz, calculator (simple coding tasks)
- **Conditions**: baseline, idle-full, idle-no-alice (ablation)
- **Runs**: 2-5 per task per condition (17 total runs)
- **Date**: 2024-12-31

## Results Summary

### Completion Rates

| Condition | fizzbuzz | calculator | Overall |
|-----------|----------|------------|---------|
| baseline | 2/2 (100%) | 5/5 (100%) | 7/7 (100%) |
| idle-full | 2/2 (100%) | 5/5 (100%) | 7/7 (100%) |
| idle-no-alice | - | 3/3 (100%) | 3/3 (100%) |

### Duration (seconds)

| Condition | calculator avg | Notes |
|-----------|----------------|-------|
| baseline | 59.9s | Fastest |
| idle-no-alice | 61.9s | +3.3% overhead |
| idle-full | 64.7s | +8.0% overhead |

**Overhead**: idle-full adds ~8% latency, idle-no-alice adds ~3% (review cost is ~5%).

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
