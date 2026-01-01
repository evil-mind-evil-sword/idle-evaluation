# Evaluating idle: Mandatory Peer Review for Agentic Coding

## Abstract

We present a rigorous evaluation of the **idle** plugin for Claude Code, which enforces mandatory peer review through an independent reviewer agent (alice) before task completion. Our experiments measure improvements in task completion rate, reliability (pass^k), and error accumulation across long-horizon coding tasks, error-critical multi-step workflows, and real-world software engineering issues. We find that [RESULTS SUMMARY].

## 1. Introduction

Large language model agents have demonstrated remarkable capabilities in software engineering tasks, yet they exhibit persistent error rates that compound over extended workflows [@meyerson2025maker]. When asked to verify their own outputs, models tend to confirm rather than critique [@huang2023selfverification], limiting the effectiveness of self-review mechanisms.

The **idle** plugin addresses this fundamental limitation through mandatory peer review: every agent exit attempt is blocked until an independent reviewer agent explicitly approves the work. This design is motivated by research showing that multi-agent debate produces more accurate outputs [@du2023debate; @liang2023encouraging].

Our evaluation tests three core hypotheses:

1. **H1**: Mandatory alice review reduces task failure rates
2. **H2**: Multi-agent review prevents error accumulation in long-horizon tasks
3. **H3**: The overhead of review is offset by reduced rework

## 2. Related Work

### 2.1 Multi-Agent Systems
Du et al. demonstrate that multiple LLM agents engaging in debate can improve factual accuracy and reasoning [@du2023debate]. idle's alice reviewer implements a similar principle in a practitioner-oriented tool.

### 2.2 Error Accumulation in Long-Horizon Tasks
The MAKER framework [@meyerson2025maker] shows that even with 99% per-step accuracy, agents fail after ~100 steps. Their solution—extreme decomposition with multi-agent voting—validates idle's architectural approach.

### 2.3 Agentic Coding Benchmarks
- **SWE-bench** [@jimenez2023swebench]: Real GitHub issue resolution
- **τ-Bench** [@sierra2024taubench]: Long-horizon conversational workflows with pass^k reliability metric
- **Terminal-Bench**: Multi-step CLI workflows with error recovery

## 3. Methodology

### 3.1 Evaluation Conditions

| Condition | Description |
|-----------|-------------|
| **baseline** | Vanilla Claude Code (Sonnet) |
| **idle-full** | Claude Code + idle plugin with Opus alice |
| **idle-no-alice** | idle with review disabled (ablation) |
| **idle-sonnet-alice** | idle with Sonnet reviewer (quality ablation) |

### 3.2 Experiments

#### Experiment 1: Long-Horizon Reliability
Multi-step coding tasks (20-50 agent steps) measuring:
- Task completion rate
- pass^k reliability (all k attempts must succeed)
- Steps to completion

#### Experiment 2: Error Accumulation
MAKER-inspired tasks with dependent steps:
- Towers of Hanoi (100-1000 moves)
- Chain-of-file-edits
Measuring per-step error rate and first-error step.

#### Experiment 3: SWE-bench Subset
Curated real GitHub issues testing code quality improvements.

### 3.3 Metrics

- **pass^k**: Fraction of tasks passing on all k attempts [@sierra2024taubench]
- **Per-step error rate**: ε = errors / total_steps [@meyerson2025maker]
- **Task completion rate**: Binary success per task
- **Cost per success**: Tokens consumed per successful completion

### 3.4 Statistical Analysis
Each condition × task combination runs n=5 times. We report means with 95% confidence intervals and perform paired t-tests for significance.

## 4. Results

### 4.1 Long-Horizon Task Completion

[FIGURE: completion_rates.png]

| Condition | Completion Rate | pass^5 |
|-----------|----------------|--------|
| baseline | X% | X% |
| idle-full | X% | X% |
| idle-no-alice | X% | X% |

[ANALYSIS]

### 4.2 Error Accumulation

[FIGURE: error_rate_box.png]

| Condition | Per-Step Error Rate | First Error Step |
|-----------|-------------------|------------------|
| baseline | X% | ~X |
| idle-full | X% | ~X |

[ANALYSIS]

### 4.3 SWE-bench Performance

[TABLE]

### 4.4 Cost-Efficiency Tradeoff

[FIGURE: cost_efficiency.png]

idle-full requires X% more tokens on average but achieves X% higher success rate, resulting in X% lower cost-per-success.

## 5. Discussion

### 5.1 Value of Mandatory Review
[FINDINGS on H1]

### 5.2 Reviewer Quality
Comparing idle-full (Opus alice) to idle-sonnet-alice reveals [FINDINGS].

### 5.3 Limitations
- Benchmarks may not fully represent real-world complexity
- Limited to Python/Claude ecosystems
- Alice reviewer adds latency

## 6. Conclusion

Our evaluation demonstrates that mandatory peer review via idle's alice mechanism [SUMMARY OF FINDINGS]. For practitioners, idle provides a simple, non-invasive mechanism to improve agentic coding reliability with measurable benefits.

## Acknowledgments

This evaluation framework draws inspiration from τ-Bench, MAKER, and SWE-bench.

## References

[GENERATED FROM references.bib]
