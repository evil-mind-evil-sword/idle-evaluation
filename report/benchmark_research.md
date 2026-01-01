# Benchmark Research: Evaluating idle

## Executive Summary

Our initial evaluation used trivial tasks (fizzbuzz, calculator) that achieved 100% success in all conditions, failing to differentiate idle's value. This document surveys established, rigorous benchmarks that would properly stress-test idle's mandatory peer review mechanism.

**Key Finding**: SWE-bench is the recommended primary benchmark because:
1. Its "fix this bug" workflow naturally maps to idle's review-and-revision loop
2. Our `swe_bench/runner.py` already correctly routes through Claude Code CLI
3. Clear pass/fail signal from actual test suites

---

## Recommended Benchmarks

### 1. SWE-bench Lite / Verified (Primary Recommendation)

**Source**: [Princeton NLP](https://github.com/SWE-bench/SWE-bench) | [HuggingFace](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)

**Why it's ideal for idle**:
- **"Fix this bug" workflow** naturally creates a review scenario
- alice can assess: Does the fix address the issue? Introduce new bugs?
- Clear pass/fail signal from actual test suites
- **Our runner already works**: `swe_bench/runner.py` invokes `claude` CLI (line 155)

**Variants**:
- SWE-bench Lite: 300 curated tasks
- SWE-bench Verified: 500 human-confirmed solvable tasks (~70% sota)
- SWE-bench Pro (2025): Much harder (~23% sota)

**Installation**:
```bash
git clone https://github.com/princeton-nlp/SWE-bench.git
pip install -e .
```

**Dataset access**:
```python
from datasets import load_dataset
swe_lite = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
```

**Key metrics**:
- Patch correctness (tests pass)
- Resolution rate

**Cost/Resource Estimates**:
- Disk space: ~120GB for full Docker environments
- Per-task cost: ~$0.50-2.00 in API tokens (depending on issue complexity)
- Time per task: 5-15 minutes
- Full Lite (300 × 5 runs): ~$750-3000, ~125 hours compute

**Integration status**: 🔶 Requires modification

> ⚠️ **Current runner limitation**: `swe_bench/runner.py` uses `--print --dangerously-skip-permissions` mode (line 155), which runs Claude non-interactively and prevents Stop hooks from triggering. The `idle_config` in harness configs is also not loaded.
>
> **To properly test idle's peer review**, the runner needs modification to:
> 1. Remove `--print` flag to enable interactive mode
> 2. Load and apply `idle_config` from harness config
> 3. Or invoke Claude Code through idle's loop mechanism

---

### 2. CyberSecEval (Security Review - Future)

**Source**: [Meta PurpleLlama](https://github.com/meta-llama/PurpleLlama) | [Paper](https://arxiv.org/abs/2404.13161)

**Why it's valuable for idle**:
- Tests alice as a **specialized security auditor**
- Objective metric: What % of vulnerabilities does alice catch?
- Tests exploit generation, prompt injection, AutoPatch capabilities
- All models show 26-41% prompt injection success - room for improvement

**Workflow for idle**:
1. Primary agent generates code from CyberSecEval prompt
2. alice runs security scan on the output
3. If vulnerabilities found, alice creates tissue issues
4. Primary agent remediates, loop continues

**Key metric**: Vulnerability Detection Rate = (alice-caught / actual-vulnerabilities)

**Cost/Resource Estimates**:
- Setup: Moderate (Docker, static analysis tools)
- Per-task: ~$0.20-0.50 (shorter tasks than SWE-bench)
- Good for targeted alice evaluation

**Integration status**: 🔶 Planned - requires custom harness for security scanning

---

### 3. MAKER-style Error Accumulation Tasks (Recommended Starting Point)

**Source**: [Cognizant AI Lab / UT Austin](https://arxiv.org/abs/2511.09030)

**Why this should be Phase 1**:
- **Lowest barrier**: No Docker, no external datasets, runs locally
- **Directly tests idle's hypothesis**: Error accumulation is exactly what alice review should prevent
- **Granular metrics**: Per-step error rate, first-error-step (not just binary pass/fail)
- **Scalable difficulty**: 10 disks (1,023 steps) → 15 disks (32,767 steps)

**Key insight from paper**:
> "A system with a 1% per-step error rate is expected to fail after only 100 steps."

**Two task types implemented** (`experiments/error_correction/runner.py`):

| Task | Description | Metric |
|------|-------------|--------|
| **Towers of Hanoi** | Multi-step puzzle, each move depends on state | Per-step error rate, optimality |
| **Chain Edit** | Sequential file edits, each depends on previous | Error propagation, file correctness |

**Cost/Resource Estimates**:
- Setup: None (pure Python)
- Per-run (10 disks): ~$0.10-0.30
- Full experiment (3 sizes × 5 runs × 2 conditions): ~$3-10, <1 hour

**Integration status**: 🔶 Runner needs fix

> ⚠️ Same `--print` issue as SWE-bench (line 107). But simpler to fix - no external dependencies.

**What we need**:
- Fix runner to enable idle hooks
- Run at various sizes (10, 12, 15 disks)
- Compare per-step error rates: baseline vs idle-full
- Plot first-error-step distributions

---

### 4. AgentBench (Comprehensive)

**Source**: [THUDM](https://github.com/THUDM/AgentBench) | [ICLR 2024](https://arxiv.org/abs/2308.03688)

**Why it's useful**:
- **8 distinct environments** (OS, web, games, etc.)
- Tests reasoning and decision-making
- Docker-based isolation

**Key finding from paper**:
> "Poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents."

**Integration difficulty**: High - complex Docker setup

---

### 5. Terminal-Bench

**Source**: [Stanford / Laude Institute](https://www.tbench.ai/leaderboard)

**Why it matters**:
- Tests **CLI operations** in sandboxed environment
- Multi-step workflows (setup, debug, build, execute)
- Claude 3.5 Sonnet achieves ~50-61%

**Integration difficulty**: Medium - requires Harbor framework

---

## Benchmark Comparison Matrix

| Benchmark | Conceptual Fit | Integration Status | Setup Difficulty | idle Relevance |
|-----------|----------------|-------------------|------------------|----------------|
| **MAKER (Hanoi/Chain)** | ✅ Error accumulation | 🔶 Runner needs fix | **Lowest** | ⭐⭐⭐⭐⭐ |
| **SWE-bench** | ✅ Bug-fix = review | 🔶 Runner needs fix | High (Docker) | ⭐⭐⭐⭐⭐ |
| **CyberSecEval** | ✅ Security audit | 🔶 Planned | Medium | ⭐⭐⭐⭐ |
| **τ-bench** | ⚠️ Conversational | ❌ Bypass + mismatch | Medium | ⭐⭐ |
| **AgentBench** | ⚠️ Multi-env | ❌ Not started | High | ⭐⭐ |

> **Note**: All current runners use `--print` mode which prevents idle Stop hooks from triggering. Runner modification is required before any benchmark can properly evaluate idle's peer review mechanism.

### ⚠️ τ-bench Integration Issue

**Current `tau_bench/runner.py` bypasses Claude Code entirely**. Line 90 calls:
```python
"--model-provider", "anthropic"  # Calls API directly, not claude CLI
```

This means idle hooks never trigger. Additionally, τ-bench tests conversational tool-calling workflows, not code review - a conceptual mismatch with idle's value proposition.

**Recommendation**: Deprioritize τ-bench. If pursued, requires rewriting runner to invoke `claude` CLI.

---

## Recommended Evaluation Strategy

### Phase 1: MAKER Tasks (Start Here)

**Why first**: Lowest barrier, directly tests idle's core hypothesis, cheapest to run.

**Prerequisites** (runner modification required):
1. Remove `--print` flag from `experiments/error_correction/runner.py` line 107
2. Configure idle hooks (Stop hook must trigger alice review)

**Execution plan**:
```bash
# After runner fix, run comparison
for SIZE in 10 12 15; do
  python experiments/error_correction/runner.py \
    --task hanoi --condition baseline --size $SIZE --runs 5
  python experiments/error_correction/runner.py \
    --task hanoi --condition idle-full --size $SIZE --runs 5
done
```

**Key metrics**:
- Per-step error rate (should be lower with idle-full)
- First-error-step (should be later with idle-full)
- Solve rate (should be higher with idle-full)

**Expected cost**: ~$3-10 total, <1 hour

---

### Phase 2: SWE-bench PoC

**Objective**: Validate on real-world bug fixes after MAKER shows signal.

**Prerequisites**:
1. Docker installed with 120GB disk
2. Runner modification (same `--print` fix)

```bash
# Setup SWE-bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench && pip install -e .

# After runner modification, run single task
python experiments/swe_bench/runner.py \
  --condition idle-full \
  --issues django__django-10899 \
  --runs 1
```

**Evaluation Criteria**:

| Scenario | Test Result | alice Review | Interpretation |
|----------|-------------|--------------|----------------|
| A (ideal) | PASS | Correct approve | System works |
| B | FAIL | Caught flaws | alice valuable |
| C | FAIL | Missed flaws | alice needs work |
| D | PASS | Wrong reasoning | alice not adding value |

---

### Phase 3: CyberSecEval (Future)
1. Build harness for security scanning
2. Measure alice's vulnerability detection rate
3. Compare catch rate vs static analysis baseline

---

## Expected Outcomes

Based on idle's design (mandatory alice review), we hypothesize:

1. **MAKER/Hanoi** (Phase 1): idle-full should show:
   - Lower per-step error rate (alice catches mistakes before they compound)
   - Later first-error-step (errors prevented earlier in the sequence)
   - Higher solve rate at larger disk counts

2. **SWE-bench** (Phase 2): idle-full should show:
   - Higher patch quality and pass rate
   - Fewer regression failures (alice catches unintended side effects)

3. **CyberSecEval** (Phase 3): alice should catch vulnerabilities that static analysis misses

## References

### Primary Benchmarks
- [SWE-bench](https://www.swebench.com/) - Jimenez et al., ICLR 2024 | [Paper](https://arxiv.org/abs/2310.06770)
- [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) - OpenAI, 2024
- [SWE-bench Pro](https://scale.com/leaderboard/swe_bench_pro_public) - Scale AI, 2025
- [CyberSecEval](https://github.com/meta-llama/PurpleLlama) - Meta, 2024 | [Paper](https://arxiv.org/abs/2404.13161)

### Additional Benchmarks (Research Phase)
- [RE-Bench](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/) - METR, ICML 2025
- [SWE-Lancer](https://openai.com/index/swe-lancer/) - OpenAI, 2025
- [MLE-bench](https://openai.com/index/mle-bench/) - OpenAI, 2024
- [BigCodeBench](https://bigcode-bench.github.io/) - ICLR 2025
- [LiveCodeBench](https://livecodebench.github.io/) - 2024

### Background
- [τ-bench Paper](https://arxiv.org/abs/2406.12045) - Yao et al., 2024 (deprioritized - conceptual mismatch)
- [MAKER Paper](https://arxiv.org/abs/2511.09030) - Meyerson et al., 2025
- [AgentBench Paper](https://arxiv.org/abs/2308.03688) - Liu et al., ICLR 2024
- [EvidentlyAI Benchmark Survey](https://www.evidentlyai.com/blog/ai-agent-benchmarks)
- [Symflower Benchmark Survey](https://symflower.com/en/company/blog/2025/benchmarks-llm-agents/)

---

*Last updated: 2025-12-31 - Revised to prioritize MAKER tasks as lowest-barrier starting point, then SWE-bench.*
