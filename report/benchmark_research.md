# Benchmark Research: Evaluating idle

## Executive Summary

Our initial evaluation used trivial tasks (fizzbuzz, calculator) that achieved 100% success in all conditions, failing to differentiate idle's value. This document surveys established, rigorous benchmarks that would properly stress-test idle's mandatory peer review mechanism.

## Recommended Benchmarks

### 1. τ-bench (Top Recommendation)

**Source**: [Sierra Research](https://github.com/sierra-research/tau-bench) | [Paper](https://arxiv.org/abs/2406.12045)

**Why it's ideal for idle evaluation**:
- Tests **long-horizon, multi-turn conversations** (exactly what idle targets)
- Uses **pass^k metric** for reliability (we already implemented this)
- State-of-the-art models achieve <50% success, <25% pass^8
- Real-world domains (airline, retail) with API tools and policies

**Installation**:
```bash
pip install git+https://github.com/sierra-research/tau-bench
```

**Running**:
```bash
python run.py --agent-strategy tool-calling --env retail \
  --model claude-3-5-sonnet-20240620 --model-provider anthropic \
  --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm --max-concurrency 10
```

**Key metrics**:
- Task success rate
- pass^k (reliability over k trials)
- Database state correctness

**Integration difficulty**: Medium - need to wrap Claude Code invocations

---

### 2. SWE-bench Lite / Verified

**Source**: [Princeton NLP](https://github.com/SWE-bench/SWE-bench) | [HuggingFace](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)

**Why it's valuable**:
- **Industry standard** for agentic coding evaluation
- Real GitHub issues with ground-truth patches
- SWE-bench Lite: 300 curated tasks
- SWE-bench Verified: 500 human-confirmed solvable tasks
- SWE-bench Pro: Much harder (~23% vs ~70% on Verified)

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

**Integration difficulty**: Medium-High - requires Docker, substantial disk space (120GB)

---

### 3. MAKER-style Error Accumulation Tasks

**Source**: [Cognizant AI Lab / UT Austin](https://arxiv.org/abs/2511.09030)

**Why it's relevant**:
- Tests **error accumulation** over many steps
- Validates idle's multi-agent review approach
- Simple to implement (Towers of Hanoi)
- Scales from 10 disks (1,023 steps) to 20 disks (1,048,575 steps)

**Key insight from paper**:
> "A system with a 1% per-step error rate is expected to fail after only 100 steps."

**Our implementation**: Already in `experiments/error_correction/runner.py`

**What we need**:
- More runs at various disk counts (10, 12, 15)
- Compare per-step error rates across conditions
- Measure first-error step distribution

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
- Claude Sonnet 4.5 achieves ~50-61%

**Integration difficulty**: Medium - requires Harbor framework

---

## Benchmark Comparison Matrix

| Benchmark | Long-horizon | Error Recovery | Open Source | Setup Difficulty | idle Relevance |
|-----------|--------------|----------------|-------------|------------------|----------------|
| **τ-bench** | ✅ Excellent | ✅ pass^k metric | ✅ Yes | Medium | ⭐⭐⭐⭐⭐ |
| **SWE-bench** | ⚠️ Moderate | ⚠️ Binary only | ✅ Yes | High | ⭐⭐⭐⭐ |
| **MAKER (Hanoi)** | ✅ Excellent | ✅ Per-step | ✅ Ours | Low | ⭐⭐⭐⭐ |
| **AgentBench** | ✅ Good | ⚠️ Limited | ✅ Yes | High | ⭐⭐⭐ |
| **Terminal-Bench** | ✅ Good | ✅ Yes | ⚠️ Partial | Medium | ⭐⭐⭐ |

## Recommended Evaluation Strategy

### Phase 1: τ-bench Integration (Highest Priority)
1. Install τ-bench
2. Create Claude Code wrapper for agent interface
3. Run retail domain with baseline vs idle-full
4. Measure pass^k for k=1,3,5,8

### Phase 2: MAKER-style Scaling (Medium Priority)
1. Run Towers of Hanoi at 10, 12, 15 disks
2. Measure per-step error rates
3. Plot first-error-step distributions

### Phase 3: SWE-bench Lite Subset (Lower Priority)
1. Select 20-30 issues from SWE-bench Lite
2. Run with baseline vs idle-full
3. Compare patch acceptance rates

## Expected Outcomes

Based on idle's design (mandatory alice review), we hypothesize:

1. **τ-bench**: idle-full should show higher pass^k than baseline, especially at higher k values (reliability improvement)

2. **Hanoi**: idle-full should show lower per-step error rate and later first-error-step (error prevention)

3. **SWE-bench**: idle-full should show higher patch quality (review catches bugs)

## References

- [τ-bench Paper](https://arxiv.org/abs/2406.12045) - Yao et al., 2024
- [SWE-bench Paper](https://arxiv.org/abs/2310.06770) - Jimenez et al., ICLR 2024
- [MAKER Paper](https://arxiv.org/abs/2511.09030) - Meyerson et al., 2025
- [AgentBench Paper](https://arxiv.org/abs/2308.03688) - Liu et al., ICLR 2024
- [8 Benchmarks Shaping AI Agents](https://ainativedev.io/news/8-benchmarks-shaping-the-next-generation-of-ai-agents)
- [Symflower Benchmark Survey](https://symflower.com/en/company/blog/2025/benchmarks-llm-agents/)
