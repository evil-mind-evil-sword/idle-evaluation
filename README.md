# idle-evaluation

**Evaluation framework for idle.** Measure quality gates on long-horizon tasks.

Reproducible experiments comparing Claude Code with and without the [idle](https://github.com/evil-mind-evil-sword/idle) plugin on tasks requiring many steps and low failure tolerance.

## Overview

This repository contains reproducible experiments comparing:

| Condition | Description |
|-----------|-------------|
| **baseline** | Vanilla Claude Code (Sonnet) |
| **idle-full** | Claude Code + idle plugin (alice reviewer) |
| **idle-no-alice** | idle with review disabled (ablation) |
| **idle-sonnet-alice** | idle with Sonnet reviewer (quality ablation) |

## Experiments

### 1. Long-Horizon Reliability (τ-Bench inspired)

Multi-step conversational coding tasks requiring 20-50 agent steps.

```bash
python experiments/long_horizon/runner.py --condition baseline --runs 5
python experiments/long_horizon/runner.py --condition idle-full --runs 5
```

### 2. Error Accumulation (MAKER inspired)

Tasks with many dependent steps where errors compound:
- Towers of Hanoi (100-1000 steps)
- Chain-of-file-edits

```bash
python experiments/error_correction/runner.py --task hanoi --disks 10
```

### 3. SWE-bench Subset

Curated subset of 20-30 real GitHub issues.

```bash
python experiments/swe_bench/runner.py --condition idle-full --runs 5
```

### 4. τ-bench (Recommended)

Long-horizon conversational workflows with pass^k reliability metrics.
Best benchmark for testing idle's value on complex, multi-turn tasks.

```bash
# Install τ-bench
uv pip install git+https://github.com/sierra-research/tau-bench

# Run comparison (baseline vs idle-full)
python experiments/tau_bench/runner.py --env retail --trials 3

# Run specific condition
python experiments/tau_bench/runner.py --env retail --condition idle-full --trials 5
```

See [benchmark_research.md](report/benchmark_research.md) for detailed analysis of benchmark options.

## Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **pass^k** | Reliability across k repeated runs | τ-Bench |
| **per-step error rate** | ε = errors / total_steps | MAKER |
| **task completion rate** | Binary success per task | Standard |
| **cost per success** | API tokens per completed task | Efficiency |

## Quick Start

```bash
# Setup
./scripts/setup.sh

# Run full evaluation suite
./scripts/run_all.sh

# Generate figures
python figures/generate.py

# View results
cat results/summary.json
```

## Repository Structure

```
idle-evaluation/
├── experiments/
│   ├── long_horizon/       # τ-Bench style experiments
│   ├── error_correction/   # MAKER-inspired error tests
│   └── swe_bench/          # GitHub issue resolution
├── harnesses/              # Condition configurations
├── metrics/                # Metric implementations
├── figures/                # Figure generation
├── report/                 # Technical report
├── results/                # Experiment outputs
└── scripts/                # Automation
```

## Citations

This evaluation framework draws from:

- **τ-Bench**: Sierra AI's long-horizon conversational benchmark
- **MAKER**: Cognizant/UT Austin's million-step zero-error framework ([arXiv:2511.09030](https://arxiv.org/abs/2511.09030))
- **SWE-bench**: Princeton's real-world software engineering benchmark ([ICLR 2024](https://www.swebench.com/))
- **Terminal-Bench**: Stanford/Laude Institute CLI workflow benchmark

## License

MIT

## Related

- [idle](https://github.com/evil-mind-evil-sword/idle) - Quality gate plugin for Claude Code
- [tissue](https://github.com/evil-mind-evil-sword/tissue) - Issue tracking for agents
- [zawinski](https://github.com/evil-mind-evil-sword/zawinski) - Messaging for agents
