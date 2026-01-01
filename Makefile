# idle-evaluation Makefile
# Run MAKER and other evaluations comparing baseline vs idle conditions

.PHONY: help setup test-quick test-baseline test-idle eval-maker eval-baseline eval-idle eval-chain-baseline eval-chain-idle results figures clean

PYTHON := .venv/bin/python
RUNNER := experiments/error_correction/runner.py

# Default sizes for evaluation
SIZES := 10 12 15
RUNS := 5

help:
	@echo "idle-evaluation - Evaluate idle on coding benchmarks"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Create venv and install dependencies"
	@echo ""
	@echo "Quick Tests (single run, small size):"
	@echo "  make test-quick     Run both baseline and idle-full (5 disks, 1 run)"
	@echo "  make test-baseline  Run baseline only (5 disks, 1 run)"
	@echo "  make test-idle      Run idle-full only (5 disks, 1 run)"
	@echo ""
	@echo "Full Evaluation (Phase 1 from benchmark_research.md):"
	@echo "  make eval-maker     Run baseline + idle-full (10,12,15 disks × 5 runs)"
	@echo "  make eval-baseline  Run baseline only (10,12,15 disks × 5 runs)"
	@echo "  make eval-idle      Run idle-full only (10,12,15 disks × 5 runs)"
	@echo ""
	@echo "Chain Edit Experiments:"
	@echo "  make eval-chain-baseline  Run chain edit baseline (5,10,15 steps × 5 runs)"
	@echo "  make eval-chain-idle      Run chain edit idle-full (5,10,15 steps × 5 runs)"
	@echo ""
	@echo "Results:"
	@echo "  make results        Show summary of results"
	@echo "  make figures        Generate comparison plots"
	@echo "  make clean          Remove results and workspaces"

setup:
	@./scripts/setup.sh

# Quick tests (5 disks, 1 run)
test-quick: test-baseline test-idle

test-baseline:
	$(PYTHON) $(RUNNER) --task hanoi --condition baseline --size 5 --runs 1

test-idle:
	$(PYTHON) $(RUNNER) --task hanoi --condition idle-full --size 5 --runs 1

# Full MAKER evaluation (Phase 1)
eval-maker: eval-baseline eval-idle
	@echo ""
	@echo "=== Evaluation Complete ==="
	@echo "Results in: results/"
	@make results

eval-baseline:
	@echo "=== Running Baseline (no hooks) ==="
	@for size in $(SIZES); do \
		echo "  Size: $$size disks"; \
		$(PYTHON) $(RUNNER) --task hanoi --condition baseline --size $$size --runs $(RUNS); \
	done

eval-idle:
	@echo "=== Running idle-full (with alice review) ==="
	@for size in $(SIZES); do \
		echo "  Size: $$size disks"; \
		$(PYTHON) $(RUNNER) --task hanoi --condition idle-full --size $$size --runs $(RUNS); \
	done

# Chain edit experiments
eval-chain-baseline:
	@for size in 5 10 15; do \
		$(PYTHON) $(RUNNER) --task chain --condition baseline --size $$size --runs $(RUNS); \
	done

eval-chain-idle:
	@for size in 5 10 15; do \
		$(PYTHON) $(RUNNER) --task chain --condition idle-full --size $$size --runs $(RUNS); \
	done

# Results summary
results:
	@$(PYTHON) scripts/summarize_results.py

figures:
	$(PYTHON) figures/generate.py

clean:
	rm -rf results/*.jsonl
	rm -rf workspaces/error-correction/*
	@echo "Cleaned results and workspaces"
