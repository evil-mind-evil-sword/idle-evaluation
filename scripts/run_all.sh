#!/usr/bin/env bash
# Run all idle-evaluation experiments
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUNS="${RUNS:-5}"

echo "=========================================="
echo "idle-evaluation: Full Experiment Suite"
echo "=========================================="
echo "Runs per condition: $RUNS"
echo "Results directory: $REPO_ROOT/results"
echo ""

CONDITIONS=("baseline" "idle-full" "idle-no-alice" "idle-sonnet-alice")

# Long-horizon experiments
echo "=== Long-Horizon Experiments ==="
for cond in "${CONDITIONS[@]}"; do
    echo "Running $cond..."
    python "$REPO_ROOT/experiments/long-horizon/runner.py" \
        --condition "$cond" \
        --runs "$RUNS" \
        --output-dir "$REPO_ROOT/results" || true
done

# Error accumulation experiments
echo ""
echo "=== Error Accumulation Experiments ==="
for cond in "${CONDITIONS[@]}"; do
    echo "Running Hanoi (10 disks) for $cond..."
    python "$REPO_ROOT/experiments/error-correction/runner.py" \
        --task hanoi \
        --size 10 \
        --condition "$cond" \
        --runs "$RUNS" \
        --output-dir "$REPO_ROOT/results" || true
done

# SWE-bench experiments (if issues configured)
if [[ -f "$REPO_ROOT/experiments/swe-bench/selected_issues.json" ]]; then
    echo ""
    echo "=== SWE-bench Experiments ==="
    for cond in "${CONDITIONS[@]}"; do
        echo "Running SWE-bench for $cond..."
        python "$REPO_ROOT/experiments/swe-bench/runner.py" \
            --condition "$cond" \
            --runs "$RUNS" \
            --output-dir "$REPO_ROOT/results" || true
    done
else
    echo ""
    echo "Skipping SWE-bench (no issues configured)"
fi

# Generate figures
echo ""
echo "=== Generating Figures ==="
python "$REPO_ROOT/figures/generate.py" \
    --results-dir "$REPO_ROOT/results" \
    --output-dir "$REPO_ROOT/figures/output" || true

# Run analysis
echo ""
echo "=== Analysis ==="
python "$REPO_ROOT/experiments/long-horizon/analysis.py" \
    --results-dir "$REPO_ROOT/results" \
    --output "$REPO_ROOT/results/analysis.json" || true

echo ""
echo "=========================================="
echo "Experiments complete!"
echo "Results: $REPO_ROOT/results/"
echo "Figures: $REPO_ROOT/figures/output/"
echo "=========================================="
