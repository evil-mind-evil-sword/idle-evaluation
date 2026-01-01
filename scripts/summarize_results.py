#!/usr/bin/env python3
"""Summarize MAKER evaluation results from JSONL files."""

import json
import sys
from pathlib import Path


def summarize_file(path: Path) -> None:
    """Print summary statistics for a results file."""
    print(f"\n{path}:")

    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("  (empty file)")
        return

    total = len(results)
    solved = sum(1 for r in results if r.get("solved"))
    avg_err = sum(r.get("error_rate", 0) for r in results) / total

    print(f"  Solved: {solved}/{total} ({100*solved/total:.0f}%)")
    print(f"  Avg error rate: {avg_err:.4f}")


def main():
    results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print("No results yet. Run: make eval-maker")
        return

    print("\n=== Results Summary ===")

    files = sorted(results_dir.glob("error-correction_*.jsonl"))
    if not files:
        print("No results yet. Run: make eval-maker")
        return

    for f in files:
        summarize_file(f)


if __name__ == "__main__":
    main()
