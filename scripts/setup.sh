#!/usr/bin/env bash
# Setup script for idle-evaluation using uv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up idle-evaluation with uv..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$REPO_ROOT"

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Create necessary directories
mkdir -p "$REPO_ROOT/results"
mkdir -p "$REPO_ROOT/workspaces"
mkdir -p "$REPO_ROOT/figures/output"

# Verify Claude Code is available
if ! command -v claude &> /dev/null; then
    echo "Warning: 'claude' command not found. Install Claude Code to run experiments."
else
    echo "Claude Code found: $(which claude)"
fi

echo ""
echo "Setup complete!"
echo "Run experiments with: uv run scripts/run_all.sh"
echo "Or activate: source .venv/bin/activate"
