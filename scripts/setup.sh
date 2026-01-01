#!/usr/bin/env bash
# Setup script for idle-evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up idle-evaluation..."

# Create virtual environment if it doesn't exist
if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$REPO_ROOT/.venv"
fi

# Activate and install dependencies
source "$REPO_ROOT/.venv/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install matplotlib numpy seaborn

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

# Check for idle plugin
if [[ -d "$HOME/.claude/plugins/idle" ]]; then
    echo "idle plugin found"
else
    echo "Note: idle plugin not found at ~/.claude/plugins/idle"
    echo "Install idle for idle-full experiments"
fi

echo ""
echo "Setup complete!"
echo "Activate with: source $REPO_ROOT/.venv/bin/activate"
echo "Run experiments with: ./scripts/run_all.sh"
