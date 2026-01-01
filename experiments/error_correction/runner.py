#!/usr/bin/env python3
"""
Error accumulation experiment runner.

Inspired by MAKER: measures per-step error rates on tasks with many
dependent steps where errors compound.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pexpect

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics.error_rate import StepResult, per_step_error_rate, first_error_step


def run_claude_with_hooks(
    prompt: str,
    work_dir: Path,
    config: dict,
    session_id: str = None,
    timeout: int = 600
) -> str:
    """
    Run Claude Code with hooks enabled (no --print flag).

    Uses pexpect to spawn Claude in a pseudo-terminal so hooks trigger normally.
    The Stop hook will fire when Claude tries to exit, automatically triggering
    alice review for idle-full conditions.
    """
    env = os.environ.copy()
    env.update(config.get("environment", {}))

    # Apply idle_config if present (hooks, agents, skills directories)
    idle_config = config.get("idle_config", {})
    if idle_config.get("hooks_dir"):
        env["CLAUDE_HOOKS_DIR"] = idle_config["hooks_dir"]
    if idle_config.get("agents_dir"):
        env["CLAUDE_AGENTS_DIR"] = idle_config["agents_dir"]
    if idle_config.get("skills_dir"):
        env["CLAUDE_SKILLS_DIR"] = idle_config["skills_dir"]

    # Generate session ID if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Escape prompt for shell
    escaped_prompt = prompt.replace('"', '\\"').replace('$', '\\$')
    cmd = f'claude --session-id {session_id} --dangerously-skip-permissions "{escaped_prompt}"'

    child = pexpect.spawn(
        '/bin/bash', ['-c', cmd],
        cwd=str(work_dir),
        env=env,
        timeout=timeout,
        encoding='utf-8',
        dimensions=(50, 200)  # rows, cols
    )

    output_lines = []
    try:
        while True:
            try:
                line = child.readline()
                if not line:
                    break
                output_lines.append(line)
            except pexpect.TIMEOUT:
                break
            except pexpect.EOF:
                break
    finally:
        child.close()

    return "".join(output_lines)


@dataclass
class HanoiState:
    """State of Towers of Hanoi puzzle."""
    pegs: List[List[int]]  # 3 pegs, each with stack of disks

    @classmethod
    def initial(cls, n_disks: int) -> "HanoiState":
        """Create initial state with all disks on first peg."""
        return cls(pegs=[[i for i in range(n_disks, 0, -1)], [], []])

    @classmethod
    def goal(cls, n_disks: int) -> "HanoiState":
        """Create goal state with all disks on third peg."""
        return cls(pegs=[[], [], [i for i in range(n_disks, 0, -1)]])

    def is_valid_move(self, from_peg: int, to_peg: int) -> bool:
        """Check if move is valid."""
        if not self.pegs[from_peg]:
            return False
        if self.pegs[to_peg] and self.pegs[to_peg][-1] < self.pegs[from_peg][-1]:
            return False
        return True

    def apply_move(self, from_peg: int, to_peg: int) -> bool:
        """Apply move and return success."""
        if not self.is_valid_move(from_peg, to_peg):
            return False
        disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk)
        return True

    def copy(self) -> "HanoiState":
        return HanoiState(pegs=[list(p) for p in self.pegs])


def generate_hanoi_solution(n_disks: int) -> List[Tuple[int, int]]:
    """Generate optimal solution for Towers of Hanoi."""
    moves = []

    def hanoi(n, source, target, auxiliary):
        if n > 0:
            hanoi(n - 1, source, auxiliary, target)
            moves.append((source, target))
            hanoi(n - 1, auxiliary, target, source)

    hanoi(n_disks, 0, 2, 1)
    return moves


def run_hanoi_with_agent(
    n_disks: int,
    config: dict,
    work_dir: Path,
    condition: str = "baseline",
    max_steps: int = None
) -> Tuple[List[StepResult], bool]:
    """
    Run Towers of Hanoi with Claude Code agent.

    For baseline: uses --print mode (fast, no hooks)
    For idle-full: uses pexpect (hooks enabled, alice review triggers)

    Returns list of step results and whether puzzle was solved.
    """
    if max_steps is None:
        max_steps = 2 ** n_disks  # Optimal is 2^n - 1

    prompt = f"""Solve the Towers of Hanoi puzzle with {n_disks} disks.

Rules:
- Three pegs: A (left), B (middle), C (right)
- All {n_disks} disks start on peg A, largest on bottom
- Goal: Move all disks to peg C
- Only move one disk at a time
- Never place larger disk on smaller disk

IMPORTANT: You MUST output each move on its own line in EXACTLY this format:
MOVE A C

List ALL {2**n_disks - 1} moves. Do not summarize. Do not use code blocks.
Start outputting the moves now:
"""

    state = HanoiState.initial(n_disks)
    goal = HanoiState.goal(n_disks)
    expected_moves = generate_hanoi_solution(n_disks)

    try:
        # Generate fresh session ID to avoid context pollution
        session_id = str(uuid.uuid4())

        if condition == "baseline":
            # Use --print for baseline (fast, no hooks)
            env = dict(config.get("environment", {}))
            cmd = [
                "claude", "--print",
                "--session-id", session_id,
                "--dangerously-skip-permissions",
                prompt
            ]
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, **env}
            )
            output = result.stdout + result.stderr
        else:
            # Use pexpect for idle conditions (hooks enabled)
            output = run_claude_with_hooks(prompt, work_dir, config, session_id=session_id, timeout=600)

        # Debug: save raw output
        debug_file = work_dir / f"debug_output_{session_id[:8]}.txt"
        with open(debug_file, "w") as f:
            f.write(output)

    except Exception as e:
        print(f"  Error running agent: {e}")
        return [], False

    # Parse moves from output
    step_results = []
    actual_moves = []
    peg_map = {"A": 0, "B": 1, "C": 2, "0": 0, "1": 1, "2": 2}

    for line in output.split("\n"):
        if "MOVE" in line.upper():
            parts = line.upper().split()
            try:
                if "MOVE" not in parts:
                    continue
                move_idx = parts.index("MOVE")
                if move_idx + 2 >= len(parts):
                    continue
                from_peg = peg_map.get(parts[move_idx + 1], -1)
                to_peg = peg_map.get(parts[move_idx + 2], -1)

                if from_peg >= 0 and to_peg >= 0:
                    actual_moves.append((from_peg, to_peg))
            except (IndexError, KeyError, ValueError):
                continue

    # Validate moves
    test_state = HanoiState.initial(n_disks)

    for i, (from_peg, to_peg) in enumerate(actual_moves):
        if i >= max_steps:
            break

        # Determine expected move if within optimal path
        expected = expected_moves[i] if i < len(expected_moves) else None
        expected_str = f"{expected[0]}->{expected[1]}" if expected else "N/A"
        actual_str = f"{from_peg}->{to_peg}"

        # Check if move is valid for current state
        valid = test_state.is_valid_move(from_peg, to_peg)

        if valid:
            test_state.apply_move(from_peg, to_peg)
            # Check if it matches expected optimal move
            correct = expected is not None and (from_peg, to_peg) == expected
        else:
            correct = False

        step_results.append(StepResult(
            step_id=i,
            expected=expected_str,
            actual=actual_str,
            correct=correct and valid,
            error_type=None if (correct and valid) else ("invalid" if not valid else "suboptimal")
        ))

    # Check if puzzle was solved
    solved = test_state.pegs == goal.pegs

    return step_results, solved


def run_chain_edit_experiment(
    chain_length: int,
    config: dict,
    work_dir: Path,
    condition: str = "baseline"
) -> Tuple[List[StepResult], bool]:
    """
    Run chain-of-file-edits experiment.

    Creates a chain of files where each edit depends on the previous.
    Tests error propagation in dependent multi-step tasks.

    For baseline: uses --print mode (fast, no hooks)
    For idle-full: uses pexpect (hooks enabled, alice review triggers)
    """
    # Create initial file chain
    task_dir = work_dir / f"chain_{chain_length}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create chain definition
    chain = []
    for i in range(chain_length):
        chain.append({
            "file": f"step_{i}.txt",
            "input_value": i * 10,
            "operation": "add_5",
            "expected_output": i * 10 + 5
        })

    with open(task_dir / "chain.json", "w") as f:
        json.dump(chain, f, indent=2)

    # Write initial file
    with open(task_dir / "step_0.txt", "w") as f:
        f.write("0")

    prompt = f"""Complete this chain of file operations in {task_dir}:

For each step i from 0 to {chain_length - 1}:
1. Read step_{{i}}.txt
2. Add 5 to the number
3. Write result to step_{{i+1}}.txt

Start with step_0.txt which contains "0".
After all steps, step_{chain_length}.txt should contain "{5 * chain_length}".

Perform each step carefully, verifying each file before proceeding.
"""

    session_id = str(uuid.uuid4())

    try:
        if condition == "baseline":
            # Use --print for baseline (fast, no hooks)
            env = dict(config.get("environment", {}))
            cmd = [
                "claude", "--print",
                "--session-id", session_id,
                "--dangerously-skip-permissions",
                prompt
            ]
            subprocess.run(
                cmd,
                cwd=task_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, **env}
            )
        else:
            # Use pexpect for idle conditions (hooks enabled)
            run_claude_with_hooks(prompt, task_dir, config, session_id=session_id, timeout=300)
    except Exception as e:
        print(f"  Error running agent: {e}")

    # Verify chain
    step_results = []
    expected = 0

    for i in range(chain_length + 1):
        expected = i * 5
        file_path = task_dir / f"step_{i}.txt"

        try:
            with open(file_path) as f:
                actual = int(f.read().strip())
            correct = actual == expected
            error_type = None if correct else "wrong_value"
        except FileNotFoundError:
            actual = -1
            correct = False
            error_type = "missing_file"
        except ValueError:
            actual = -1
            correct = False
            error_type = "invalid_content"

        step_results.append(StepResult(
            step_id=i,
            expected=str(expected),
            actual=str(actual),
            correct=correct,
            error_type=error_type
        ))

    solved = all(s.correct for s in step_results)
    return step_results, solved


def run_experiment(
    task: str,
    condition: str,
    size: int,
    runs: int = 5,
    output_dir: Optional[Path] = None
):
    """Run error accumulation experiment."""
    from experiments.long_horizon.runner import load_harness_config

    config = load_harness_config(condition)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(__file__).parent.parent.parent / "workspaces" / "error-correction"
    work_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"error-correction_{task}_{condition}_{timestamp}.jsonl"

    print(f"Running {task} (size={size}) × {runs} runs for condition: {condition}")

    all_results = []

    for run_id in range(1, runs + 1):
        print(f"\n=== Run {run_id}/{runs} ===")

        if task == "hanoi":
            steps, solved = run_hanoi_with_agent(size, config, work_dir, condition)
        elif task == "chain":
            steps, solved = run_chain_edit_experiment(size, config, work_dir, condition)
        else:
            print(f"Unknown task: {task}")
            return

        error_rate = per_step_error_rate(steps)
        first_err = first_error_step(steps)

        result = {
            "task": task,
            "size": size,
            "condition": condition,
            "run_id": run_id,
            "solved": solved,
            "total_steps": len(steps),
            "error_rate": error_rate,
            "first_error_step": first_err,
            "errors": [asdict(s) for s in steps if not s.correct]
        }

        all_results.append(result)

        status = "SOLVED" if solved else "FAILED"
        print(f"  {status} | {len(steps)} steps | error_rate: {error_rate:.4f} | first_error: {first_err}")

        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    print(f"\n=== Summary for {condition} ===")
    solved_count = sum(1 for r in all_results if r["solved"])
    avg_error_rate = sum(r["error_rate"] for r in all_results) / len(all_results)

    print(f"Solved: {solved_count}/{runs}")
    print(f"Average error rate: {avg_error_rate:.4f}")
    print(f"\nResults written to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run error accumulation experiments")
    parser.add_argument("--task", required=True, choices=["hanoi", "chain"],
                       help="Task type")
    parser.add_argument("--condition", required=True,
                       choices=["baseline", "idle-full", "idle-no-alice", "idle-sonnet-alice"])
    parser.add_argument("--size", type=int, default=10,
                       help="Task size (disks for hanoi, chain length for chain)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()

    run_experiment(
        task=args.task,
        condition=args.condition,
        size=args.size,
        runs=args.runs,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
