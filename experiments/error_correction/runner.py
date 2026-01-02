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

    Uses pexpect to spawn Claude in interactive mode so hooks trigger.
    After Claude responds, sends /exit to trigger the Stop hook.
    The idle plugin activates when #idle:on is in the prompt.
    """
    env = os.environ.copy()
    env.update(config.get("environment", {}))

    # Generate session ID if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Add #idle:on to prompt if idle is enabled
    if config.get("idle_enabled"):
        prompt = f"#idle:on\n\n{prompt}"

    # Write prompt to a temp file
    prompt_file = work_dir / f"prompt_{session_id[:8]}.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)

    cmd = f'claude --session-id {session_id} --dangerously-skip-permissions'

    child = pexpect.spawn(
        '/bin/bash', ['-c', cmd],
        cwd=str(work_dir),
        env=env,
        timeout=timeout,
        encoding='utf-8',
        dimensions=(80, 200)
    )

    output_lines = []
    try:
        # Wait for the initial prompt (Claude is ready for input)
        child.expect([r'>', r'\$', pexpect.TIMEOUT], timeout=30)

        # Send the prompt
        with open(prompt_file) as f:
            child.sendline(f.read())

        # Collect output until we see the prompt again (Claude is done responding)
        # Look for patterns that indicate Claude is waiting for input
        while True:
            try:
                # Match on common end-of-response patterns
                index = child.expect([
                    r'\n>',           # New prompt
                    r'waiting for',   # Waiting message
                    pexpect.TIMEOUT,
                    pexpect.EOF
                ], timeout=300)

                output_lines.append(child.before)
                if child.after and isinstance(child.after, str):
                    output_lines.append(child.after)

                if index == 0:  # Got new prompt - Claude is done
                    break
                elif index in [2, 3]:  # Timeout or EOF
                    break

            except pexpect.TIMEOUT:
                break
            except pexpect.EOF:
                break

        # Send /exit to trigger the Stop hook
        child.sendline('/exit')

        # Wait for exit or hook interactions
        try:
            child.expect(pexpect.EOF, timeout=120)
            output_lines.append(child.before)
        except pexpect.TIMEOUT:
            pass

    finally:
        child.close()
        try:
            prompt_file.unlink()
        except:
            pass

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

        # Build prompt - add #idle:on for idle conditions
        full_prompt = prompt
        if config.get("idle_enabled"):
            full_prompt = f"#idle:on\n\n{prompt}"

        # Use --print for all conditions (idle plugin now works in --print mode)
        env = dict(config.get("environment", {}))
        cmd = [
            "claude", "--print",
            "--session-id", session_id,
            "--dangerously-skip-permissions",
            full_prompt
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

    # Validate moves - track both validity and puzzle completion
    test_state = HanoiState.initial(n_disks)
    optimal_length = 2 ** n_disks - 1
    solved_at_step = None

    for i, (from_peg, to_peg) in enumerate(actual_moves):
        actual_str = f"{from_peg}->{to_peg}"

        # Check if move is valid for current state
        valid = test_state.is_valid_move(from_peg, to_peg)

        if valid:
            test_state.apply_move(from_peg, to_peg)
            # Check if puzzle is now solved
            if test_state.pegs == goal.pegs and solved_at_step is None:
                solved_at_step = i + 1

        # For MAKER-style error tracking, an error is an INVALID move
        # (Valid but non-optimal moves are still acceptable)
        step_results.append(StepResult(
            step_id=i,
            expected=f"valid_move",
            actual=actual_str,
            correct=valid,
            error_type=None if valid else "invalid_move"
        ))

    # Check if puzzle was solved
    solved = test_state.pegs == goal.pegs

    # Print diagnostic info
    total_moves = len(actual_moves)
    invalid_moves = sum(1 for s in step_results if not s.correct)
    efficiency = optimal_length / total_moves if total_moves > 0 and solved else 0

    print(f"    Moves made: {total_moves} (optimal: {optimal_length})")
    print(f"    Invalid moves: {invalid_moves}")
    print(f"    Solved: {solved} (at step {solved_at_step})" if solved else f"    Solved: {solved}")
    if solved:
        print(f"    Efficiency: {efficiency:.1%}")

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
