# idle-evaluation

Evaluation framework for the idle Claude Code plugin.

## Purpose

This repository measures how idle improves Claude Code through:
1. Mandatory peer review (alice)
2. Multi-model consensus (reviewing skill)
3. Persistent state (jwz)
4. Issue-driven workflow (tissue)

## Running Experiments

### Long-Horizon Tasks
```bash
python experiments/long-horizon/runner.py --condition <baseline|idle-full|idle-no-alice> --runs 5
```

### Error Accumulation
```bash
python experiments/error-correction/runner.py --task hanoi --disks 10 --condition <condition>
```

### SWE-bench Subset
```bash
python experiments/swe-bench/runner.py --condition <condition> --runs 5
```

## Analyzing Results

```bash
# Generate figures
python figures/generate.py

# View summary
python -c "from metrics import summarize; summarize('results/')"
```

## Conditions

| Condition | Config Location |
|-----------|-----------------|
| baseline | harnesses/baseline/config.json |
| idle-full | harnesses/idle/config.json |
| idle-no-alice | harnesses/idle-no-review/config.json |
| idle-sonnet-alice | harnesses/idle-no-consensus/config.json |

## Key Metrics

- **pass^k**: All k runs must pass (reliability)
- **per-step error rate**: Track errors at each step
- **task completion**: Binary success
- **cost efficiency**: Tokens per success

## Output Format

Results are stored as JSONL in `results/`:
```json
{"task_id": "...", "condition": "...", "run": 1, "success": true, "steps": 42, "errors": [], "tokens": 15000}
```
