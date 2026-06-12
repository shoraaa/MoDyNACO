# Experiment Pipeline Infrastructure

This document describes the structured experiment management system implemented for DyNACO. The pipeline automates the full research workflow: training, evaluation, comparison, and indexing.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Makefile Commands](#makefile-commands)
6. [Scripts](#scripts)
7. [Workflow Examples](#workflow-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline addresses key challenges in research codebases:

- **Reproducibility**: Every experiment captures its full configuration, git state, and timestamp
- **Organization**: All artifacts (checkpoints, logs, results) live in a single self-contained directory
- **Automation**: One-command training, evaluation, and comparison across experiments
- **Indexing**: Central registry (`experiments/index.csv`) tracks all experiments for quick lookup

---

## Directory Structure

```
experiments/
├── {experiment_name}/
│   ├── config.yaml          # Full YAML config + metadata (git commit, timestamp)
│   ├── checkpoints/         # Model checkpoints
│   │   └── {problem}/n{n_node}/
│   │       ├── best.pt
│   │       ├── last.pt
│   │       └── epoch{0,1,...}.pt
│   ├── logs/
│   │   └── stdout.txt       # Real-time training output
│   └── results/
│       ├── final.json       # Aggregated evaluation metrics
│       └── {problem}_n{n_node}_summary.json  # Raw test output
├── index.csv                # Master index (auto-generated)
```

**Generated comparison files** (in project root):
- `comparison.csv` - Spreadsheet-ready
- `comparison.tex` - LaTeX table
- `comparison.md` - Markdown table

---

## Quick Start

### 1. Create a YAML Configuration

```yaml
# configs/my_experiment.yaml
problem: tsp
n_node: 1000
k_sparse: 32
n_ants: 100
H: 10
mini_H: 100
epochs: 10
algo: ppo
rho: 0.5
lr: 0.000005
device: cuda:0
seed: 1234

# Experiment metadata (not passed to train.py)
experiment_name: my_tsp_1000_run1
description: "TSP-1000 baseline with PPO"
author: "Your Name"
use_wandb: false
```

### 2. Train

```bash
make train CONFIG=configs/my_experiment.yaml
```

Or manually:
```bash
uv run scripts/train.py --config configs/my_experiment.yaml
```

### 3. Evaluate

```bash
make eval EXP=experiments/my_tsp_1000_run1_20260420_123456
```

Or manually:
```bash
uv run scripts/evaluate.py --experiment experiments/my_tsp_1000_run1_20260420_123456
```

### 4. Compare All Experiments

```bash
make compare
```

This scans `experiments/*/results/final.json` and generates `comparison.csv`, `comparison.tex`, `comparison.md`.

---

## Configuration

### YAML Schema

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `problem` | str | `tsp`, `cvrp`, `bpp`, `mkp`, or `op` |
| `n_node` | int | Number of nodes/items |
| `k_sparse` | int | K-NN graph sparsity |
| `n_ants` | int | Number of ants |
| `H` | int | Outer iterations (guidance updates) |
| `mini_H` | int | Inner iterations per update |
| `epochs` | int | Training epochs |
| `algo` | str | `ppo` or `reinforce` |
| `rho` | float | Pheromone evaporation (0.0-1.0) |
| `lr` | float | Learning rate (e.g., `5e-6`) |
| `device` | str | `cuda:0` or `cpu` |

#### Problem-Specific Fields

| Problem | Additional Fields |
|---------|------------------|
| `bpp` | `capacity` (float) |
| `mkp` | `m` (int, number of constraints) |
| `op` | `max_len` (float, route length budget) |

#### Metadata Fields (Excluded from train.py)

These are captured for bookkeeping but **not** passed to `train.py`:

| Field | Type | Description |
|-------|------|-------------|
| `experiment_name` | str | Short name (auto-generated from filename if omitted) |
| `description` | str | Human-readable description |
| `author` | str | Experiment author |
| `use_wandb` | bool | Enable Weights & Biases logging |

---

## Makefile Commands

```makefile
# Train with YAML config
make train CONFIG=configs/experiment.yaml

# Evaluate an experiment directory
make eval EXP=experiments/exp_name_timestamp

# Generate comparison tables across all experiments
make compare

# Rebuild experiment index manually
make index

# Run linter (requires ruff in environment)
make lint

# Clean up old artifacts (optional)
make clean-old
```

**Note**: `make train` creates the experiment directory automatically under `experiments/` with a timestamp suffix if `experiment_name` is provided in the config, or uses the config filename + timestamp.

---

## Scripts

### `scripts/train.py` - Training Wrapper

**Purpose**: Entry point for training with YAML configuration and experiment management.

**Key features**:
- Loads YAML config
- Creates `experiments/{experiment_name}/` with subdirectories
- Saves full config (including git metadata) to `config.yaml`
- Streams training output in real-time to both console and `logs/stdout.txt`
- Passes CLI arguments to `train.py` (filtering out metadata fields)
- Sets `--save_dir` to experiment's `checkpoints/` subfolder
- Launches index rebuild in background (non-blocking)

**Invocation**:
```bash
uv run scripts/train.py --config path/to/config.yaml [--experiment-name override_name] [--no-index]
```

### `scripts/evaluate.py` - Batch Evaluation

**Purpose**: Evaluate a trained experiment and save standardized results.

**Key features**:
- Loads experiment's `config.yaml`
- Recursively finds best checkpoint (`best.pt`, `last.pt`, or first `*.pt`)
- Builds appropriate command for base (`tsp`, `cvrp`) vs extended (`bpp`, `mkp`, `op`) problems
- For base problems: uses `--summary_json`
- For extended problems: uses `--test_size`, `--save_results`, `--save_dir`
- Copies output to `results/final.json` for consistency

**Invocation**:
```bash
uv run scripts/evaluate.py --experiment experiments/{exp_dir}
```

**Output**: `experiments/{exp_dir}/results/final.json`

### `scripts/compare.py` - Cross-Experiment Comparison

**Purpose**: Aggregate results from all experiments into comparison tables.

**Key features**:
- Scans `experiments/*/results/final.json`
- Extracts metrics: `test_cost_mean`, `test_cost_std`, improvement percentages
- Includes hyperparameters from each config
- Outputs three formats: CSV, LaTeX, Markdown

**Invocation**:
```bash
uv run scripts/compare.py
```

**Output files**:
- `comparison.csv` - Spreadsheet-ready
- `comparison.tex` - LaTeX table with `\toprule`, `\midrule`
- `comparison.md` - Markdown table

### `scripts/indexExperiments.py` - Master Index

**Purpose**: Maintain a central CSV index of all experiments with results.

**Invocation**:
```bash
uv run scripts/indexExperiments.py --rebuild
```

**Output**: `experiments/index.csv`

**Columns**:
- `experiment` - Experiment directory name
- `problem`, `n_node` - Problem type and size
- `test_cost_mean`, `test_cost_std` - Primary metrics
- `aco_improvement_pct_mean` - vs. pure ACO
- `static_gap_pct_mean` - vs. static prior
- `checkpoint` - Path to best checkpoint
- `git_commit`, `timestamp` - Version control metadata
- Plus all hyperparameters (H, algo, epochs, k_sparse, lr, mini_H, n_ants, rho, seed, etc.)

The index is automatically updated in the background after each training run (unless `--no-index` is passed).

---

## Workflow Examples

### Example 1: TSP-1000 Baseline

```bash
# 1. Create config
cat > configs/tsp1000_baseline.yaml <<EOF
problem: tsp
n_node: 1000
k_sparse: 32
n_ants: 100
H: 10
mini_H: 100
epochs: 10
algo: ppo
rho: 0.5
lr: 0.000005
device: cuda:0
seed: 1234

experiment_name: tsp1000_baseline
description: "TSP-1000 baseline PPO"
use_wandb: false
EOF

# 2. Train
make train CONFIG=configs/tsp1000_baseline.yaml

# 3. Evaluate
EXP_DIR=$(ls -d experiments/tsp1000_baseline_* | head -1)
make eval EXP=$EXP_DIR

# 4. Compare across all experiments
make compare

# 5. Check index
cat experiments/index.csv
```

### Example 2: Hyperparameter Sweep

```bash
# Create multiple configs with different learning rates
for lr in 1e-6 3e-6 5e-6 1e-5; do
  cat > configs/sweep_lr_${lr}.yaml <<EOF
problem: tsp
n_node: 100
k_sparse: 20
n_ants: 20
H: 10
mini_H: 100
epochs: 2
algo: ppo
rho: 0.1
lr: ${lr}
device: cuda:0
seed: 1234
experiment_name: tsp100_lr_${lr}
use_wandb: false
EOF
done

# Train all configs in parallel (background)
for cfg in configs/sweep_lr_*.yaml; do
  make train CONFIG=$cfg &
done
wait

# Evaluate all
for exp in experiments/tsp100_lr_*; do
  make eval EXP=$exp &
done
wait

# Generate comparison
make compare

# View results
column -t -s, comparison.csv | less
```

### Example 3: Extended Problem (BPP-50)

```bash
# Config
cat > configs/bpp50.yaml <<EOF
problem: bpp
n_node: 50
capacity: 150.0
k_sparse: 32
n_ants: 20
H: 5
mini_H: 5
epochs: 5
algo: ppo
rho: 0.9
lr: 0.0003
device: cuda:0
seed: 1234
experiment_name: bpp50_run1
description: "BPP-50 baseline"
use_wandb: false
EOF

# Train & evaluate
make train CONFIG=configs/bpp50.yaml
EXP_DIR=$(ls -d experiments/bpp50_run1_* | head -1)
make eval EXP=$EXP_DIR
make compare
```

---

## Troubleshooting

### Issue: "No checkpoint files found" during evaluation

**Cause**: Checkpoints saved in subdirectories (`checkpoints/tsp/n100/`) but evaluator was only checking top-level.

**Fix**: Updated `scripts/evaluate.py` uses `rglob("*.pt")` to search recursively. Ensure you have the latest version.

### Issue: Evaluation fails with "unrecognized arguments: --test_size"

**Cause**: The `test.py` base parser (for TSP/CVRP) doesn't have `--test_size`, `--save_dir`, `--save_results`. Those are only for extended problems.

**Fix**: `scripts/evaluate.py` now detects problem type and passes appropriate flags:
- Base problems (TSP/CVRP): uses `--summary_json`
- Extended problems (BPP/MKP/OP): uses `--test_size --save_results --save_dir`

### Issue: Index not updating after training

**Cause**: Background index rebuild may have failed or been killed.

**Fix**: Run manually:
```bash
uv run scripts/indexExperiments.py --rebuild
```

### Issue: Linter fails with "Failed to spawn: ruff"

**Cause**: `ruff` not installed in uv environment.

**Fix**: Either install ruff (`uv add ruff --dev`) or skip linting. The pipeline otherwise works without it.

### Issue: Checkpoints appear empty or missing after training

**Cause**: Wrapper wasn't passing `--save_dir` to `train.py`, so checkpoints went to default `pretrained/` directory.

**Fix**: Ensure `scripts/train.py` includes:
```python
cli_args.extend([f"--save_dir={exp_dir / 'checkpoints'}"])
```

### Issue: "Error: No checkpoint files in experiments/.../checkpoints"

**Cause**: Either training failed or the checkpoint path is wrong.

**Debug**:
```bash
# Check if training completed successfully
tail -50 experiments/{exp_dir}/logs/stdout.txt

# Find checkpoints
find experiments/{exp_dir}/checkpoints -name "*.pt" -type f
```

If checkpoints exist in subdirectories, the evaluator's recursive search should find them.

---

## Best Practices

1. **Version Control**: Commit experiments configs (not the `experiments/` output) to track what was run.
2. **Naming**: Use descriptive `experiment_name` in configs for easy identification.
3. **Seeds**: Always set `seed` for reproducibility.
4. **Metadata**: Fill in `description` and `author` for lab notebook provenance.
5. **W&B**: Enable `use_wandb: true` for detailed run tracking and plots.
6. **Indexing**: Run `make compare` regularly to see the leaderboard.
7. **Cleanup**: Periodically archive old experiments with `tar -czf` and remove from working dir.

---

## Future Enhancements

Planned improvements:
- Hyperparameter sweep automation (grid/random search)
- Automatic plotting of learning curves across experiments
- Experiment pruning based on early stopping criteria
- Integration with SLURM/LSF for cluster scheduling
- Regression test suite for pipeline scripts

---

## Changelog

**2026-04-20** (v1.0):
- Initial implementation of structured experiment pipeline
- Wrapper scripts: `train.py`, `evaluate.py`, `compare.py`, `indexExperiments.py`
- Makefile automation
- Recursive checkpoint discovery
- Base/extended problem flag handling
