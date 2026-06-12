# DyNACO Experiment Pipeline - Quick Reference

## Common Commands

```bash
# Train an experiment from YAML config
make train CONFIG=configs/experiment.yaml

# Evaluate a completed experiment
make eval EXP=experiments/experiment_name_timestamp

# Compare all experiments (generates CSV, LaTeX, Markdown)
make compare

# Rebuild experiment index manually
make index

# View master index
cat experiments/index.csv

# Run linter (if ruff installed)
make lint
```

## Experiment Directory Structure

```
experiments/{name}/
├── config.yaml          # Full config + git metadata
├── checkpoints/
│   └── {problem}/n{n_node}/
│       ├── best.pt
│       ├── last.pt
│       └── epoch*.pt
├── logs/
│   └── stdout.txt
└── results/
    └── final.json
```

## Key Scripts

- `scripts/train.py` - Training wrapper with YAML support
- `scripts/evaluate.py` - Batch evaluator (handles TSP/CVRP vs BPP/MKP/OP)
- `scripts/compare.py` - Cross-experiment comparison tables
- `scripts/indexExperiments.py` - Master index builder

## Configuration Template

```yaml
problem: tsp              # tsp, cvrp, bpp, mkp, op
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

# Metadata (not passed to train.py)
experiment_name: my_experiment
description: "Brief description"
use_wandb: false
```

## Problem-Specific Parameters

| Problem | Extra Fields |
|---------|--------------|
| `bpp` | `capacity: 150.0` |
| `mkp` | `m: 5` (constraints) |
| `op` | `max_len: 4.0` |

## Output Files

- `comparison.csv` - Spreadsheet-ready comparison
- `comparison.tex` - LaTeX table
- `comparison.md` - Markdown table
- `experiments/index.csv` - Master experiment registry

See [PIPELINE.md](PIPELINE.md) for full documentation.
