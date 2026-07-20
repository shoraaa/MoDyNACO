# D2NACO: Dynamic and Diverse Neural Guidance for Large-Scale Ant Colony Optimization

> **D2NACO** — *Dynamic and Diverse Neural ACO* | Paper (in preparation)

## Overview

Neural-guided Ant Colony Optimization (ACO) suffers from two coupled limitations. First, a **training–inference misalignment**: policies are trained to produce a static heatmap from instance geometry alone, yet are deployed inside iterative search loops where the pheromone field evolves over hundreds of iterations — the model never observes search progress and cannot adapt. Second, a **single-policy bottleneck**: one heatmap drives every ant, so the colony explores a narrow region of the solution space and stalls prematurely.

**D2NACO** addresses both. It is built on two pillars:

### 1. Dynamic guidance

Formulated as a semi-MDP, a lightweight meta-policy (~50K parameters) periodically observes the evolving pheromone distribution and incumbent solution, emitting updated edge-level guidance throughout the search trajectory.

- **State-Aware Representation** — conditions the policy on per-edge pheromone statistics and incumbent topology, enabling search-phase-dependent guidance.
- **Trajectory-Aware Training** — optimizes expected cost across the full search history via PPO, aligning training with iterative search dynamics.

### 2. Diverse guidance

A shared encoder feeds a **multi-head decoder** that emits a *population* of distinct edge-guidance policies. A learned **allocator** distributes heads across the colony, and a diversity objective keeps the heads complementary rather than collapsed onto one policy.

- **Multi-Head Policy Population** — one shared GNN encoder + several low-overhead decoder heads (LoRA / deep-LoRA / PolyNet / FiLM adapters), each specializing to a different guidance regime.
- **Learned Allocator** — a PPO-trained router assigns heads to ants, balancing exploitation of strong heads with exploration of the rest.
- **Diversity Objective** — a Jensen–Shannon head-divergence reward prevents head collapse, so the colony searches a genuinely broader region.

To scale to 100K nodes, D2NACO pairs the policy with a perturbation-based ACO backend and **Scope-Restricted Refinement (SRR)**, which preserves gradient fidelity at $O(M \cdot K)$ cost per ant, independent of instance size.

### Key Results (dynamic guidance)

| | TSP-1K | TSP-10K | TSP-100K | CVRP-1K | CVRP-100K |
|---|---|---|---|---|---|
| **Gap** | 0.20% | 0.82% | 1.90% | 1.04% | 6.75% |
| **vs. ACO** | −58% | −59% | −39% | −44% | −7% |

- Outperforms neural baselines across all scales (1K–100K)
- Reduces runtime by 20–33% on TSP compared to unguided ACO; <1% overhead on CVRP
- Zero-shot transfer from 1K training to 86K-node TSPLIB instances
- Diverse multi-head guidance further improves in-distribution search and robustness across TSP, CVRP, BPP, MKP, and OP

## Installation

### Prerequisites

- Linux (tested on Ubuntu)
- Python ≥ 3.13
- CUDA-capable GPU
- C++17 compiler with OpenMP support
- [`uv`](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Create environment and install dependencies
uv sync

# Build the C++ backends: faco_opt (perturbation-based ACO + SRR for TSP/CVRP)
# and alphaant_tsp_aco_cpp (extended-problem ACO for BPP/MKP/OP)
cd src
uv run python setup.py build_ext --inplace
cd ..
```

### Verify installation

```bash
uv run python -c "import faco_opt; print('faco_opt OK')"
uv run python -c "import faco; faco.load_alphaant_cpp_module(); print('extended backend OK')"
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

## Usage

### Quick Pipeline (Recommended)

The repository includes a complete experiment-management pipeline. See [PIPELINE.md](PIPELINE.md) for full documentation.

```bash
# 1. Configure experiment via YAML (see configs/ for examples)
# 2. Train
make train CONFIG=configs/train/tsp_n1000_ppo.yaml
# 3. Evaluate
make eval EXP=experiments/{experiment_name}_timestamp
# 4. Compare across all experiments
make compare
# 5. View master index
cat experiments/index.csv
```

All artifacts are saved under `experiments/{name}/` with `config.yaml` (config + git metadata), `checkpoints/`, `logs/stdout.txt`, and `results/final.json`.

### Direct Training

Train the dynamic single-head policy on TSP-1K (~30 min on RTX 5090):

```bash
uv run python train.py --problem tsp --n_node 1000
```

Train on CVRP-1K:

```bash
uv run python train.py --problem cvrp --n_node 1000
```

Train the **diverse multi-head** policy (adds the diversity pillar on top of dynamic guidance):

```bash
uv run python train.py --problem tsp --n_node 1000 \
    --multi_head --num_heads 4 \
    --head_decoder_type polynet \
    --loss_js 0.1 --allocator_temperature 1.0
```

Train the unified entrypoint on extended combinatorial problems:

```bash
# Bin Packing
uv run python train.py --problem bpp --n_node 50 --capacity 150
# Multiple Knapsack
uv run python train.py --problem mkp --n_node 50 --m 5
# Orienteering
uv run python train.py --problem op --n_node 50 --max_len 4.0
```

Scale to larger instances:

```bash
# TSP-10K (~2 hours)
uv run python train.py --problem tsp --n_node 10000
# TSP-100K (~4 hours)
uv run python train.py --problem tsp --n_node 100000
```

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--problem` | (required) | `tsp`, `cvrp`, `bpp`, `mkp`, or `op` |
| `--n_node` | 1000 | Problem size |
| `--k_sparse` | 32 | K-NN candidate graph size |
| `--n_ants` | 100 | Number of ants |
| `--H` | 10 | Outer steps (guidance updates) |
| `--mini_H` | 100 | Inner steps per guidance update |
| `--epochs` | 10 | Training epochs |
| `--algo` | ppo | `ppo` or `reinforce` |
| `--rho` | 0.5 | Pheromone evaporation rate |
| `--lr` / `--ppo_lr` | 5e-6 | Learning rate |
| `--multi_head` | off | Enable the diverse multi-head policy population |
| `--num_heads` | 1 | Number of decoder heads |
| `--head_decoder_type` | lora | Head adapter: `lora`, `deep_lora`, `polynet`, `film`, `multi_decoder`, `lowrank` |
| `--loss_js` | 0.0 | Jensen–Shannon head-diversity reward coefficient |
| `--allocator_temperature` | 1.0 | Softmax temperature for the head-to-ant allocator |
| `--device` | cuda:0 | Device |
| `--save_dir` | pretrained | Checkpoint directory |
| `--capacity` | 150.0 | Bin capacity for `bpp` |
| `--m` | 5 | Number of constraints for `mkp` |
| `--max_len` | 4.0 | Route-length budget for `op` |

### Evaluation

Evaluate a trained checkpoint:

```bash
# TSP-1K with 1K iterations
uv run python test.py \
    --problem tsp --n_node 1000 \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --H 10 --mini_H 100

# Diverse multi-head checkpoint
uv run python test.py \
    --problem tsp --n_node 1000 \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --multi_head --num_heads 4 --head_decoder_type polynet \
    --H 10 --mini_H 100
```

Evaluate on TSPLIB/CVRPlib real-world instances:

```bash
uv run python test.py \
    --problem tsp \
    --checkpoint pretrained/tsp/n1000/best.pt \
    --rl_data --H 10 --mini_H 100
```

Run the unguided ACO baseline (no neural guidance):

```bash
uv run python test.py --problem tsp --n_node 1000 --no_model
```

Key evaluation arguments:

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | none | Path to trained model |
| `--H` | 10 | Outer steps |
| `--mini_H` | 100 | Inner steps per outer step |
| `--multi_head` | off | Load and evaluate a multi-head policy |
| `--num_heads` | 1 | Number of heads (usually inferred from checkpoint) |
| `--allocator_temperature` | 1.0 | Head-to-ant allocation temperature |
| `--rl_data` | false | Use TSPLIB/CVRPlib instances |
| `--dataset` | none | Custom dataset path |
| `--no_model` | false | Run unguided ACO only |
| `--timed` | false | Enable detailed timing |
| `--warmup` | true | Phased injection |
| `--no_anneal` | false | Disable guidance annealing |
| `--test_size` | 16 | Test instances for `bpp`/`mkp`/`op` |
| `--static_compare` | false | Compare dynamic vs static prior on `bpp`/`mkp`/`op` |

## Project Structure

```
├── train.py            # Unified training entrypoint (TSP/CVRP/BPP/MKP/OP; single- & multi-head)
├── test.py             # Unified evaluation entrypoint
├── net.py              # GNN encoder + dynamic decoder + MultiHeadNet decoder population (~50K params)
├── faco.py             # ACO environment wrapper (perturbation-based backend + SRR)
├── extended_common.py  # Shared utilities for BPP/MKP/OP
├── baselines.py        # Baseline solvers
├── utils.py            # Utilities, metrics, analysis tools
├── src/                # C++ backends: faco_opt (mfaco_train.cpp/binding.cpp) +
│                       #   extended ACO (aco.cpp/aco_more.inc) + pybind11 bindings
├── configs/            # YAML experiment configs (train / eval / ablation / extended)
├── data/               # Benchmark datasets
├── scripts/            # Experiment and analysis scripts
├── tools/              # Log explorer and tooling
└── tests/              # Regression tests
```

## Acknowledgements

The code was built partly on top of [DeepACO](https://github.com/henry-yeh/DeepACO) structures.
