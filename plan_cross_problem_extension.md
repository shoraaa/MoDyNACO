# Plan: Cross-Problem Extension for BPP, MKP, OP

## Overview

Extend NGFACO to support Bin Packing Problem (BPP), Multi-dimensional Knapsack Problem (MKP), and Orienteering Problem (OP) by leveraging existing C++ implementations from AlphaAnt.

## Requirements

- Test at small scales only: H=10, mini_H=10
- Use test data from AlphaAnt directory
- Separate scripts from current `reviewer_experiments.py`
- Run end-to-end test with small parameters before completion

## Architecture

### Current NGFACO Structure

```
NGFACO/
├── faco.py              # ACO classes for TSP, CVRP
├── net.py               # Neural network architecture
├── train.py             # Training script
├── test.py              # Testing script
├── utils.py             # Utility functions
└── reviewer_experiments.py  # Reviewer experiment pipeline
```

### Proposed New Structure

```
NGFACO/
├── faco.py              # Existing TSP, CVRP
├── faco_extended.py     # NEW: BPP, MKP, OP ACO classes
├── net_extended.py      # NEW: Network architectures for BPP, MKP, OP
├── train_extended.py    # NEW: Training script for BPP, MKP, OP
├── test_extended.py     # NEW: Testing script for BPP, MKP, OP
├── utils_extended.py    # NEW: Data generation utilities
├── cross_problem_experiments.py  # NEW: Cross-problem experiment pipeline
└── reviewer_experiments.py  # Existing (unchanged)
```

## Implementation Plan

### Phase 1: C++ Module Integration

**Goal**: Load and use AlphaAnt's C++ ACO implementations

**Tasks**:
1. Create `cpp_aco_wrapper.py` to load AlphaAnt's C++ module
   - Use `~/Research/AlphaAnt/src/aco.cpp` and `aco_more.inc`
   - Expose `ACO_BPP`, `ACO_MKP`, `ACO_OP` classes
   - Handle numpy/torch conversions

**Files to create**:
- `cpp_aco_wrapper.py`

**Key functions**:
```python
def load_alphaant_cpp_module():
    """Load the AlphaAnt C++ ACO module"""
    # Load from ~/Research/AlphaAnt/src/aco.cpp
    # Return module with ACO_BPP, ACO_MKP, ACO_OP classes
```

### Phase 2: ACO Wrapper Classes

**Goal**: Create Python ACO classes that integrate with learning guidance

**Tasks**:
1. Create `faco_extended.py` with:
   - `MFACO_BPP` - Learning-guided ACO for Bin Packing
   - `MFACO_MKP` - Learning-guided ACO for Multi-dimensional Knapsack
   - `MFACO_OP` - Learning-guided ACO for Orienteering Problem

**Interface requirements** (matching existing `MFACO_TSP`/`MFACO_CVRP`):
```python
class MFACO_BPP:
    def __init__(self, demand, capacity, n_ants, k_sparse, ...):
        # Initialize with problem data

    def sample(self, invtemp=1.0, require_prob=False, prior=None, parallel_traced=True):
        # Sample solutions with optional neural prior
        # Returns: (solutions, log_probs, trace)

    def update_pheromone(self, best_solution, best_cost):
        # Update pheromone using best solution

    def prob_sparse_torch(self, invtemp=1.0, prior=None):
        # Compute probability tensor for learning
```

**Key differences from TSP/CVRP**:
- BPP: Bin capacity constraints, minimize number of bins
- MKP: Multiple knapsack capacity constraints, maximize total value
- OP: Budget/time constraint, maximize collected reward

### Phase 3: Data Generation

**Goal**: Create utilities for generating problem instances

**Tasks**:
1. Create `utils_extended.py` with:
   - `gen_bpp_instance(n, device)` - Generate BPP instance
   - `gen_mkp_instance(n, m, device)` - Generate MKP instance
   - `gen_op_instance(n, device)` - Generate OP instance
   - `build_pyg_data_bpp(demand, device)` - Build PyG data for BPP
   - `build_pyg_data_mkp(prize, weight, device)` - Build PyG data for MKP
   - `build_pyg_data_op(coords, device)` - Build PyG data for OP

**Data sources**:
- Use AlphaAnt's data generation functions from `~/Research/AlphaAnt/core/problem_data/`
- Load test datasets from `~/Research/AlphaAnt/data/`

### Phase 4: Network Architecture

**Goal**: Create network architectures for BPP, MKP, OP

**Tasks**:
1. Create `net_extended.py` with:
   - `NetBPP` - GNN for Bin Packing
   - `NetMKP` - GNN for Multi-dimensional Knapsack
   - `NetOP` - GNN for Orienteering Problem

**Architecture considerations**:
- BPP: Node features = demand (1D)
- MKP: Node features = prize + weight (m+1D)
- OP: Node features = distance to depot + prize (2D)

### Phase 5: Training Script

**Goal**: Create training script for BPP, MKP, OP

**Tasks**:
1. Create `train_extended.py` with:
   - Command-line argument parsing
   - Training loop with REINFORCE/PPO
   - Validation and checkpointing
   - WandB logging

**Training parameters (small scale)**:
```python
H = 10              # Outer iterations
mini_H = 10         # Inner iterations (macro-action duration)
n_ants = 10         # Number of ants
k_sparse = 10       # K-NN size
epochs = 5          # Training epochs
steps_per_epoch = 4 # Steps per epoch
```

### Phase 6: Testing Script

**Goal**: Create testing script for BPP, MKP, OP

**Tasks**:
1. Create `test_extended.py` with:
   - Load trained models
   - Run ACO with learned heuristics
   - Compare against baselines
   - Report metrics

### Phase 7: Cross-Problem Experiment Pipeline

**Goal**: Create experiment pipeline for cross-problem evaluation

**Tasks**:
1. Create `cross_problem_experiments.py` with:
   - `run_bpp_experiments()` - BPP experiments
   - `run_mkp_experiments()` - MKP experiments
   - `run_op_experiments()` - OP experiments
   - Progress tracking and result aggregation

**Experiment phases**:
1. Training stability - Compare learning vs. no learning
2. Inference comparison - Compare with baselines
3. Cross-scale generalization - Test on different problem sizes

### Phase 8: End-to-End Testing

**Goal**: Verify the entire pipeline works correctly

**Tasks**:
1. Run end-to-end test with minimal parameters:
   ```bash
   python train_extended.py --problem bpp --n_node 20 --H 5 --mini_H 5 --n_ants 5 --epochs 2 --steps_per_epoch 2
   python test_extended.py --problem bpp --n_node 20 --checkpoint checkpoints/bpp_n20_*.pt
   ```
2. Verify:
   - Training completes without errors
   - Checkpoints are saved
   - Testing loads checkpoints correctly
   - Metrics are computed and reported

## File Structure

```
NGFACO/
├── cpp_aco_wrapper.py          # NEW: C++ module loader
├── faco_extended.py            # NEW: ACO classes for BPP, MKP, OP
├── net_extended.py             # NEW: Network architectures
├── train_extended.py           # NEW: Training script
├── test_extended.py            # NEW: Testing script
├── utils_extended.py           # NEW: Data generation utilities
├── cross_problem_experiments.py  # NEW: Experiment pipeline
├── faco.py                     # Existing (unchanged)
├── net.py                      # Existing (unchanged)
├── train.py                    # Existing (unchanged)
├── test.py                     # Existing (unchanged)
├── utils.py                    # Existing (unchanged)
└── reviewer_experiments.py     # Existing (unchanged)
```

## Dependencies

- AlphaAnt C++ module: `~/Research/AlphaAnt/src/aco.cpp`
- AlphaAnt data: `~/Research/AlphaAnt/data/`
- PyTorch, PyTorch Geometric
- NumPy

## Success Criteria

1. All three problems (BPP, MKP, OP) can be trained and tested
2. Training completes without errors on small-scale instances
3. Checkpoints can be saved and loaded
4. Metrics are computed and reported correctly
5. Cross-problem experiment pipeline runs successfully

## Timeline

- Phase 1: C++ Module Integration - 1 day
- Phase 2: ACO Wrapper Classes - 2 days
- Phase 3: Data Generation - 1 day
- Phase 4: Network Architecture - 1 day
- Phase 5: Training Script - 2 days
- Phase 6: Testing Script - 1 day
- Phase 7: Cross-Problem Experiments - 2 days
- Phase 8: End-to-End Testing - 1 day

Total: ~11 days

## Notes

- The C++ implementations from AlphaAnt are already efficient and well-tested
- The main adaptation work is integrating learning guidance and creating the training/testing infrastructure
- Small-scale testing (H=10, mini_H=10) will be used for initial validation
- Cross-problem experiments will be separate from the existing reviewer_experiments.py pipeline
