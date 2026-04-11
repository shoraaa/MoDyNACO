#!/usr/bin/env python3
"""
Diagnostic test to investigate pheromone behavior during ACO iterations.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_pheromone_evolution(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test pheromone evolution during ACO iterations."""
    print(f"\n{'='*60}")
    print(f"Testing pheromone evolution (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Create ACO solver
    aco = MFACO_BPP(
        demand=demand,
        capacity=capacity,
        n_ants=n_ants,
        decay=0.5,
        alpha=1.0,
        beta=1.0,
        elitist=True,
        device=device,
    )

    print(f"Initial pheromone statistics:")
    print(f"  Mean: {aco.pheromone.mean().item():.6f}")
    print(f"  Std: {aco.pheromone.std().item():.6f}")
    print(f"  Min: {aco.pheromone.min().item():.6f}")
    print(f"  Max: {aco.pheromone.max().item():.6f}")
    print(f"  Non-zero ratio: {(aco.pheromone > 0).float().mean().item():.4f}\n")

    # Track pheromone evolution
    print("Running ACO iterations and tracking pheromone:\n")

    for i in range(10):
        # Sample solutions
        costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        best_cost = float(torch.tensor(costs).min().item())
        mean_cost = float(torch.tensor(costs).mean().item())

        # Run 1 iteration
        aco.run(1)

        # Check pheromone
        phero_mean = aco.pheromone.mean().item()
        phero_std = aco.pheromone.std().item()
        phero_min = aco.pheromone.min().item()
        phero_max = aco.pheromone.max().item()
        phero_nonzero = (aco.pheromone > 0).float().mean().item()

        print(f"Iteration {i+1}:")
        print(f"  Cost: best={best_cost:.2f}, mean={mean_cost:.2f}")
        print(f"  Pheromone: mean={phero_mean:.6f}, std={phero_std:.6f}")
        print(f"  Pheromone range: [{phero_min:.6f}, {phero_max:.6f}]")
        print(f"  Non-zero ratio: {phero_nonzero:.4f}")

        # Check if pheromone is being reset
        if i > 0 and phero_mean < 0.01:
            print(f"  WARNING: Pheromone mean dropped to {phero_mean:.6f}!")

    print(f"\n{'='*60}")
    print("Analysis:")
    print("  - Check if pheromone is decaying too fast")
    print("  - Check if pheromone is being reset incorrectly")
    print("  - Check if elitist update is working")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_pheromone_evolution(n_node=30, device=device)
