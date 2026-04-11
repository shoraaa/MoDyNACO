#!/usr/bin/env python3
"""
Test ACO with rho=0.3 and initial pheromone=5.0.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_aco_with_updated_params(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test ACO with updated parameters."""
    print(f"\n{'='*60}")
    print(f"Testing ACO with updated parameters (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test different iteration counts
    iteration_counts = [1, 5, 10, 20]

    for n_iters in iteration_counts:
        print(f"\n--- Testing with {n_iters} iterations ---")

        # Create ACO solver with updated parameters
        aco = MFACO_BPP(
            demand=demand,
            capacity=capacity,
            n_ants=n_ants,
            decay=0.3,  # Updated from 0.5 to 0.3
            alpha=1.0,
            beta=1.0,
            elitist=True,
            device=device,
        )

        # Sample initial solutions
        costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        initial_best = float(torch.tensor(costs).min().item())
        initial_mean = float(torch.tensor(costs).mean().item())

        print(f"  Initial sample: best={initial_best:.4f}, mean={initial_mean:.4f}")

        # Run ACO for n_iters iterations
        best_fitness = aco.run(n_iters)

        # Sample final solutions
        costs_final, paths_final, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        final_best = float(torch.tensor(costs_final).min().item())
        final_mean = float(torch.tensor(costs_final).mean().item())

        improvement = initial_best - final_best
        print(f"  After {n_iters} iterations: best={final_best:.4f}, mean={final_mean:.4f}")
        print(f"  Improvement: {improvement:+.4f}")

        # Check pheromone statistics
        pheromone = aco.pheromone
        print(f"  Pheromone: mean={pheromone.mean().item():.6f}, std={pheromone.std().item():.6f}")

    print(f"\n{'='*60}")
    print("Summary:")
    print("  - Check if updated parameters improve ACO performance")
    print("  - rho=0.3, initial_pheromone=5.0")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_aco_with_updated_params(n_node=30, device=device)
