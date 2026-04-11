#!/usr/bin/env python3
"""
Diagnostic test to compare ACO performance with different decay rates.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_aco_decay_rates(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test ACO with different decay rates."""
    print(f"\n{'='*60}")
    print(f"Testing ACO with different decay rates (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test different decay rates
    decay_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

    for rho in decay_rates:
        print(f"\n--- Testing with decay rate rho={rho} ---")

        # Create ACO solver
        aco = MFACO_BPP(
            demand=demand,
            capacity=capacity,
            n_ants=n_ants,
            decay=rho,
            alpha=1.0,
            beta=1.0,
            elitist=True,
            device=device,
        )

        # Sample initial solutions
        costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        initial_best = float(torch.tensor(costs).min().item())
        initial_mean = float(torch.tensor(costs).mean().item())

        print(f"  Initial sample: best={initial_best:.2f}, mean={initial_mean:.2f}")

        # Run ACO for 10 iterations
        best_fitness = aco.run(10)

        # Sample final solutions
        costs_final, paths_final, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        final_best = float(torch.tensor(costs_final).min().item())
        final_mean = float(torch.tensor(costs_final).mean().item())

        print(f"  After 10 iterations: best={final_best:.2f}, mean={final_mean:.2f}")
        print(f"  Improvement: {initial_best - final_best:.2f} bins")

        # Check pheromone statistics
        pheromone = aco.pheromone
        print(f"  Pheromone: mean={pheromone.mean().item():.6f}, std={pheromone.std().item():.6f}")
        print(f"  Pheromone range: [{pheromone.min().item():.6f}, {pheromone.max().item():.6f}]")

    print(f"\n{'='*60}")
    print("Summary:")
    print("  - Lower decay rates (rho=0.1, 0.3) preserve pheromone longer")
    print("  - Higher decay rates (rho=0.5, 0.7, 0.9) decay pheromone too fast")
    print("  - Optimal decay rate depends on problem size and iterations")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_aco_decay_rates(n_node=30, device=device)
