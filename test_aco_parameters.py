#!/usr/bin/env python3
"""
Diagnostic test to find optimal ACO parameters for BPP.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_aco_parameters(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test ACO with different parameter combinations."""
    print(f"\n{'='*60}")
    print(f"Testing ACO parameters (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test different parameter combinations
    test_configs = [
        # (rho, alpha, beta, n_iterations, elitist)
        (0.3, 1.0, 1.0, 10, True),
        (0.3, 1.0, 1.0, 20, True),
        (0.3, 1.0, 1.0, 50, True),
        (0.1, 1.0, 1.0, 20, True),
        (0.3, 2.0, 1.0, 20, True),
        (0.3, 1.0, 2.0, 20, True),
        (0.3, 1.0, 1.0, 20, False),
    ]

    for rho, alpha, beta, n_iters, elitist in test_configs:
        print(f"\n--- Testing: rho={rho}, alpha={alpha}, beta={beta}, iters={n_iters}, elitist={elitist} ---")

        # Create ACO solver
        aco = MFACO_BPP(
            demand=demand,
            capacity=capacity,
            n_ants=n_ants,
            decay=rho,
            alpha=alpha,
            beta=beta,
            elitist=elitist,
            device=device,
        )

        # Sample initial solutions
        costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        initial_best = float(torch.tensor(costs).min().item())
        initial_mean = float(torch.tensor(costs).mean().item())

        print(f"  Initial sample: best={initial_best:.2f}, mean={initial_mean:.2f}")

        # Run ACO for n_iters iterations
        best_fitness = aco.run(n_iters)

        # Sample final solutions
        costs_final, paths_final, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        final_best = float(torch.tensor(costs_final).min().item())
        final_mean = float(torch.tensor(costs_final).mean().item())

        improvement = initial_best - final_best
        print(f"  After {n_iters} iterations: best={final_best:.2f}, mean={final_mean:.2f}")
        print(f"  Improvement: {improvement:+.2f} bins")

        # Check pheromone statistics
        pheromone = aco.pheromone
        print(f"  Pheromone: mean={pheromone.mean().item():.6f}, std={pheromone.std().item():.6f}")

    print(f"\n{'='*60}")
    print("Summary:")
    print("  - Find the best parameter combination")
    print("  - Use these parameters for training")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_aco_parameters(n_node=30, device=device)
