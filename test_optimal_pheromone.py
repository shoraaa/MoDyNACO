#!/usr/bin/env python3
"""
Test to find optimal initial pheromone value.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_optimal_initial_pheromone(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test to find optimal initial pheromone value."""
    print(f"\n{'='*60}")
    print(f"Finding optimal initial pheromone (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test different initial pheromone values
    initial_pheromones = [1.0, 2.0, 5.0, 10.0, 20.0]

    best_improvement = -float('inf')
    best_init_phero = None

    for init_phero in initial_pheromones:
        print(f"\n--- Testing with initial pheromone={init_phero} ---")

        # Create ACO solver
        aco = MFACO_BPP(
            demand=demand,
            capacity=capacity,
            n_ants=n_ants,
            decay=0.3,
            alpha=1.0,
            beta=1.0,
            elitist=True,
            device=device,
        )

        # Set initial pheromone
        aco.pheromone.fill_(init_phero)
        aco._sync_cpp_inputs()

        # Sample initial solutions
        costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        initial_best = float(torch.tensor(costs).min().item())

        # Run ACO for 10 iterations
        best_fitness = aco.run(10)

        # Sample final solutions
        costs_final, paths_final, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        final_best = float(torch.tensor(costs_final).min().item())

        improvement = initial_best - final_best
        print(f"  Initial: {initial_best:.4f}, Final: {final_best:.4f}, Improvement: {improvement:+.4f}")

        if improvement > best_improvement:
            best_improvement = improvement
            best_init_phero = init_phero

    print(f"\n{'='*60}")
    print(f"Best initial pheromone: {best_init_phero} (improvement: {best_improvement:+.4f})")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_optimal_initial_pheromone(n_node=30, device=device)
