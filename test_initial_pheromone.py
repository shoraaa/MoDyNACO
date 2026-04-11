#!/usr/bin/env python3
"""
Test to check if initial pheromone value affects ACO performance.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_initial_pheromone(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test different initial pheromone values."""
    print(f"\n{'='*60}")
    print(f"Testing initial pheromone values (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test different initial pheromone values
    initial_pheromones = [0.1, 0.5, 1.0, 2.0]

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
        print(f"  Initial sample: best={initial_best:.4f}")

        # Run ACO for 10 iterations
        best_fitness = aco.run(10)
        print(f"  After 10 iterations: best_fitness={best_fitness:.4f}")

        # Sample final solutions
        costs_final, paths_final, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
        final_best = float(torch.tensor(costs_final).min().item())
        print(f"  Final sample: best={final_best:.4f}")
        print(f"  Improvement: {initial_best - final_best:.4f}")

        # Check pheromone statistics
        pheromone = aco.pheromone
        print(f"  Pheromone: mean={pheromone.mean().item():.6f}, std={pheromone.std().item():.6f}")

    print(f"\n{'='*60}")
    print("Summary:")
    print("  - Check if initial pheromone affects performance")
    print("  - Lower initial pheromone might help deposits have more impact")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_initial_pheromone(n_node=30, device=device)
