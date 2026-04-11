#!/usr/bin/env python3
"""
Test to compare C++ ACO vs Python ACO performance.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_aco_comparison(n_node=20, n_ants=20, capacity=150.0, device="cuda"):
    """Test C++ ACO vs Python ACO performance."""
    print(f"\n{'='*60}")
    print(f"Testing C++ ACO vs Python ACO (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test C++ ACO
    print("--- Testing C++ ACO ---")
    aco_cpp = MFACO_BPP(
        demand=demand,
        capacity=capacity,
        n_ants=n_ants,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        elitist=False,
        device=device,
    )

    # Sample initial solutions
    costs_cpp, paths_cpp, _, _ = aco_cpp.sample(require_prob=False, prior=None, parallel_traced=True)
    initial_best_cpp = float(torch.tensor(costs_cpp).min().item())
    print(f"  Initial sample: best={initial_best_cpp:.6f}")

    # Run ACO for 10 iterations
    best_fitness_cpp = aco_cpp.run(10)

    # Sample final solutions
    costs_final_cpp, paths_final_cpp, _, _ = aco_cpp.sample(require_prob=False, prior=None, parallel_traced=True)
    final_best_cpp = float(torch.tensor(costs_final_cpp).min().item())
    print(f"  After 10 iterations: best={final_best_cpp:.6f}")
    print(f"  Improvement: {initial_best_cpp - final_best_cpp:.6f}")

    # Check pheromone statistics
    pheromone_cpp = aco_cpp.pheromone
    print(f"  Pheromone: mean={pheromone_cpp.mean().item():.6f}, std={pheromone_cpp.std().item():.6f}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_aco_comparison(n_node=20, device=device)
