#!/usr/bin/env python3
"""
Test to compare elitist vs non-elitist ACO.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_elitist_vs_non_elitist(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test elitist vs non-elitist ACO."""
    print(f"\n{'='*60}")
    print(f"Testing elitist vs non-elitist ACO (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test elitist
    print("--- Testing with elitist=True ---")
    aco_elitist = MFACO_BPP(
        demand=demand,
        capacity=capacity,
        n_ants=n_ants,
        decay=0.3,
        alpha=1.0,
        beta=1.0,
        elitist=True,
        device=device,
    )

    costs, paths, _, _ = aco_elitist.sample(require_prob=False, prior=None, parallel_traced=True)
    initial_best = float(torch.tensor(costs).min().item())
    print(f"  Initial sample: best={initial_best:.4f}")

    best_fitness = aco_elitist.run(10)
    print(f"  After 10 iterations: best_fitness={best_fitness:.4f}")

    costs_final, paths_final, _, _ = aco_elitist.sample(require_prob=False, prior=None, parallel_traced=True)
    final_best = float(torch.tensor(costs_final).min().item())
    print(f"  Final sample: best={final_best:.4f}")
    print(f"  Improvement: {initial_best - final_best:.4f}")

    # Test non-elitist
    print(f"\n--- Testing with elitist=False ---")
    aco_non_elitist = MFACO_BPP(
        demand=demand,
        capacity=capacity,
        n_ants=n_ants,
        decay=0.3,
        alpha=1.0,
        beta=1.0,
        elitist=False,
        device=device,
    )

    costs, paths, _, _ = aco_non_elitist.sample(require_prob=False, prior=None, parallel_traced=True)
    initial_best = float(torch.tensor(costs).min().item())
    print(f"  Initial sample: best={initial_best:.4f}")

    best_fitness = aco_non_elitist.run(10)
    print(f"  After 10 iterations: best_fitness={best_fitness:.4f}")

    costs_final, paths_final, _, _ = aco_non_elitist.sample(require_prob=False, prior=None, parallel_traced=True)
    final_best = float(torch.tensor(costs_final).min().item())
    print(f"  Final sample: best={final_best:.4f}")
    print(f"  Improvement: {initial_best - final_best:.4f}")

    print(f"\n{'='*60}")
    print("Analysis:")
    print("  - Compare elitist vs non-elitist performance")
    print("  - Check if elitist update is working correctly")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_elitist_vs_non_elitist(n_node=30, device=device)
