#!/usr/bin/env python3
"""
Test to check if C++ run method respects the prior.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_prior_in_run(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test if C++ run method respects the prior."""
    print(f"\n{'='*60}")
    print(f"Testing if C++ run method respects prior (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Test 1: Without prior
    print("--- Test 1: Without prior ---")
    aco_no_prior = MFACO_BPP(
        demand=demand,
        capacity=capacity,
        n_ants=n_ants,
        decay=0.3,
        alpha=1.0,
        beta=1.0,
        elitist=True,
        device=device,
    )

    costs, paths, _, _ = aco_no_prior.sample(require_prob=False, prior=None, parallel_traced=True)
    initial_best = float(torch.tensor(costs).min().item())
    print(f"  Initial sample (no prior): best={initial_best:.2f}")

    best_fitness = aco_no_prior.run(10)
    print(f"  After 10 iterations (no prior): best_fitness={best_fitness:.2f}")

    costs_final, paths_final, _, _ = aco_no_prior.sample(require_prob=False, prior=None, parallel_traced=True)
    final_best = float(torch.tensor(costs_final).min().item())
    print(f"  Final sample (no prior): best={final_best:.2f}")

    # Test 2: With random prior
    print(f"\n--- Test 2: With random prior ---")
    aco_with_prior = MFACO_BPP(
        demand=demand,
        capacity=capacity,
        n_ants=n_ants,
        decay=0.3,
        alpha=1.0,
        beta=1.0,
        elitist=True,
        device=device,
    )

    # Create a random prior
    random_prior = torch.randn(n_node + 1, n_node + 1, device=device) * 0.1

    costs, paths, _, _ = aco_with_prior.sample(require_prob=False, prior=random_prior, parallel_traced=True)
    initial_best = float(torch.tensor(costs).min().item())
    print(f"  Initial sample (with random prior): best={initial_best:.2f}")

    # Check if prior is being used
    print(f"  Prior mean: {random_prior.mean().item():.6f}, std: {random_prior.std().item():.6f}")

    best_fitness = aco_with_prior.run(10)
    print(f"  After 10 iterations (with random prior): best_fitness={best_fitness:.2f}")

    costs_final, paths_final, _, _ = aco_with_prior.sample(require_prob=False, prior=random_prior, parallel_traced=True)
    final_best = float(torch.tensor(costs_final).min().item())
    print(f"  Final sample (with random prior): best={final_best:.2f}")

    print(f"\n{'='*60}")
    print("Analysis:")
    print("  - Check if C++ run method uses the prior")
    print("  - If not, the prior is only used during sampling")
    print("  - This could explain why training doesn't improve performance")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_prior_in_run(n_node=30, device=device)
