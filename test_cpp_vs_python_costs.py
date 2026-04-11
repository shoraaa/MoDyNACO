#!/usr/bin/env python3
"""
Test to verify the C++ fitness calculation vs Python bin count.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_cpp_vs_python_costs(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test C++ fitness vs Python bin count."""
    print(f"\n{'='*60}")
    print(f"Testing C++ fitness vs Python bin count (n={n_node})")
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
        decay=0.3,
        alpha=1.0,
        beta=1.0,
        elitist=True,
        device=device,
    )

    # Sample solutions
    costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)

    print(f"Sampled {n_ants} solutions:\n")

    for ant_idx in range(min(5, n_ants)):  # Show first 5 ants
        ant_path = paths[:, ant_idx]
        ant_cost_cpp = costs[ant_idx].item()

        # Compute Python bin count
        seq_len = paths.size(0)
        problem_size = n_node
        trailing_zeros = 0
        for step in range(seq_len - 1, -1, -1):
            if ant_path[step] == 0:
                trailing_zeros += 1
            else:
                break
        n_bins_python = max(1, seq_len - trailing_zeros - problem_size + 1)

        print(f"Ant {ant_idx + 1}:")
        print(f"  C++ cost (negative fitness): {ant_cost_cpp:.6f}")
        print(f"  Python bin count: {n_bins_python:.2f}")
        print(f"  Path: {ant_path[:10].tolist()}... (showing first 10 steps)")

    # Check correlation
    print(f"\n--- Correlation Analysis ---")
    costs_np = costs.cpu().numpy()

    # Compute Python bin counts for all ants
    n_bins_list = []
    for ant_idx in range(n_ants):
        ant_path = paths[:, ant_idx]
        trailing_zeros = 0
        for step in range(seq_len - 1, -1, -1):
            if ant_path[step] == 0:
                trailing_zeros += 1
            else:
                break
        n_bins = max(1, seq_len - trailing_zeros - problem_size + 1)
        n_bins_list.append(n_bins)

    n_bins_np = np.array(n_bins_list)

    # Find best according to C++ (most negative = best)
    best_cpp_idx = int(np.argmin(costs_np))
    best_cpp_cost = costs_np[best_cpp_idx]
    best_cpp_bins = n_bins_np[best_cpp_idx]

    # Find best according to Python (fewest bins = best)
    best_python_idx = int(np.argmin(n_bins_np))
    best_python_bins = n_bins_np[best_python_idx]
    best_python_cost = costs_np[best_python_idx]

    print(f"Best according to C++ (most negative fitness):")
    print(f"  Ant {best_cpp_idx + 1}: cost={best_cpp_cost:.6f}, bins={best_cpp_bins:.2f}")
    print(f"\nBest according to Python (fewest bins):")
    print(f"  Ant {best_python_idx + 1}: cost={best_python_cost:.6f}, bins={best_python_bins:.2f}")

    # Check if they agree
    if best_cpp_idx == best_python_idx:
        print(f"\n✓ C++ and Python agree on best solution!")
    else:
        print(f"\n✗ C++ and Python DISAGREE on best solution!")
        print(f"  C++ prefers ant {best_cpp_idx + 1} with {best_cpp_bins:.2f} bins")
        print(f"  Python prefers ant {best_python_idx + 1} with {best_python_bins:.2f} bins")

    print(f"\n{'='*60}")
    print("Conclusion:")
    print("  - C++ optimizes for fitness (squared capacity utilization)")
    print("  - Python optimizes for bin count")
    print("  - These are different objectives!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_cpp_vs_python_costs(n_node=30, device=device)
