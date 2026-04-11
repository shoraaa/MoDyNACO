#!/usr/bin/env python3
"""
Test to understand the C++ fitness calculation in detail.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_cpp_fitness_calculation(n_node=30, n_ants=20, capacity=150.0, device="cuda"):
    """Test C++ fitness calculation in detail."""
    print(f"\n{'='*60}")
    print(f"Testing C++ fitness calculation (n={n_node})")
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

    # Analyze one solution in detail
    ant_idx = 0
    ant_path = paths[:, ant_idx]
    ant_cost_cpp = costs[ant_idx].item()

    print(f"Analyzing Ant {ant_idx + 1}:\n")
    print(f"  C++ cost: {ant_cost_cpp:.6f}")
    print(f"  Path (first 20 steps): {ant_path[:20].tolist()}")

    # Compute bin count
    seq_len = paths.size(0)
    problem_size = n_node
    trailing_zeros = 0
    for step in range(seq_len - 1, -1, -1):
        if ant_path[step] == 0:
            trailing_zeros += 1
        else:
            break
    n_bins = max(1, seq_len - trailing_zeros - problem_size + 1)

    print(f"  Sequence length: {seq_len}")
    print(f"  Trailing zeros: {trailing_zeros}")
    print(f"  Problem size: {problem_size}")
    print(f"  Bin count: {n_bins}")

    # Compute fitness manually
    print(f"\n  Computing fitness manually:")
    fitness = 0.0
    sub_fitness = 0.0
    current_bin_items = []

    for step in range(1, seq_len):
        node = ant_path[step].item()
        if node != 0:
            node_demand = demand[node].item()
            sub_fitness += node_demand
            current_bin_items.append((int(node), float(node_demand)))
        else:
            # End of bin
            if sub_fitness > 0:
                bin_fitness = (sub_fitness / capacity) ** 2
                fitness += bin_fitness
                print(f"    Bin {len(current_bin_items)}: items={current_bin_items}, "
                      f"total_demand={sub_fitness:.1f}, utilization={sub_fitness/capacity:.2%}, "
                      f"fitness={bin_fitness:.4f}")
                current_bin_items = []
                sub_fitness = 0.0

    print(f"  Total fitness: {fitness:.6f}")
    print(f"  Expected C++ cost: {-fitness / max(n_bins, 1):.6f}")

    # Check if C++ cost matches expected
    expected_cost = -fitness / max(n_bins, 1)
    if abs(ant_cost_cpp - expected_cost) < 0.01:
        print(f"\n  ✓ C++ cost matches expected fitness calculation!")
    else:
        print(f"\n  ✗ C++ cost does NOT match expected fitness calculation!")
        print(f"    C++ cost: {ant_cost_cpp:.6f}")
        print(f"    Expected: {expected_cost:.6f}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_cpp_fitness_calculation(n_node=30, device=device)
