#!/usr/bin/env python3
"""
Test to analyze correlation between fitness and bin count.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance

def test_fitness_bin_correlation(n_node=30, n_ants=100, capacity=150.0, device="cuda"):
    """Test correlation between fitness and bin count."""
    print(f"\n{'='*60}")
    print(f"Testing fitness vs bin count correlation (n={n_node})")
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

    # Sample many solutions
    costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)

    # Compute bin counts for all ants
    seq_len = paths.size(0)
    problem_size = n_node
    n_bins_list = []
    for ant_idx in range(paths.size(1)):
        ant_path = paths[:, ant_idx]
        trailing_zeros = 0
        for step in range(seq_len - 1, -1, -1):
            if ant_path[step] == 0:
                trailing_zeros += 1
            else:
                break
        n_bins = max(1, seq_len - trailing_zeros - problem_size + 1)
        n_bins_list.append(n_bins)

    costs_np = costs.cpu().numpy()
    n_bins_np = np.array(n_bins_list)

    # Analyze correlation
    print(f"Analyzed {n_ants} solutions:\n")

    # Group by bin count
    unique_bins = np.unique(n_bins_np)
    print(f"Bin count distribution:")
    for bins in unique_bins:
        mask = n_bins_np == bins
        avg_fitness = -costs_np[mask].mean()  # Convert back to positive fitness
        print(f"  {bins} bins: {mask.sum()} solutions, avg fitness = {avg_fitness:.4f}")

    # Compute correlation
    correlation = np.corrcoef(-costs_np, n_bins_np)[0, 1]
    print(f"\nCorrelation between fitness and bin count: {correlation:.4f}")

    # Check if higher fitness correlates with fewer bins
    if correlation < -0.5:
        print(f"✓ Strong negative correlation: higher fitness → fewer bins")
    elif correlation < -0.2:
        print(f"⚠ Moderate negative correlation: higher fitness → fewer bins")
    elif correlation > 0.2:
        print(f"✗ Positive correlation: higher fitness → MORE bins (BAD!)")
    else:
        print(f"⚠ Weak correlation: fitness and bin count are not well aligned")

    # Find best solutions by each metric
    best_fitness_idx = int(np.argmin(costs_np))  # Most negative = best fitness
    best_bins_idx = int(np.argmin(n_bins_np))  # Fewest bins

    print(f"\nBest by fitness (most negative cost):")
    print(f"  Ant {best_fitness_idx + 1}: cost={costs_np[best_fitness_idx]:.4f}, bins={n_bins_np[best_fitness_idx]:.0f}")

    print(f"\nBest by bin count (fewest bins):")
    print(f"  Ant {best_bins_idx + 1}: cost={costs_np[best_bins_idx]:.4f}, bins={n_bins_np[best_bins_idx]:.0f}")

    if best_fitness_idx == best_bins_idx:
        print(f"\n✓ Both metrics agree on best solution!")
    else:
        print(f"\n✗ Metrics disagree on best solution!")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_fitness_bin_correlation(n_node=30, n_ants=100, device=device)
