#!/usr/bin/env python3
"""
Test to check if neural prior affects ACO sampling.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from faco_extended import MFACO_BPP
from utils_extended import gen_bpp_instance
from net_extended import NetBPP

def test_prior_effect(n_node=20, n_ants=20, capacity=150.0, device="cuda"):
    """Test if neural prior affects ACO sampling."""
    print(f"\n{'='*60}")
    print(f"Testing neural prior effect (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Create model
    model = NetBPP(feats=1, edge_feats=2).to(device)
    model.eval()

    # Create ACO solver
    aco = MFACO_BPP(
        demand=demand,
        capacity=capacity,
        n_ants=n_ants,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        elitist=False,
        device=device,
    )

    # Build PyG data
    from utils_extended import build_pyg_data_bpp
    pyg_data = build_pyg_data_bpp(demand, device)

    # Generate neural prior
    with torch.no_grad():
        prior_output = model(pyg_data)
        prior = prior_output.view(aco.n + 1, aco.n + 1)

    print(f"Prior statistics:")
    print(f"  Mean: {prior.mean().item():.6f}")
    print(f"  Std: {prior.std().item():.6f}")
    print(f"  Min: {prior.min().item():.6f}")
    print(f"  Max: {prior.max().item():.6f}")

    # Test 1: Without prior
    print(f"\n--- Test 1: Without prior ---")
    costs_no_prior, paths_no_prior, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
    best_no_prior = float(torch.tensor(costs_no_prior).min().item())
    mean_no_prior = float(torch.tensor(costs_no_prior).mean().item())
    print(f"  Best: {best_no_prior:.6f}, Mean: {mean_no_prior:.6f}")

    # Test 2: With neural prior
    print(f"\n--- Test 2: With neural prior ---")
    costs_with_prior, paths_with_prior, _, _ = aco.sample(require_prob=False, prior=prior, parallel_traced=True)
    best_with_prior = float(torch.tensor(costs_with_prior).min().item())
    mean_with_prior = float(torch.tensor(costs_with_prior).mean().item())
    print(f"  Best: {best_with_prior:.6f}, Mean: {mean_with_prior:.6f}")

    # Test 3: With random prior
    print(f"\n--- Test 3: With random prior ---")
    random_prior = torch.randn(aco.n + 1, aco.n + 1, device=device) * 0.1
    costs_random_prior, paths_random_prior, _, _ = aco.sample(require_prob=False, prior=random_prior, parallel_traced=True)
    best_random_prior = float(torch.tensor(costs_random_prior).min().item())
    mean_random_prior = float(torch.tensor(costs_random_prior).mean().item())
    print(f"  Best: {best_random_prior:.6f}, Mean: {mean_random_prior:.6f}")

    # Compare results
    print(f"\n--- Comparison ---")
    print(f"  Neural prior improvement: {best_no_prior - best_with_prior:.6f}")
    print(f"  Random prior improvement: {best_no_prior - best_random_prior:.6f}")

    if best_with_prior < best_no_prior:
        print(f"  ✓ Neural prior improves results!")
    else:
        print(f"  ✗ Neural prior does NOT improve results")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_prior_effect(n_node=20, device=device)
