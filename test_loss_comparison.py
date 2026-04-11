#!/usr/bin/env python3
"""
Test to compare loss calculation with original DeepACO.
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

def test_loss_calculation(n_node=20, n_ants=20, capacity=150.0, device="cuda"):
    """Test loss calculation comparison."""
    print(f"\n{'='*60}")
    print(f"Testing loss calculation (n={n_node})")
    print(f"{'='*60}\n")

    # Generate a single instance
    demand = gen_bpp_instance(n_node, device)
    print(f"Generated BPP instance with {n_node} items")
    print(f"Capacity: {capacity}\n")

    # Create model
    model = NetBPP(feats=1, edge_feats=2).to(device)
    model.train()

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

    # Generate prior
    prior_output = model(pyg_data)
    prior = prior_output.view(aco.n + 1, aco.n + 1)

    print(f"Prior statistics:")
    print(f"  Mean: {prior.mean().item():.6f}")
    print(f"  Std: {prior.std().item():.6f}")
    print(f"  Min: {prior.min().item():.6f}")
    print(f"  Max: {prior.max().item():.6f}")

    # Sample solutions with prior
    costs, paths, logps, _ = aco.sample(require_prob=True, prior=prior, parallel_traced=True)

    print(f"\nCost statistics:")
    print(f"  Mean: {costs.mean().item():.6f}")
    print(f"  Std: {costs.std().item():.6f}")
    print(f"  Min: {costs.min().item():.6f}")
    print(f"  Max: {costs.max().item():.6f}")

    # Compute log probabilities
    logp_steps = aco._replay_logp_from_paths(paths, prior, pheromone=aco.pheromone, heuristic=aco.heuristic)

    print(f"\nLog probability statistics:")
    print(f"  Shape: {logp_steps.shape}")
    if logp_steps.size(0) > 0:
        print(f"  Mean: {logp_steps.mean().item():.6f}")
        print(f"  Std: {logp_steps.std().item():.6f}")

        # Test different loss calculations
        logp_new = logp_steps.sum(dim=0)
        ndec_f = torch.full((paths.size(1),), logp_steps.size(0), device=device, dtype=torch.float32).clamp_min(1.0)
        logp_new = logp_new / ndec_f

        print(f"\nLog probability (normalized):")
        print(f"  Mean: {logp_new.mean().item():.6f}")
        print(f"  Std: {logp_new.std().item():.6f}")

        # Method 1: DeepACO's loss calculation
        baseline = costs.mean()
        adv1 = (costs - baseline).detach()
        loss1 = -torch.sum(adv1 * logp_new) / aco.n_ants
        print(f"\nMethod 1 (DeepACO):")
        print(f"  Baseline: {baseline.item():.6f}")
        print(f"  Advantage mean: {adv1.mean().item():.6f}")
        print(f"  Loss: {loss1.item():.6f}")

        # Method 2: Current NGFACO loss calculation
        adv2 = (costs - baseline).detach()
        loss2 = -(logp_new * adv2).mean()
        print(f"\nMethod 2 (Current NGFACO):")
        print(f"  Baseline: {baseline.item():.6f}")
        print(f"  Advantage mean: {adv2.mean().item():.6f}")
        print(f"  Loss: {loss2.item():.6f}")

        # Method 3: Without normalization
        logp_sum = logp_steps.sum(dim=0)
        loss3 = -torch.sum((costs - baseline).detach() * logp_sum) / aco.n_ants
        print(f"\nMethod 3 (Without normalization):")
        print(f"  Loss: {loss3.item():.6f}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loss_calculation(n_node=20, device=device)
