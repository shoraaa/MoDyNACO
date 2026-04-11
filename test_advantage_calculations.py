#!/usr/bin/env python3
"""
Test different advantage calculations to increase loss.
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

def test_advantage_calculations(n_node=20, n_ants=20, capacity=150.0, device="cuda"):
    """Test different advantage calculations."""
    print(f"\n{'='*60}")
    print(f"Testing advantage calculations (n={n_node})")
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
        decay=0.01,
        alpha=1.0,
        beta=1.0,
        elitist=True,
        device=device,
    )

    # Build PyG data
    from utils_extended import build_pyg_data_bpp
    pyg_data = build_pyg_data_bpp(demand, device, pheromone=aco.pheromone, dynamic=True)

    # Generate prior
    prior_output = model(pyg_data)
    prior = prior_output.view(aco.n + 1, aco.n + 1)

    # Sample solutions with prior
    costs, paths, logps, _ = aco.sample(require_prob=True, prior=prior, parallel_traced=True)

    print(f"Cost statistics:")
    print(f"  Mean: {costs.mean().item():.6f}")
    print(f"  Std: {costs.std().item():.6f}")
    print(f"  Min: {costs.min().item():.6f}")
    print(f"  Max: {costs.max().item():.6f}")

    # Compute log probabilities
    logp_steps = aco._replay_logp_from_paths(paths, prior, pheromone=aco.pheromone, heuristic=aco.heuristic)
    logp_new = logp_steps.sum(dim=0)
    ndec_f = torch.full((paths.size(1),), logp_steps.size(0), device=device, dtype=torch.float32).clamp_min(1.0)
    logp_new = logp_new / ndec_f

    print(f"\nLog probability statistics:")
    print(f"  Mean: {logp_new.mean().item():.6f}")
    print(f"  Std: {logp_new.std().item():.6f}")

    # Test different advantage calculations
    print(f"\n--- Testing different advantage calculations ---\n")

    # Method 1: baseline - costs (current)
    baseline = costs.mean()
    adv1 = (baseline - costs).detach()
    loss1 = (logp_new * adv1).mean()
    print(f"Method 1 (baseline - costs):")
    print(f"  Advantage mean: {adv1.mean().item():.6f}")
    print(f"  Loss: {loss1.item():.6f}")

    # Method 2: (baseline - costs) / std
    adv2 = ((baseline - costs) / costs.std()).detach()
    loss2 = (logp_new * adv2).mean()
    print(f"\nMethod 2 ((baseline - costs) / std):")
    print(f"  Advantage mean: {adv2.mean().item():.6f}")
    print(f"  Loss: {loss2.item():.6f}")

    # Method 3: (baseline - costs) * 10
    adv3 = ((baseline - costs) * 10).detach()
    loss3 = (logp_new * adv3).mean()
    print(f"\nMethod 3 ((baseline - costs) * 10):")
    print(f"  Advantage mean: {adv3.mean().item():.6f}")
    print(f"  Loss: {loss3.item():.6f}")

    # Method 4: -costs (no baseline)
    adv4 = (-costs).detach()
    loss4 = (logp_new * adv4).mean()
    print(f"\nMethod 4 (-costs):")
    print(f"  Advantage mean: {adv4.mean().item():.6f}")
    print(f"  Loss: {loss4.item():.6f}")

    # Method 5: (min_cost - costs)
    min_cost = costs.min()
    adv5 = (min_cost - costs).detach()
    loss5 = (logp_new * adv5).mean()
    print(f"\nMethod 5 (min_cost - costs):")
    print(f"  Advantage mean: {adv5.mean().item():.6f}")
    print(f"  Loss: {loss5.item():.6f}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_advantage_calculations(n_node=20, device=device)
