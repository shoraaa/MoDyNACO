#!/usr/bin/env python3
"""
Debug training to check why gradients are zero.
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

def debug_training(n_node=20, n_ants=20, capacity=150.0, device="cuda"):
    """Debug training to check why gradients are zero."""
    print(f"\n{'='*60}")
    print(f"Debugging training (n={n_node})")
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

    # Compute advantage
    baseline = costs.mean()
    adv = (baseline - costs).detach()

    print(f"\nAdvantage statistics:")
    print(f"  Mean: {adv.mean().item():.6f}")
    print(f"  Std: {adv.std().item():.6f}")
    print(f"  Min: {adv.min().item():.6f}")
    print(f"  Max: {adv.max().item():.6f}")

    # Compute log probabilities
    logp_steps = aco._replay_logp_from_paths(paths, prior, pheromone=aco.pheromone, heuristic=aco.heuristic)

    print(f"\nLog probability statistics:")
    print(f"  Shape: {logp_steps.shape}")
    if logp_steps.size(0) > 0:
        print(f"  Mean: {logp_steps.mean().item():.6f}")
        print(f"  Std: {logp_steps.std().item():.6f}")
        print(f"  Min: {logp_steps.min().item():.6f}")
        print(f"  Max: {logp_steps.max().item():.6f}")

        # Compute loss
        logp_new = logp_steps.sum(dim=0)
        ndec_f = torch.full((paths.size(1),), logp_steps.size(0), device=device, dtype=torch.float32).clamp_min(1.0)
        logp_new = logp_new / ndec_f

        loss = (logp_new * adv).mean()

        print(f"\nLoss statistics:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Logp mean: {logp_new.mean().item():.6f}")
        print(f"  Adv mean: {adv.mean().item():.6f}")

        # Check if loss requires grad
        print(f"\n  Loss requires grad: {loss.requires_grad}")
        print(f"  Prior requires grad: {prior.requires_grad}")

        # Compute gradients
        loss.backward()

        # Check gradient norms
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print(f"  Gradient norm: {grad_norm.item():.6f}")

        # Check individual parameter gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"  {name} grad norm: {param.grad.norm().item():.6f}")
            else:
                print(f"  {name} grad: None")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_training(n_node=20, device=device)
