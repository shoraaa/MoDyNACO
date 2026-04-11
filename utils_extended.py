#!/usr/bin/env python3
"""
Data Generation Utilities for BPP, MKP, OP

Provides utilities for generating problem instances and building PyG data
for Bin Packing Problem (BPP), Multi-dimensional Knapsack Problem (MKP),
and Orienteering Problem (OP).
"""

import torch
from torch import Tensor
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
from typing import Any, Optional


# Data paths
_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = (_THIS_DIR / "data").resolve()
ALPHAANT_ROOT = Path("~/Research/AlphaAnt").expanduser()
ALPHAANT_DATA_DIR = ALPHAANT_ROOT / "data"


def _augment_edge_attr_with_dynamic_features(
    base_edge_attr: Tensor,
    aco: Any,
    dynamic: bool,
    device: str,
) -> Tensor:
    """Append normalized pheromone and incumbent channels to dense edge attributes."""
    base_edge_attr = base_edge_attr.to(device=device, dtype=torch.float32)
    edge_count = base_edge_attr.size(0)
    n_nodes = int(round(edge_count ** 0.5))

    if dynamic:
        # 1. Pheromone feature (log-relative normalization)
        tau = aco.pheromone.to(device=device, dtype=torch.float32)
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(1e-12)
        tau_rel = (tau / tau_mean).clamp_min(1e-12)
        pheromone_feat = torch.log(tau_rel).clamp(-5.0, 5.0).reshape(edge_count, 1)

        # 2. Incumbent feature (is_source_succ)
        sp = getattr(aco, 'shortest_path', None)
        if sp is not None and sp.numel() > 1:
            is_source_succ = torch.zeros((n_nodes, n_nodes), device=device)
            u_indices = sp[:-1]
            v_indices = sp[1:]
            # Ensure indices are within bounds (handles dummy nodes etc.)
            mask = (u_indices < n_nodes) & (v_indices < n_nodes)
            is_source_succ[u_indices[mask], v_indices[mask]] = 1.0
            incumbent_feat = is_source_succ.reshape(edge_count, 1)
        else:
            incumbent_feat = torch.zeros((edge_count, 1), device=device)
        
        return torch.cat((base_edge_attr, pheromone_feat, incumbent_feat), dim=1)
    else:
        # Static mode: just return the base edge attribute (1 channel)
        return base_edge_attr


# =============================================================================
# BPP (Bin Packing Problem)
# =============================================================================

DEMAND_LOW = 20
DEMAND_HIGH = 100
CAPACITY = 150


def gen_bpp_instance(n: int, device: torch.device) -> Tensor:
    """
    Generate a BPP instance.

    Args:
        n: Number of items
        device: Device to create tensors on

    Returns:
        demand: Tensor of shape (n+1,) with demand values (first element is 0 for depot)
    """
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH + 1, size=(n,), device=device)
    all_demands = torch.cat((torch.zeros((1,), device=device), demands))
    return all_demands  # (n+1)


def build_pyg_data_bpp(
    demand: Tensor,
    aco: Any,
    device: str = 'cpu',
    dynamic: bool = True,
) -> Data:
    """
    Build PyG data for BPP.

    Args:
        demand: Demand tensor of shape (n+1,)
        aco: ACO instance
        device: Device to create tensors on
        dynamic: Whether to include dynamic features

    Returns:
        pyg_data: PyG Data instance
    """
    demand = demand.to(device=device, dtype=torch.float32)
    n = demand.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    base_edge_attr = torch.ones((edge_index.size(1), 1), device=device)
    edge_attr = _augment_edge_attr_with_dynamic_features(base_edge_attr, aco, dynamic, device)
    x = demand.unsqueeze(1)  # (n+1, 1)
    pyg_data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data


def load_bpp_test_dataset(problem_size: int, device: torch.device):
    """
    Load BPP test dataset.
    """
    dataset_path = DATA_DIR / f"bpp_{problem_size}.pt"
    if not dataset_path.exists():
        # Generate on the fly if not exists
        instances = [gen_bpp_instance(problem_size, device) for _ in range(100)]
        return instances
    return torch.load(dataset_path, map_location=device)


# =============================================================================
# MKP (Multi-dimensional Knapsack Problem)
# =============================================================================

def gen_mkp_instance(n: int, m: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Generate an MKP instance.

    Args:
        n: Number of items
        m: Number of dimensions (constraints)
        device: Device to create tensors on

    Returns:
        prize: Prize tensor of shape (n,)
        weight: Weight tensor of shape (n, m)
    """
    prize = torch.rand(n, device=device)
    weight = torch.rand(n, m, device=device)
    return prize, weight


def build_pyg_data_mkp(
    prize: Tensor,
    weight: Tensor,
    aco: Any,
    device: str = 'cpu',
    dynamic: bool = True,
) -> Data:
    """
    Build PyG data for MKP.

    Args:
        prize: Prize tensor of shape (n,)
        weight: Weight tensor of shape (n, m)
        aco: ACO instance
        device: Device to create tensors on
        dynamic: Whether to include dynamic features

    Returns:
        pyg_data: PyG Data instance
    """
    prize = prize.to(device=device, dtype=torch.float32)
    weight = weight.to(device=device, dtype=torch.float32)
    n = prize.size(0)
    m = weight.size(1)
    
    # x: prize and weights for each item, plus a dummy node
    x = torch.zeros((n + 1, m + 1), device=device)
    x[:n, 0] = prize
    x[:n, 1:] = weight
    
    n_total = n + 1
    nodes = torch.arange(n_total, device=device)
    u = nodes.repeat(n_total)
    v = torch.repeat_interleave(nodes, n_total)
    edge_index = torch.stack((u, v))
    
    # Base edge attribute: 1 for edges from items to other items, 0 for depot
    base_edge_attr = torch.ones((edge_index.size(1), 1), device=device)
    edge_attr = _augment_edge_attr_with_dynamic_features(base_edge_attr, aco, dynamic, device)
    
    pyg_data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data


def load_mkp_test_dataset(problem_size: int, m: int, device: torch.device):
    """Load MKP test dataset."""
    dataset_path = DATA_DIR / f"mkp_{problem_size}_{m}.pt"
    if not dataset_path.exists():
        instances = [gen_mkp_instance(problem_size, m, device) for _ in range(100)]
        return instances
    return torch.load(dataset_path, map_location=device)


# =============================================================================
# OP (Orienteering Problem)
# =============================================================================

def gen_op_instance(n: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Generate an OP instance.

    Args:
        n: Number of items (including depot)
        device: Device to create tensors on

    Returns:
        coords: Coordinates of nodes shape (n, 2)
        prizes: Prize of nodes shape (n,)
    """
    coords = torch.rand(n, 2, device=device)
    prizes = torch.rand(n, device=device)
    prizes[0] = 0.0  # Depot prize is 0
    return coords, prizes


def build_pyg_data_op(
    distances: Tensor,
    prizes: Tensor,
    aco: Any,
    device: str = 'cpu',
    dynamic: bool = True,
) -> Data:
    """
    Build PyG data for OP.

    Args:
        distances: Distance matrix of shape (n, n)
        prizes: Prize tensor of shape (n,)
        aco: ACO instance
        device: Device to create tensors on
        dynamic: Whether to include dynamic features

    Returns:
        pyg_data: PyG Data instance
    """
    distances = distances.to(device=device, dtype=torch.float32)
    prizes = prizes.to(device=device, dtype=torch.float32)
    n = prizes.size(0)
    n_total = n + 1  # Including dummy node
    
    # x: prize and distance to depot
    x = torch.zeros((n_total, 2), device=device)
    x[:n, 0] = prizes
    x[:n, 1] = distances[0, :]
    
    nodes = torch.arange(n_total, device=device)
    u = nodes.repeat(n_total)
    v = torch.repeat_interleave(nodes, n_total)
    edge_index = torch.stack((u, v))
    
    # Base edge attribute: 1/distance
    dist_flat = torch.zeros((n_total * n_total, 1), device=device)
    dist_flat[:n*n, 0] = 1.0 / (distances.reshape(-1) + 1e-10)
    
    edge_attr = _augment_edge_attr_with_dynamic_features(dist_flat, aco, dynamic, device)
    
    pyg_data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data


def load_op_test_dataset(problem_size: int, device: torch.device):
    """Load OP test dataset."""
    dataset_path = DATA_DIR / f"op_{problem_size}.pt"
    if not dataset_path.exists():
        instances = [gen_op_instance(problem_size, device) for _ in range(100)]
        return instances
    return torch.load(dataset_path, map_location=device)


# =============================================================================
# General Utilities
# =============================================================================

def get_problem_data(problem: str, n: int, device: str, k_sparse: int = 32):
    """Generate or load problem data in the format expected by MFACO classes."""
    if problem == 'bpp':
        demand = gen_bpp_instance(n, torch.device(device))
        return {'demand': demand}
    elif problem == 'mkp':
        prize, weight = gen_mkp_instance(n, 5, torch.device(device))
        return {'prize': prize, 'weight': weight}
    elif problem == 'op':
        coords, prizes = gen_op_instance(n, torch.device(device))
        # Calculate distance matrix
        distances = torch.cdist(coords, coords)
        return {'distances': distances, 'prizes': prizes}
    else:
        raise ValueError(f"Unknown problem: {problem}")


if __name__ == "__main__":
    # Test instance generation
    print("Testing BPP instance generation...")
    demand = gen_bpp_instance(20, torch.device('cpu'))
    print(f"BPP demand: {demand.shape}")
    # We need a mock aco for testing build_pyg_data
    class MockACO:
        def __init__(self, n):
            self.pheromone = torch.ones((n, n))
            self.shortest_path = None
    
    pyg_data = build_pyg_data_bpp(demand, MockACO(21), 'cpu')
    print(f"PyG data: {pyg_data}")

    print("\nTesting MKP instance generation...")
    prize, weight = gen_mkp_instance(20, 5, torch.device('cpu'))
    print(f"MKP prize: {prize.shape}, weight: {weight.shape}")
    pyg_data = build_pyg_data_mkp(prize, weight, MockACO(21), 'cpu')
    print(f"PyG data: {pyg_data}")

    print("\nTesting OP instance generation...")
    coords, prizes = gen_op_instance(20, torch.device('cpu'))
    distances = torch.cdist(coords, coords)
    print(f"OP coords: {coords.shape}, prizes: {prizes.shape}")
    pyg_data = build_pyg_data_op(distances, prizes, MockACO(21), 'cpu')
    print(f"PyG data: {pyg_data}")

    print("\nAll tests passed!")
