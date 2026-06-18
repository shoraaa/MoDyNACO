#!/usr/bin/env python3
"""
Training script for Neural-Guided Fast ACO (NGFACO).

This module implements PPO and REINFORCE training for learning neural priors
that guide ant colony optimization for TSP and CVRP problems.
"""

import time
import torch
import os
import psutil
import argparse
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional, Dict, List, Any, Tuple, Union, NamedTuple
import gc
import yaml

# Shared helpers
import wandb
import functools

# Unified modules
import net
import net as dynaco_net
import faco
import utils
import baselines

# Specific class imports
from net import MultiHeadNet, Net
from baselines import get_baseline

# Import from utils
from utils import (
    row_softmax, mean_row_kl, rel_l2_drift, top_set, top_turnover,
    top1_flip_rate, safe_corr, top_overlap_frac, row_top1_match_rate, EPS,
    Logger, MetricsCollector, get_logger, init_logger
)
from extended_common import (
    add_extended_problem_args,
    align_edge_attr_width as align_extended_edge_attr_width,
    build_model_name as build_extended_model_name,
    build_pyg_data as build_extended_pyg_data,
    compute_relative_improvement as compute_extended_relative_improvement,
    create_model as create_extended_model,
    extract_problem_data as extract_extended_problem_data,
    get_model_edge_feats as get_extended_model_edge_feats,
    prior_kwargs as get_extended_prior_kwargs,
    raw_values_to_objective as extended_raw_values_to_objective,
    reshape_prior_output as reshape_extended_prior_output,
    select_best_value as select_extended_best_value,
    setup_aco as setup_extended_aco,
    use_dynamic_edge_features as use_extended_dynamic_edge_features,
)

BASE_PROBLEMS = {"tsp", "cvrp"}
EXTENDED_PROBLEMS = {"bpp", "mkp", "op"}
_REPLAY_MAGIC_MASK_CACHE: Dict[Tuple[str, int, int], torch.Tensor] = {}


def _replay_magic_mask(k: int, device: torch.device) -> torch.Tensor:
    key = (device.type, device.index if device.index is not None else -1, int(k))
    mask = _REPLAY_MAGIC_MASK_CACHE.get(key)
    if mask is None or mask.device != device:
        mask = (1 << torch.arange(k, device=device, dtype=torch.int64))
        _REPLAY_MAGIC_MASK_CACHE[key] = mask
    return mask


def _head_ant_range(head_counts: List[int], head_idx: int) -> Tuple[int, int]:
    start = int(sum(int(c) for c in head_counts[:head_idx]))
    end = start + int(head_counts[head_idx])
    return start, end


def save_experiment_config(args, exp_dir: Path):
    """Save experiment configuration to YAML with git metadata."""
    import yaml
    import subprocess

    config_dict = _serializable_args(args)

    # Add git information
    try:
        config_dict['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        config_dict['git_diff_stat'] = subprocess.check_output(
            ['git', 'diff', '--stat'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        config_dict['git_commit'] = 'unknown'
        config_dict['git_diff_stat'] = ''

    # Add timestamp
    from datetime import datetime
    config_dict['timestamp'] = datetime.now().isoformat()

    config_path = exp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)

    print(f"[Experiment] Saved config to {config_path}")


def _serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            config[key] = value
        else:
            config[key] = str(value)
    return config


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _popcount_i64(x: torch.Tensor) -> torch.Tensor:
    """Compute population count (number of set bits) for int64 tensors."""
    if x.dtype != torch.int64:
        raise TypeError(f"_popcount_i64 expects torch.int64, got {x.dtype}")
    m1  = x.new_tensor(0x5555555555555555, dtype=torch.int64)
    m2  = x.new_tensor(0x3333333333333333, dtype=torch.int64)
    m4  = x.new_tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64)
    h01 = x.new_tensor(0x0101010101010101, dtype=torch.int64)
    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    return (x * h01) >> 56


def replay_logp_from_cpp_batch_trace(
    traces,
    log_prob_sparse: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replay traces to compute log probabilities using log-space arithmetic.
    
    Args:
        traces: Trace object from C++ ACO containing decision history
        log_prob_sparse: (n, k) tensor of UNNORMALIZED log-probabilities
    
    Returns:
        Tuple of (logp_sum, ndec) where:
            - logp_sum: Sum of log probabilities per ant
            - ndec: Number of decisions per ant
    """
    device = log_prob_sparse.device
    k = int(log_prob_sparse.size(1))
    
    # Trace arrays -> torch
    curr = torch.as_tensor(
        np.asarray(traces.curr_nodes, dtype=np.int64), device=device
    )
    is_stoch = torch.as_tensor(
        np.asarray(traces.is_stochastic, dtype=np.uint8), device=device
    ).bool()
    pick = torch.as_tensor(
        np.asarray(traces.pick_j, dtype=np.int64), device=device
    )
    
    # Mask might be 64-bit
    vm_arr = np.asarray(traces.valid_mask, dtype=np.uint64)
    vm_i64 = torch.as_tensor(vm_arr, device=device).to(torch.int64)

    starts_t = torch.as_tensor(
        np.asarray(traces.starts, dtype=np.int64),
        device=device,
        dtype=torch.int64
    )
    n_ants = int(getattr(traces, "n_ants", int(starts_t.numel() - 1)))
    counts_t = (starts_t[1:1+n_ants] - starts_t[:n_ants]).to(torch.int64)
    
    # Map trace steps to ant index
    ant_idx_all = torch.repeat_interleave(
        torch.arange(n_ants, device=device, dtype=torch.int64),
        counts_t,
    )

    ndec = torch.bincount(ant_idx_all[is_stoch], minlength=n_ants).to(torch.int32)
    logp_sum = torch.zeros((n_ants,), device=device, dtype=torch.float32)

    roulette = is_stoch & (pick >= 0)
    idx = roulette.nonzero(as_tuple=False).squeeze(1)
    
    if idx.numel() > 0:
        curr_r = curr[idx]   # current node
        pick_r = pick[idx]   # chosen neighbor index in sparse list
        vm_r = vm_i64[idx]   # valid mask
        
        # log weights for the current node's candidates: (batch, k)
        log_w = log_prob_sparse[curr_r]
        
        # Determine valid mask as boolean
        magic_mask = _replay_magic_mask(k, device)
        valid_bits = (vm_r.unsqueeze(1) & magic_mask) != 0
        
        # log_mask: 0.0 if valid, -inf otherwise
        log_mask_val = torch.empty_like(log_w).fill_(float('-inf'))
        log_mask_val.masked_fill_(valid_bits, 0.0)
        
        log_w_valid = log_w + log_mask_val
        
        # Log-Sum-Exp for denominator
        log_denom = torch.logsumexp(log_w_valid, dim=1)
        
        # Numerator is just the log_w of the picked choice
        log_numer = log_w.gather(1, pick_r.unsqueeze(1)).squeeze(1)
        
        # log p = log_numer - log_denom
        step_logp = log_numer - log_denom
        
        logp_sum.scatter_add_(0, ant_idx_all[idx], step_logp)

    return logp_sum, ndec


def replay_logp_from_cpp_batch_trace_ant_priors(
    traces,
    log_prob_ant_sparse: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replay C++ traces when each ant has its own sparse log-weight matrix."""
    device = log_prob_ant_sparse.device
    k = int(log_prob_ant_sparse.size(2))
    curr = torch.as_tensor(
        np.asarray(traces.curr_nodes, dtype=np.int64), device=device
    )
    is_stoch = torch.as_tensor(
        np.asarray(traces.is_stochastic, dtype=np.uint8), device=device
    ).bool()
    pick = torch.as_tensor(
        np.asarray(traces.pick_j, dtype=np.int64), device=device
    )
    vm_arr = np.asarray(traces.valid_mask, dtype=np.uint64)
    vm_i64 = torch.as_tensor(vm_arr, device=device).to(torch.int64)
    starts_t = torch.as_tensor(
        np.asarray(traces.starts, dtype=np.int64),
        device=device,
        dtype=torch.int64,
    )
    n_ants = int(getattr(traces, "n_ants", int(starts_t.numel() - 1)))
    counts_t = (starts_t[1:1 + n_ants] - starts_t[:n_ants]).to(torch.int64)
    ant_idx_all = torch.repeat_interleave(
        torch.arange(n_ants, device=device, dtype=torch.int64),
        counts_t,
    )
    ndec = torch.bincount(ant_idx_all[is_stoch], minlength=n_ants).to(torch.int32)
    logp_sum = torch.zeros((n_ants,), device=device, dtype=torch.float32)
    roulette = is_stoch & (pick >= 0)
    idx = roulette.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() > 0:
        ants_r = ant_idx_all[idx]
        curr_r = curr[idx]
        pick_r = pick[idx]
        vm_r = vm_i64[idx]
        log_w = log_prob_ant_sparse[ants_r, curr_r]
        magic_mask = _replay_magic_mask(k, device)
        valid_bits = (vm_r.unsqueeze(1) & magic_mask) != 0
        log_mask_val = torch.empty_like(log_w).fill_(float('-inf'))
        log_mask_val.masked_fill_(valid_bits, 0.0)
        log_w_valid = log_w + log_mask_val
        log_denom = torch.logsumexp(log_w_valid, dim=1)
        log_numer = log_w.gather(1, pick_r.unsqueeze(1)).squeeze(1)
        step_logp = log_numer - log_denom
        logp_sum.scatter_add_(0, ants_r, step_logp)
    return logp_sum, ndec


def replay_logp_from_cpp_batch_trace_ant_slice(
    traces,
    log_prob_ant_sparse: torch.Tensor,
    ant_start: int,
    ant_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replay a contiguous ant block from C++ traces.

    Multi-head routing assigns each head a contiguous ant block. PolyNet only
    updates the winning head, so replaying just that block avoids constructing
    and reducing log-probabilities for heads that will not receive gradients.
    """
    device = log_prob_ant_sparse.device
    k = int(log_prob_ant_sparse.size(2))
    ant_start = int(ant_start)
    ant_end = int(ant_end)
    if ant_end <= ant_start:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return empty, torch.empty((0,), device=device, dtype=torch.int32)

    starts_np = np.asarray(traces.starts, dtype=np.int64)
    offsets_np = starts_np[ant_start:ant_end + 1]
    lo = int(offsets_np[0])
    hi = int(offsets_np[-1])
    counts_t = torch.as_tensor(
        offsets_np[1:] - offsets_np[:-1],
        device=device,
        dtype=torch.int64,
    )
    n_slice_ants = ant_end - ant_start

    ant_idx_all = torch.repeat_interleave(
        torch.arange(n_slice_ants, device=device, dtype=torch.int64),
        counts_t,
    )
    is_stoch = torch.as_tensor(
        np.asarray(traces.is_stochastic, dtype=np.uint8)[lo:hi], device=device
    ).bool()
    ndec = torch.bincount(ant_idx_all[is_stoch], minlength=n_slice_ants).to(torch.int32)
    logp_sum = torch.zeros((n_slice_ants,), device=device, dtype=torch.float32)
    pick = torch.as_tensor(
        np.asarray(traces.pick_j, dtype=np.int64)[lo:hi], device=device
    )
    roulette = is_stoch & (pick >= 0)
    idx = roulette.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() > 0:
        curr = torch.as_tensor(
            np.asarray(traces.curr_nodes, dtype=np.int64)[lo:hi], device=device
        )
        vm_arr = np.asarray(traces.valid_mask, dtype=np.uint64)[lo:hi]
        vm_i64 = torch.as_tensor(vm_arr, device=device).to(torch.int64)
        ants_r = ant_idx_all[idx]
        curr_r = curr[idx]
        pick_r = pick[idx]
        vm_r = vm_i64[idx]
        log_w = log_prob_ant_sparse[ants_r, curr_r]
        valid_bits = (vm_r.unsqueeze(1) & _replay_magic_mask(k, device)) != 0
        log_w_valid = log_w.masked_fill(~valid_bits, float("-inf"))
        log_denom = torch.logsumexp(log_w_valid, dim=1)
        log_numer = log_w.gather(1, pick_r.unsqueeze(1)).squeeze(1)
        logp_sum.scatter_add_(0, ants_r, log_numer - log_denom)
    return logp_sum, ndec

# =============================================================================
# PROBABILITY COMPUTATION
# =============================================================================

def log_prob_sparse_from_tau_eta_prior(
    tau_nk: torch.Tensor,
    eta_nk: torch.Tensor,
    prior_nk: Optional[torch.Tensor],
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute log probabilities from pheromone, heuristic, and neural prior.
    
    Args:
        tau_nk: Pheromone values (n, k)
        eta_nk: Heuristic values (n, k)
        prior_nk: Neural prior logits (n, k), optional
        alpha: Pheromone exponent
        beta: Heuristic exponent
        eps: Small constant for numerical stability
    
    Returns:
        Log weights (unnormalized log probabilities)
    """
    tau = tau_nk.clamp_min(eps)
    eta = eta_nk.clamp_min(eps)
    log_w = alpha * torch.log(tau) + beta * torch.log(eta)
    if prior_nk is not None:
        log_w = log_w + prior_nk
    return log_w


def _device_uses_cuda(device: Any) -> bool:
    """Return True only when the requested device is explicitly CUDA."""
    try:
        return torch.device(device).type == "cuda"
    except (TypeError, ValueError, RuntimeError):
        return False


def _gpu_memory_allocated_gb(device: Any) -> Optional[float]:
    """Return current allocated CUDA memory in GB for CUDA devices, else None."""
    if not torch.cuda.is_available() or not _device_uses_cuda(device):
        return None
    return torch.cuda.memory_allocated(device) / 1024**3





# =============================================================================
# ACO SETUP
# =============================================================================

def setup_aco(
    args: argparse.Namespace,
    instance_data: Any,
    problem_type: str
) -> Tuple[Any, Tuple]:
    """
    Setup ACO solver for the given problem instance.
    
    Args:
        args: Training arguments
        instance_data: Problem instance data (coords for TSP, tuple for CVRP)
        problem_type: 'tsp' or 'cvrp'
    
    Returns:
        Tuple of (aco_solver, pyg_args)
    """
    if problem_type == 'tsp':
        coords = instance_data
        kwargs = {
            'n_ants': args.n_ants,
            'coords': coords,
            'cand_list_size': args.k_sparse,
            'backup_list_size': args.k_sparse,
            'disable_heuristic': args.disable_heuristic,
            'use_local_search': not args.no_local_search,
            'decay': args.rho,
            'device': args.device,
            'enable_torch_sync': True,
            'smooth_mmas': not args.no_smooth_mmas,
            'min_new_edges': args.min_new_edges,
            'extend_ls': not args.no_extend_ls,
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L,
            'nls': args.nls,
            'T_nls': args.T_nls,
            'ls_scope': args.ls_scope,
            'ls_budget': args.ls_budget,
            'ls_max_opt': args.ls_max_opt,
        }
        pyg_args = (
            coords,
            args.device,
            args.ablation_pheromone_features,
            args.ablation_incumbent_features,
            args.edge_feature_set,
        )
        
        if args.alg == 'mmas':
            aco = faco.ACO_TSP(
                coords=coords,
                n_ants=args.n_ants,
                cand_list_size=args.k_sparse,
                decay=args.rho,
                alpha=args.alpha,
                beta=args.beta,
                p_best=0.05,
                min_max=True,
                device=args.device,
                enable_torch_sync=True
            )
        else:
            aco = faco.MFACO_TSP(**kwargs)
    else:  # cvrp
        coords, demand, capacity = instance_data
        kwargs = {
            'coords': coords,
            'demand': demand,
            'capacity': float(capacity),
            'n_ants': args.n_ants,
            'cand_list_size': args.k_sparse,
            'backup_list_size': max(args.k_sparse, 64),
            'min_new_edges': args.min_new_edges,
            'decay': args.rho,
            'p_best': 0.05,
            'use_local_search': not args.no_local_search,
            'disable_heuristic': args.disable_heuristic,
            'extend_ls': not args.no_extend_ls, 
            'smooth_mmas': not args.no_smooth_mmas,
            'device': args.device,
            'enable_torch_sync': True,
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L,
            'nls': args.nls,
            'T_nls': args.T_nls,
            'ls_scope': args.ls_scope,
            'ls_budget': args.ls_budget,
            'ls_max_opt': args.ls_max_opt,
        }
        pyg_args = (
            coords,
            demand,
            args.device,
            args.ablation_pheromone_features,
            args.ablation_incumbent_features,
            args.edge_feature_set,
        )
        
        if args.alg == 'mmas':
            aco = faco.ACO_CVRP(
                coords=coords,
                demand=demand,
                capacity=float(capacity),
                n_ants=args.n_ants,
                cand_list_size=args.k_sparse,
                decay=args.rho,
                alpha=args.alpha,
                beta=args.beta,
                p_best=0.05,
                min_max=True,
                device=args.device,
                enable_torch_sync=True
            )
        else:
            aco = faco.MFACO_CVRP(**kwargs)
    
    return aco, pyg_args


def get_heuristic_tensor(
    aco: Any,
    problem_type: str,
    device: str
) -> torch.Tensor:
    """Get heuristic tensor from ACO solver (unified API)."""
    return aco.h_sparse_torch


def compute_annealing_factor(
    inner: int,
    mini_H: int,
    gamma: float,
    min_gamma: float
) -> float:
    """Compute annealing factor for prior scaling."""
    if mini_H > 1:
        ratio = inner / (mini_H - 1)
        return gamma * (1.0 - ratio) + min_gamma * ratio
    return gamma


def compute_warmup_steps(args: argparse.Namespace, use_train: bool = True) -> int:
    """Compute number of warmup steps based on configuration."""
    warmup_attr = 'train_warmup' if use_train else 'warmup'
    if not getattr(args, warmup_attr, False):
        return 0

    # Warmup should use at most half of H, but always leave at least 1 step for neural guidance
    max_limit = int(args.H * getattr(args, 'warmup_ratio', 0.5))

    # Ensure we don't warmup for all H steps - need at least 1 step with neural guidance
    max_limit = min(max_limit, args.H - 1)

    if max_limit <= 0:
        return 0

    return np.random.randint(1, max_limit + 1) if use_train else max_limit


def _multi_head_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "multi_head", False) or getattr(args, "num_heads", 1) > 1)


def _head_input_transform_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "head_input_transform", "none") or "none").lower()


def _transform_coords_for_head(coords: Any, head_idx: int, mode: str) -> Any:
    if mode in {"none", "identity"}:
        return coords
    if mode != "d4":
        raise ValueError(f"Unsupported head_input_transform: {mode}")

    variants = int(head_idx) % 8
    if torch.is_tensor(coords):
        c = coords.clone()
        center = (c.amin(dim=0, keepdim=True) + c.amax(dim=0, keepdim=True)) * 0.5
        rel = c - center
        x, y = rel[..., 0], rel[..., 1]
        if variants == 0:
            out = torch.stack((x, y), dim=-1)
        elif variants == 1:
            out = torch.stack((-x, y), dim=-1)
        elif variants == 2:
            out = torch.stack((x, -y), dim=-1)
        elif variants == 3:
            out = torch.stack((-x, -y), dim=-1)
        elif variants == 4:
            out = torch.stack((y, x), dim=-1)
        elif variants == 5:
            out = torch.stack((-y, x), dim=-1)
        elif variants == 6:
            out = torch.stack((y, -x), dim=-1)
        else:
            out = torch.stack((-y, -x), dim=-1)
        return out + center

    arr = np.array(coords, copy=True)
    center = (arr.min(axis=0, keepdims=True) + arr.max(axis=0, keepdims=True)) * 0.5
    rel = arr - center
    x, y = rel[..., 0], rel[..., 1]
    if variants == 0:
        out = np.stack((x, y), axis=-1)
    elif variants == 1:
        out = np.stack((-x, y), axis=-1)
    elif variants == 2:
        out = np.stack((x, -y), axis=-1)
    elif variants == 3:
        out = np.stack((-x, -y), axis=-1)
    elif variants == 4:
        out = np.stack((y, x), axis=-1)
    elif variants == 5:
        out = np.stack((-y, x), axis=-1)
    elif variants == 6:
        out = np.stack((y, -x), axis=-1)
    else:
        out = np.stack((-y, -x), axis=-1)
    return out + center


def _build_head_pyg_data(
    aco: Any,
    build_fn: Any,
    pyg_args: Tuple[Any, ...],
    problem: str,
    head_idx: int,
    args: argparse.Namespace,
    dynamic: bool,
) -> Any:
    mode = _head_input_transform_mode(args)
    if mode in {"none", "identity"}:
        return build_fn(aco, *pyg_args, dynamic=dynamic)

    if problem == "tsp":
        coords, device, ablation_pheromone, ablation_incumbent, edge_feature_set = pyg_args
        return build_fn(
            aco,
            _transform_coords_for_head(coords, head_idx, mode),
            device,
            ablation_pheromone=ablation_pheromone,
            ablation_incumbent=ablation_incumbent,
            edge_feature_set=edge_feature_set,
            dynamic=dynamic,
        )

    coords, demand, device, ablation_pheromone, ablation_incumbent, edge_feature_set = pyg_args
    return build_fn(
        aco,
        _transform_coords_for_head(coords, head_idx, mode),
        demand,
        device,
        ablation_pheromone=ablation_pheromone,
        ablation_incumbent=ablation_incumbent,
        edge_feature_set=edge_feature_set,
        dynamic=dynamic,
    )


def _model_to_prior(model: Net, pyg_data: Any, n: int, k: int, args: argparse.Namespace) -> torch.Tensor:
    output = model(pyg_data)
    return net.output_to_sparse_prior(output, n, k)


def _model_to_multi_priors(
    model: Net,
    pyg_data: Any,
    n: int,
    k: int,
    args: Optional[argparse.Namespace] = None,
    aco: Any = None,
    build_fn: Any = None,
    pyg_args: Optional[Tuple[Any, ...]] = None,
    problem: Optional[str] = None,
    dynamic: bool = True,
) -> torch.Tensor:
    if (
        args is not None
        and _head_input_transform_mode(args) not in {"none", "identity"}
        and aco is not None
        and build_fn is not None
        and pyg_args is not None
        and problem is not None
    ):
        outputs = []
        num_heads = int(getattr(args, "num_heads", 1))
        for head_idx in range(num_heads):
            head_pyg = _build_head_pyg_data(aco, build_fn, pyg_args, problem, head_idx, args, dynamic)
            head_output = model(head_pyg)
            if head_output.dim() != 2:
                raise ValueError(
                    f"Expected multi-head output for transformed head inference, got {tuple(head_output.shape)}"
                )
            outputs.append(head_output[:, head_idx])
        return torch.stack(outputs, dim=0).contiguous().view(num_heads, n, k)
    return net.output_to_multi_sparse_priors(model(pyg_data), n, k)


def _model_to_multi_priors_and_alloc(
    model: Net,
    pyg_data: Any,
    n: int,
    k: int,
    args: argparse.Namespace,
    aco: Any = None,
    build_fn: Any = None,
    pyg_args: Optional[Tuple[Any, ...]] = None,
    problem: Optional[str] = None,
    dynamic: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if _learned_router_enabled(args) and hasattr(model, "forward_with_alloc"):
        if _head_input_transform_mode(args) in {"none", "identity"}:
            output, alloc_logits = model.forward_with_alloc(pyg_data)
            return net.output_to_multi_sparse_priors(output, n, k), alloc_logits
        alloc_pyg = pyg_data
        _, alloc_logits = model.forward_with_alloc(alloc_pyg)
        priors = _model_to_multi_priors(
            model, pyg_data, n, k, args=args, aco=aco, build_fn=build_fn,
            pyg_args=pyg_args, problem=problem, dynamic=dynamic
        )
        return priors, alloc_logits
    return _model_to_multi_priors(
        model, pyg_data, n, k, args=args, aco=aco, build_fn=build_fn,
        pyg_args=pyg_args, problem=problem, dynamic=dynamic
    ), None


def _head_cost_scores(head_costs: torch.Tensor, args: Optional[argparse.Namespace] = None) -> torch.Tensor:
    if isinstance(head_costs, (list, tuple)):
        return torch.stack([c.mean() for c in head_costs])
    return head_costs.mean(dim=1)


def _split_ant_counts(n_ants: int, num_heads: int) -> List[int]:
    return _split_ant_counts_from_weights(n_ants, num_heads, None)


def _parse_head_ant_weights(raw: Any, num_heads: int) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple)):
        vals = [float(x) for x in raw]
    else:
        raise ValueError(f"Unsupported head_ant_weights value: {raw!r}")
    if len(vals) != num_heads:
        raise ValueError(f"head_ant_weights must provide {num_heads} values, got {len(vals)}")
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError("head_ant_weights must be non-negative and sum to a positive value")
    return vals


def _split_ant_counts_from_weights(
    n_ants: int,
    num_heads: int,
    weights: Optional[Any],
) -> List[int]:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    vals = _parse_head_ant_weights(weights, num_heads)
    if vals is None:
        base = n_ants // num_heads
        rem = n_ants % num_heads
        counts = [base + (1 if h < rem else 0) for h in range(num_heads)]
    else:
        total = sum(vals)
        exact = [n_ants * v / total for v in vals]
        counts = [int(np.floor(x)) for x in exact]
        for i, v in enumerate(vals):
            if v > 0 and counts[i] == 0:
                counts[i] = 1
        while sum(counts) > n_ants:
            candidates = [i for i, v in enumerate(vals) if counts[i] > (1 if v > 0 else 0)]
            if not candidates:
                break
            i = min(candidates, key=lambda j: exact[j] - counts[j])
            counts[i] -= 1
        while sum(counts) < n_ants:
            i = max(range(num_heads), key=lambda j: exact[j] - counts[j])
            counts[i] += 1
    if any(c <= 0 for c in counts):
        raise ValueError(
            f"n_ants={n_ants} must allocate at least one ant to each active head; got counts={counts}"
        )
    return counts


def _expand_head_priors_to_ants(
    head_priors: torch.Tensor,
    n_ants: int,
    head_ant_weights: Optional[Any] = None,
) -> Tuple[torch.Tensor, List[int]]:
    counts = _split_ant_counts_from_weights(
        n_ants, int(head_priors.shape[0]), head_ant_weights
    )
    ant_priors = torch.cat(
        [head_priors[h:h + 1].expand(count, -1, -1) for h, count in enumerate(counts)],
        dim=0,
    ).contiguous()
    return ant_priors, counts


def _expand_head_priors_with_counts(
    head_priors: torch.Tensor,
    head_counts: List[int],
) -> torch.Tensor:
    if len(head_counts) != int(head_priors.shape[0]):
        raise ValueError(
            f"head_counts must have {head_priors.shape[0]} entries, got {len(head_counts)}"
        )
    if any(int(c) <= 0 for c in head_counts):
        raise ValueError(f"head_counts must be positive, got {head_counts}")
    return torch.cat(
        [head_priors[h:h + 1].expand(int(count), -1, -1) for h, count in enumerate(head_counts)],
        dim=0,
    ).contiguous()


def _split_ant_vector_by_heads(values: torch.Tensor, counts: List[int]) -> List[torch.Tensor]:
    return list(torch.split(values, counts, dim=0))


class MultiHeadRollout(NamedTuple):
    tau_nk: torch.Tensor
    traces: Any
    head_counts: List[int]
    selected_head: Optional[int]
    flats: Any
    costs_all: torch.Tensor
    costs_by_head: List[torch.Tensor]
    logp_old_by_head: List[torch.Tensor]
    ndec_by_head: List[torch.Tensor]
    best_idx: int
    best_cost: float
    new_edges: Any
    survival: Any
    alloc_action: Optional[Any] = None


class AllocationAction(NamedTuple):
    counts: List[int]
    residual_counts: torch.Tensor
    floor_count: int
    probs: torch.Tensor
    logits: torch.Tensor
    logp_old: torch.Tensor
    entropy_old: torch.Tensor
    reward: float


def _learned_router_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "head_router", "static") or "static").lower() == "learned"


def _allocation_floor(n_ants: int, num_heads: int, min_frac: float) -> int:
    if n_ants < num_heads:
        raise ValueError(f"n_ants={n_ants} must be >= num_heads={num_heads} for learned allocation")
    requested = max(1, int(np.floor(float(min_frac) * int(n_ants))))
    return requested if requested * num_heads <= n_ants else 1


def _largest_remainder_counts(total: int, probs: torch.Tensor) -> torch.Tensor:
    if total <= 0:
        return torch.zeros_like(probs, dtype=torch.long)
    exact = probs.detach().double().cpu().numpy() * int(total)
    counts = np.floor(exact).astype(np.int64)
    rem = int(total) - int(counts.sum())
    if rem > 0:
        order = np.argsort(-(exact - counts))
        for idx in order[:rem]:
            counts[int(idx)] += 1
    return torch.as_tensor(counts, device=probs.device, dtype=torch.long)


def _sample_learned_allocation(
    alloc_logits: torch.Tensor,
    args: argparse.Namespace,
    deterministic: bool = False,
    residual_counts: Optional[torch.Tensor] = None,
) -> Tuple[List[int], AllocationAction, torch.Tensor]:
    num_heads = int(alloc_logits.numel())
    floor_count = _allocation_floor(args.n_ants, num_heads, getattr(args, "head_router_min_frac", 0.0))
    residual_total = int(args.n_ants) - floor_count * num_heads
    temperature = max(float(getattr(args, "allocator_temperature", 1.0)), 1e-6)
    probs = torch.softmax(alloc_logits / temperature, dim=0).clamp_min(1e-12)
    probs = probs / probs.sum()
    cat = torch.distributions.Categorical(probs=probs)

    if residual_counts is None:
        if deterministic:
            residual_counts = _largest_remainder_counts(residual_total, probs)
        elif residual_total <= 0:
            residual_counts = torch.zeros(num_heads, device=alloc_logits.device, dtype=torch.long)
        else:
            dist = torch.distributions.Multinomial(total_count=residual_total, probs=probs)
            residual_counts = dist.sample().to(dtype=torch.long)
    else:
        residual_counts = residual_counts.to(device=alloc_logits.device, dtype=torch.long)

    counts_t = residual_counts + int(floor_count)
    counts = [int(x) for x in counts_t.detach().cpu().tolist()]
    if residual_total <= 0:
        logp = alloc_logits.new_tensor(0.0)
    else:
        dist = torch.distributions.Multinomial(total_count=residual_total, probs=probs)
        logp = dist.log_prob(residual_counts.to(dtype=probs.dtype))
    entropy = cat.entropy()
    action = AllocationAction(
        counts=counts,
        residual_counts=residual_counts.detach(),
        floor_count=int(floor_count),
        probs=probs.detach(),
        logits=alloc_logits.detach(),
        logp_old=logp.detach(),
        entropy_old=entropy.detach(),
        reward=0.0,
    )
    return counts, action, probs.detach()


def _ppo_clipped_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    costs: torch.Tensor,
    args: argparse.Namespace,
    baseline: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(logp_new - logp_old)
    log_ratio = logp_new - logp_old
    approx_kl = (log_ratio.pow(2) * 0.5).mean()
    clipped = (ratio > 1 + args.ppo_clip) | (ratio < 1 - args.ppo_clip)
    clip_frac = clipped.float().mean()

    if baseline is None:
        baseline = costs.mean()
    adv = (baseline - costs).detach()
    if not args.no_adv_norm:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * adv
    return -torch.mean(torch.min(surr1, surr2)), approx_kl, clip_frac


def _collect_multi_head_rollout(
    aco: Any,
    current_prior: torch.Tensor,
    eta_nk: torch.Tensor,
    args: argparse.Namespace,
    head_router: Any,
    alloc_logits: Optional[torch.Tensor] = None,
) -> MultiHeadRollout:
    tau_nk = aco.tau_nk_torch().detach()
    alloc_action = None
    if _learned_router_enabled(args):
        if alloc_logits is None:
            raise ValueError("learned head router requires allocation logits")
        with torch.no_grad():
            head_counts, alloc_action, _ = _sample_learned_allocation(
                alloc_logits.detach(), args, deterministic=False
            )
    else:
        head_counts = utils.head_counts_for_router(
            args, int(current_prior.shape[0]), args.n_ants, head_router
        )
    res = aco.sample_mixed_priors(
        current_prior,
        require_prob=True,
        parallel_traced=args.parallel_traced,
        head_counts=head_counts,
    )
    costs, flats, _, _, traces, _, _, new_edges, survival = res
    costs_all = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
    costs_by_head = [x.detach() for x in _split_ant_vector_by_heads(costs_all, head_counts)]
    selected_head = int(torch.argmin(_head_cost_scores(costs_by_head, args).detach()).item())

    with torch.no_grad():
        h0, h1 = _head_ant_range(head_counts, selected_head)
        selected_prior = current_prior[selected_head:selected_head + 1].expand(h1 - h0, -1, -1)
        tau_ant = tau_nk.unsqueeze(0).expand(h1 - h0, -1, -1)
        eta_ant = eta_nk.unsqueeze(0).expand(h1 - h0, -1, -1)
        log_prob_old = log_prob_sparse_from_tau_eta_prior(
            tau_ant, eta_ant, selected_prior,
            alpha=args.alpha, beta=args.beta, eps=EPS
        )
        logp_old_selected, ndec_selected = replay_logp_from_cpp_batch_trace_ant_slice(
            traces, log_prob_old, h0, h1
        )
        ndec_f = ndec_selected.to(torch.float32).clamp_min(1.0)
        logp_old_selected = (logp_old_selected / ndec_f).detach()
        logp_old_by_head = [
            logp_old_selected.detach() if h == selected_head
            else current_prior.new_zeros((int(count),), dtype=torch.float32).detach()
            for h, count in enumerate(head_counts)
        ]
        ndec_by_head = [
            ndec_selected.detach() if h == selected_head
            else torch.zeros((int(count),), device=current_prior.device, dtype=torch.int32)
            for h, count in enumerate(head_counts)
        ]

    best_idx = int(costs_all.argmin().item())
    if alloc_action is not None:
        alloc_action = alloc_action._replace(reward=-float(costs_all[best_idx].item()))
    return MultiHeadRollout(
        tau_nk=tau_nk,
        traces=traces,
        head_counts=head_counts,
        selected_head=selected_head,
        flats=flats,
        costs_all=costs_all,
        costs_by_head=costs_by_head,
        logp_old_by_head=logp_old_by_head,
        ndec_by_head=ndec_by_head,
        best_idx=best_idx,
        best_cost=float(costs_all[best_idx].item()),
        new_edges=new_edges,
        survival=survival,
        alloc_action=alloc_action,
    )


def _record_multi_head_rollout_metrics(
    metrics: MetricsCollector,
    rollout: MultiHeadRollout,
    args: argparse.Namespace,
    head_router: Any,
    incumbent_before: float,
) -> None:
    head_scores = _head_cost_scores(rollout.costs_by_head, args)
    metrics.add("head_selected", float(rollout.selected_head if rollout.selected_head is not None else -1))
    metrics.add("head_cost_spread", float(head_scores.std(unbiased=False).item()))
    metrics.add("head_ants_min", float(min(rollout.head_counts)))
    metrics.add("head_ants_max", float(max(rollout.head_counts)))
    for h, (count, head_costs) in enumerate(zip(rollout.head_counts, rollout.costs_by_head)):
        head_best = float(head_costs.min().item())
        metrics.add(f"head_count_{h}", float(count))
        metrics.add(f"head_mean_cost_{h}", float(head_costs.mean().item()))
        metrics.add(f"head_best_cost_{h}", head_best)
        metrics.add(f"head_win_{h}", 1.0 if h == rollout.selected_head else 0.0)
        if np.isfinite(incumbent_before):
            metrics.add(f"head_improvement_{h}", float(incumbent_before - head_best))
    if rollout.new_edges is not None:
        metrics.add("new_edges", np.asarray(rollout.new_edges, dtype=np.float32).mean())
    if rollout.survival is not None:
        metrics.add("survival", np.asarray(rollout.survival, dtype=np.float32).mean())
    if rollout.selected_head is None:
        ndec_values = rollout.ndec_by_head
        entropy_terms = [
            (-lp / nd.float().clamp_min(1.0))
            for lp, nd in zip(rollout.logp_old_by_head, rollout.ndec_by_head)
        ]
    else:
        ndec_values = [rollout.ndec_by_head[rollout.selected_head]]
        entropy_terms = [
            -rollout.logp_old_by_head[rollout.selected_head]
            / rollout.ndec_by_head[rollout.selected_head].float().clamp_min(1.0)
        ]
    metrics.add("ndec", float(torch.cat([x.float() for x in ndec_values]).mean().item()))
    entropy = float(torch.cat(entropy_terms).mean().item())
    metrics.add("entropy", entropy)

    head_utility = utils.update_head_router(
        head_router,
        rollout.head_counts,
        rollout.costs_all,
        incumbent_before=incumbent_before,
        best_idx=rollout.best_idx,
    )
    router_metrics = utils.head_router_metrics(head_router, rollout.head_counts)
    for key, value in router_metrics.items():
        if not isinstance(value, list):
            metrics.add(key, float(value))
    if head_utility is not None:
        for h, value in enumerate(head_utility):
            metrics.add(f"head_utility_{h}", float(value))
    if rollout.alloc_action is not None:
        counts_np = np.asarray(rollout.alloc_action.counts, dtype=np.float64)
        count_probs_np = counts_np / max(float(counts_np.sum()), 1e-12)
        policy_probs = rollout.alloc_action.probs.detach().cpu().numpy().astype(np.float64)
        policy_logits = rollout.alloc_action.logits.detach().cpu().numpy().astype(np.float64)
        metrics.add("allocator_entropy", float(-(count_probs_np * np.log(count_probs_np + 1e-12)).sum()))
        metrics.add("allocator_policy_entropy", float(rollout.alloc_action.entropy_old.item()))
        metrics.add("allocator_prob_max", float(policy_probs.max()))
        metrics.add("allocator_prob_min", float(policy_probs.min()))
        metrics.add("allocator_logit_max", float(policy_logits.max()))
        metrics.add("allocator_logit_min", float(policy_logits.min()))
        metrics.add("allocator_logit_spread", float(policy_logits.max() - policy_logits.min()))
        metrics.add("allocator_logprob", float(rollout.alloc_action.logp_old.item()))
        metrics.add("allocator_reward", float(rollout.alloc_action.reward))
        metrics.add("allocator_min_ants", float(counts_np.min()))
        metrics.add("allocator_max_ants", float(counts_np.max()))
        for h, c in enumerate(rollout.alloc_action.counts):
            metrics.add(f"allocator_count_{h}", float(c))
        for h, (p, z) in enumerate(zip(policy_probs, policy_logits)):
            metrics.add(f"allocator_prob_{h}", float(p))
            metrics.add(f"allocator_logit_{h}", float(z))


def _multi_head_ppo_loss(
    current_prior: torch.Tensor,
    tau_nk: torch.Tensor,
    eta_nk: torch.Tensor,
    traces: Any,
    head_counts: List[int],
    costs_by_head: List[torch.Tensor],
    logp_old_by_head: List[torch.Tensor],
    args: argparse.Namespace,
    selected_head: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_scores = _head_cost_scores(costs_by_head, args)
    selected = (
        int(selected_head)
        if selected_head is not None
        else int(torch.argmin(head_scores.detach()).item())
    )
    h0, h1 = _head_ant_range(head_counts, selected)
    selected_prior = current_prior[selected:selected + 1].expand(h1 - h0, -1, -1)
    tau_ant = tau_nk.unsqueeze(0).expand(h1 - h0, -1, -1)
    eta_ant = eta_nk.unsqueeze(0).expand(h1 - h0, -1, -1)
    log_prob_new = log_prob_sparse_from_tau_eta_prior(
        tau_ant, eta_ant, selected_prior,
        alpha=args.alpha, beta=args.beta, eps=EPS
    )
    logp_new, ndec_new = replay_logp_from_cpp_batch_trace_ant_slice(
        traces, log_prob_new, h0, h1
    )
    ndec_f = ndec_new.to(torch.float32).clamp_min(1.0)
    logp_new = logp_new / ndec_f
    return _ppo_clipped_loss(
        logp_new,
        logp_old_by_head[selected],
        costs_by_head[selected],
        args,
        baseline=head_scores.mean(),
    )


def _allocation_ppo_loss(
    alloc_logits: torch.Tensor,
    actions: List[AllocationAction],
    rewards: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not actions:
        z = alloc_logits.new_tensor(0.0)
        return z, z.detach(), z.detach(), z.detach(), z.detach()
    losses = []
    kls = []
    clip_fracs = []
    entropy_terms = []
    for action, reward in zip(actions, rewards):
        num_heads = int(alloc_logits.numel())
        floor_count = _allocation_floor(args.n_ants, num_heads, getattr(args, "head_router_min_frac", 0.0))
        residual_total = int(args.n_ants) - floor_count * num_heads
        temperature = max(float(getattr(args, "allocator_temperature", 1.0)), 1e-6)
        probs = torch.softmax(alloc_logits / temperature, dim=0).clamp_min(1e-12)
        probs = probs / probs.sum()
        cat = torch.distributions.Categorical(probs=probs)
        residual_counts = action.residual_counts.to(device=alloc_logits.device, dtype=alloc_logits.dtype)
        if residual_total <= 0:
            logp_new = alloc_logits.new_tensor(0.0)
        else:
            dist = torch.distributions.Multinomial(total_count=residual_total, probs=probs)
            logp_new = dist.log_prob(residual_counts)
        logp_old = action.logp_old.to(device=alloc_logits.device, dtype=alloc_logits.dtype)
        adv = reward.to(device=alloc_logits.device, dtype=alloc_logits.dtype)
        ratio = torch.exp(logp_new - logp_old)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * adv
        losses.append(-torch.min(surr1, surr2))
        kls.append(0.5 * (logp_new - logp_old).pow(2))
        clipped = (ratio > 1 + args.ppo_clip) | (ratio < 1 - args.ppo_clip)
        clip_fracs.append(clipped.to(dtype=alloc_logits.dtype))
        entropy_terms.append(cat.entropy())
    pg_loss = torch.stack(losses).mean()
    entropy = torch.stack(entropy_terms).mean()
    entropy_coef = float(getattr(args, "allocator_entropy_coef", 0.01))
    entropy_bonus = entropy_coef * entropy
    loss = pg_loss - entropy_bonus
    return (
        loss,
        torch.stack(kls).mean().detach(),
        torch.stack(clip_fracs).mean().detach(),
        entropy.detach(),
        entropy_bonus.detach(),
    )


def _multi_head_jensen_diversity(priors: torch.Tensor) -> torch.Tensor:
    """Jensen-to-mean head diversity over per-node sparse-edge distributions."""
    if priors.dim() != 3 or priors.shape[0] <= 1:
        return priors.new_tensor(0.0)
    probs = torch.softmax(priors, dim=-1).clamp_min(EPS)
    mean_prob = probs.mean(dim=0).clamp_min(EPS)
    entropy = -(probs * probs.log()).sum(dim=-1)
    mean_entropy = -(mean_prob * mean_prob.log()).sum(dim=-1)
    return (mean_entropy - entropy.mean(dim=0)).mean()


def _multi_head_js_loss_term(priors: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    coef = float(getattr(args, "loss_js", 0.0) or 0.0)
    if coef <= 0.0:
        return priors.new_tensor(0.0)
    return -coef * _multi_head_jensen_diversity(priors)


def _multi_head_js_diversity_metrics(priors: torch.Tensor) -> Dict[str, float]:
    if priors is None or not torch.is_tensor(priors) or priors.dim() != 3 or priors.shape[0] <= 1:
        return {}

    probs = torch.softmax(priors.detach().float(), dim=-1).clamp_min(EPS)
    mean_prob = probs.mean(dim=0).clamp_min(EPS)
    entropy = -(probs * torch.log2(probs)).sum(dim=-1)
    mean_entropy = -(mean_prob * torch.log2(mean_prob)).sum(dim=-1)
    jensen_to_mean = (mean_entropy - entropy.mean(dim=0)).mean()

    pair_jsd = []
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            m = (0.5 * (probs[i] + probs[j])).clamp_min(EPS)
            js = 0.5 * (
                (probs[i] * (torch.log2(probs[i]) - torch.log2(m))).sum(dim=-1)
                + (probs[j] * (torch.log2(probs[j]) - torch.log2(m))).sum(dim=-1)
            )
            pair_jsd.append(js.mean())

    return {
        "head_pairwise_jsd_bits": float(torch.stack(pair_jsd).mean().detach().item()) if pair_jsd else 0.0,
        "head_jensen_to_mean_bits": float(jensen_to_mean.detach().item()),
    }


def _copy_single_head_weights_into_multi_head(model: Net, checkpoint_path: str, device: str) -> None:
    """Initialize a multi-head model from a trained single-head decoder."""
    if not isinstance(model, MultiHeadNet):
        raise ValueError("Single-head initialization requires a multi-head target")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    emb_state = {
        k[len("emb_net."):]: v
        for k, v in state_dict.items()
        if k.startswith("emb_net.")
    }
    missing, unexpected = model.emb_net.load_state_dict(emb_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys while loading {checkpoint_path}: {unexpected}")
    if missing:
        print(f"Warning: missing encoder keys during single-head init: {missing}")

    single_prefix = "par_net_heu.lins."
    required = [
        f"{single_prefix}0.weight", f"{single_prefix}0.bias",
        f"{single_prefix}1.weight", f"{single_prefix}1.bias",
        f"{single_prefix}2.weight", f"{single_prefix}2.bias",
    ]
    missing_single = [k for k in required if k not in state_dict]
    if missing_single:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not look like a compatible single-head Net; "
            f"missing {missing_single}"
        )

    with torch.no_grad():
        if isinstance(model, MultiHeadNet):
            par = model.par_net_heu
            par.hidden[0].weight.copy_(state_dict[f"{single_prefix}0.weight"])
            par.hidden[0].bias.copy_(state_dict[f"{single_prefix}0.bias"])
            par.hidden[1].weight.copy_(state_dict[f"{single_prefix}1.weight"])
            par.hidden[1].bias.copy_(state_dict[f"{single_prefix}1.bias"])
            par.out.base.weight.copy_(state_dict[f"{single_prefix}2.weight"])
            par.out.base.bias.copy_(state_dict[f"{single_prefix}2.bias"])
            par.out.lora_B.zero_()
        else:
            for decoder in model.par_net_heu:
                decoder.lins[0].weight.copy_(state_dict[f"{single_prefix}0.weight"])
                decoder.lins[0].bias.copy_(state_dict[f"{single_prefix}0.bias"])
                decoder.lins[1].weight.copy_(state_dict[f"{single_prefix}1.weight"])
                decoder.lins[1].bias.copy_(state_dict[f"{single_prefix}1.bias"])
                decoder.lins[2].weight.copy_(state_dict[f"{single_prefix}2.weight"])
                decoder.lins[2].bias.copy_(state_dict[f"{single_prefix}2.bias"])
    print(f"Initialized multi-head model from single-head checkpoint: {checkpoint_path}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def _collect_prior_metrics(
    metrics: MetricsCollector,
    prior: torch.Tensor,
    prior_prev_cpu: Optional[torch.Tensor],
    eta_nk: torch.Tensor,
    simple_train: bool
) -> Optional[torch.Tensor]:
    """Collect metrics for neural prior and return CPU copy for next iteration."""
    metrics.add("prior_mean", prior.mean().item())
    metrics.add("prior_std", prior.std().item())
    
    if simple_train:
        return None
    
    # Track prior-eta correlation
    metrics.add("prior_eta_corr", safe_corr(prior, eta_nk))
    
    # Move to CPU for drift metrics
    prior_cpu = prior.detach().cpu()
    
    if prior_prev_cpu is not None:
        metrics.add("prior_l2_drift", rel_l2_drift(prior_prev_cpu, prior_cpu))
        metrics.add("prior_kl", mean_row_kl(prior_prev_cpu, prior_cpu))
        metrics.add("prior_turnover", top_turnover(prior_prev_cpu, prior_cpu))
        metrics.add("prior_flip", top1_flip_rate(prior_prev_cpu, prior_cpu))
    
    return prior_cpu


def _update_aco_timings(
    metrics: MetricsCollector,
    aco: Any,
    t_sampling: float,
    t_ls: float,
    t_update: float
) -> Tuple[float, float, float]:
    """Extract and accumulate ACO timing information."""
    if hasattr(aco, "get_timings"):
        timings = aco.get_timings()
        if "time_sampling" in timings:
            t_sampling += timings["time_sampling"] / 1000.0
        if "time_ls" in timings:
            t_ls += timings["time_ls"] / 1000.0
        if "time_update" in timings:
            t_update += timings["time_update"] / 1000.0
    return t_sampling, t_ls, t_update


def train_instance_reinforce(
    model: Net,
    optimizer: torch.optim.Optimizer,
    instance_data: Any,
    args: argparse.Namespace
) -> Tuple[float, float, Dict[str, float]]:
    """
    Train on a single instance using REINFORCE algorithm.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        instance_data: Problem instance
        args: Training arguments
    
    Returns:
        Tuple of (avg_cost, best_cost, metrics_dict)
    """
    model.train()
    
    aco, pyg_args = setup_aco(args, instance_data, args.problem)
    eta_nk = get_heuristic_tensor(aco, args.problem, args.device)
    if args.problem == 'tsp':
        build_fn = utils.build_pyg_data_tsp
    else:
        build_fn = utils.build_pyg_data_cvrp

    best_seen = float("inf")
    avg_cost_last = None
    
    metrics = MetricsCollector()
    prior_prev_cpu = None
    
    # Timing accumulators
    t_neural_total = 0.0
    t_aco_sampling = 0.0
    t_aco_ls = 0.0
    t_aco_update = 0.0
    t_aco_total = 0.0
    
    warmup_steps = compute_warmup_steps(args, use_train=True)

    for outer in tqdm(range(args.H), desc="Outer", leave=False):
        t0 = time.time()
        pyg_data = build_fn(aco, *pyg_args, dynamic=not args.no_dynamic_feats)
        
        prior_old = None
        if outer >= warmup_steps:
            with torch.no_grad():
                prior_old = _model_to_prior(model, pyg_data, aco.n, aco.k, args)
                t_neural_total += time.time() - t0
                
                prior_prev_cpu = _collect_prior_metrics(
                    metrics, prior_old, prior_prev_cpu, eta_nk, args.simple_train
                )

        tau_list = []
        traces_list = []
        costs_list = []

        if hasattr(aco, "reset_timings"):
            aco.reset_timings()

        t_aco_start_outer = time.time()

        for inner in range(args.mini_H):
            current_prior = prior_old
            if prior_old is not None and args.train_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = prior_old * factor

            res = aco.sample(require_prob=True, prior=current_prior, parallel_traced=args.parallel_traced)
            costs, flats, _, _, traces, costs_raw, flats_raw, new_edges, survival = res
            
            if survival is not None:
                metrics.add("survival", survival.mean().item())

            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            
            tau_nk = aco.tau_nk_torch().detach()
            tau_list.append(tau_nk)
            traces_list.append(traces)
            costs_list.append(costs_t.detach())
            
            if new_edges is not None:
                metrics.add("new_edges", new_edges.astype(np.float32).mean())

            best_idx = int(costs_t.argmin().item())
            best_cost_iter = float(costs[best_idx])
            best_seen = min(best_seen, best_cost_iter)
            
            if not args.train_deepaco:
                with torch.no_grad():
                    aco.update_pheromone(flats[best_idx], best_cost_iter)

            avg_cost_last = float(costs_t.mean().item())

        t_aco_total += time.time() - t_aco_start_outer
        t_aco_sampling, t_aco_ls, t_aco_update = _update_aco_timings(
            metrics, aco, t_aco_sampling, t_aco_ls, t_aco_update
        )

        if outer < warmup_steps:
            continue

        optimizer.zero_grad(set_to_none=True)
        
        t0 = time.time()
        prior_new_base = _model_to_prior(model, pyg_data, aco.n, aco.k, args)
        t_neural_total += time.time() - t0
        
        if getattr(args, 'smallvram', False):
            prior_new = prior_new_base.detach().requires_grad_(True)
        else:
            prior_new = prior_new_base
        
        all_losses = []
        all_entropies = []
        total_loss_val = 0.0
        
        for inner in range(args.mini_H):
            current_prior = prior_new
            if args.train_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = prior_new * factor

            tau_nk = tau_list[inner]
            traces = traces_list[inner]
            costs_t = costs_list[inner]
            
            log_prob_new = log_prob_sparse_from_tau_eta_prior(
                tau_nk, eta_nk, current_prior,
                alpha=args.alpha, beta=args.beta, eps=EPS
            )
            logp_new, ndec_new = replay_logp_from_cpp_batch_trace(traces, log_prob_new)
            ndec_f = ndec_new.to(torch.float32).clamp_min(1.0)
            logp_new = logp_new / ndec_f
            
            baseline = costs_t.mean()
            adv = (costs_t - baseline).detach()
            
            loss = (logp_new * adv).mean()
            if getattr(args, 'smallvram', False):
                scaled_loss = loss / args.mini_H
                scaled_loss.backward()
            else:
                all_losses.append(loss)
            
            total_loss_val += loss.item()
            
            entropy = -logp_new.mean()
            all_entropies.append(entropy.detach().item())

        if getattr(args, 'smallvram', False):
            prior_new_base.backward(prior_new.grad)
            total_loss_item = total_loss_val / args.mini_H
        else:
            total_loss = torch.stack(all_losses).mean()
            total_loss.backward()
            total_loss_item = total_loss.item()
        
        if not args.simple_train:
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            if grad_norms:
                metrics.add("grad_var", np.var(grad_norms))
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        metrics.add("loss", total_loss_item)
        metrics.add("entropy", np.mean(all_entropies))
        metrics.add("ndec", ndec_f.mean().item())

    # Finalize timing metrics
    out_metrics = metrics.get_all_means()
    out_metrics["time_neural"] = t_neural_total
    out_metrics["time_aco"] = t_aco_total
    if t_aco_sampling > 0:
        out_metrics["time_sampling"] = t_aco_sampling
    if t_aco_ls > 0:
        out_metrics["time_ls"] = t_aco_ls
    if t_aco_update > 0:
        out_metrics["time_update"] = t_aco_update
    
    return avg_cost_last, best_seen, out_metrics

def train_instance_ppo(
    model: Net,
    optimizer: torch.optim.Optimizer,
    instance_data: Any,
    args: argparse.Namespace
) -> Tuple[float, float, Dict[str, float]]:
    """
    Train on a single instance using PPO algorithm.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        instance_data: Problem instance
        args: Training arguments
    
    Returns:
        Tuple of (avg_cost, best_cost, metrics_dict)
    """
    model.train()
    if _multi_head_enabled(args) and getattr(args, "smallvram", False):
        raise ValueError("--smallvram is not implemented for multi-head PPO")
    
    aco, pyg_args = setup_aco(args, instance_data, args.problem)
    head_router = None
    if _multi_head_enabled(args) and not _learned_router_enabled(args):
        head_router = utils.make_head_router(args, int(getattr(args, "num_heads", 1)), args.n_ants)
    eta_nk = get_heuristic_tensor(aco, args.problem, args.device)
    if args.problem == 'tsp':
        build_fn = utils.build_pyg_data_tsp
    else:
        build_fn = utils.build_pyg_data_cvrp

    best_seen = float("inf")
    avg_cost_last = None
    
    metrics = MetricsCollector()
    prior_prev_cpu = None
    
    # Timing accumulators
    t_neural_total = 0.0
    t_aco_sampling = 0.0
    t_aco_ls = 0.0
    t_aco_update = 0.0
    t_aco_total = 0.0
    
    warmup_steps = compute_warmup_steps(args, use_train=True)

    for outer in tqdm(range(args.H), desc="Outer", leave=False):
        t0 = time.time()
        pyg_data = build_fn(aco, *pyg_args, dynamic=not args.no_dynamic_feats)
        
        prior_old = None
        prior_old_for_metrics = None
        if outer >= warmup_steps:
            with torch.no_grad():
                if _multi_head_enabled(args):
                    prior_old, alloc_logits_old = _model_to_multi_priors_and_alloc(
                        model,
                        pyg_data,
                        aco.n,
                        aco.k,
                        args=args,
                        aco=aco,
                        build_fn=build_fn,
                        pyg_args=pyg_args,
                        problem=args.problem,
                        dynamic=not args.no_dynamic_feats,
                    )
                    if getattr(args, "log_best_head", False):
                        metrics.add_dict(_multi_head_js_diversity_metrics(prior_old))
                    prior_old_for_metrics = prior_old.mean(dim=0)
                else:
                    prior_old = _model_to_prior(model, pyg_data, aco.n, aco.k, args)
                    alloc_logits_old = None
                    prior_old_for_metrics = prior_old
                t_neural_total += time.time() - t0
                
                prior_prev_cpu = _collect_prior_metrics(
                    metrics, prior_old_for_metrics, prior_prev_cpu, eta_nk, args.simple_train
                )

        # Storage for PPO update
        traces_list = []
        flats_list = []
        costs_list = []
        logp_old_list = []
        ndec_list = []
        tau_list = []
        alloc_actions: List[AllocationAction] = []
        costs_raw_t = None
        
        if hasattr(aco, "reset_timings"):
            aco.reset_timings()

        t_aco_start_outer = time.time()

        for inner in range(args.mini_H):
            current_prior = prior_old
            if prior_old is not None and args.train_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = prior_old * factor

            if _multi_head_enabled(args) and current_prior is not None:
                incumbent_before = best_seen
                rollout = _collect_multi_head_rollout(
                    aco,
                    current_prior,
                    eta_nk,
                    args,
                    head_router,
                    alloc_logits=alloc_logits_old,
                )
                if rollout.alloc_action is not None:
                    alloc_actions.append(rollout.alloc_action)
                costs_raw_t = rollout.costs_by_head

                tau_list.append(rollout.tau_nk)
                traces_list.append((rollout.traces, rollout.head_counts, rollout.selected_head))
                flats_list.append(rollout.flats)
                costs_list.append(rollout.costs_by_head)
                logp_old_list.append(rollout.logp_old_by_head)
                ndec_list.append(rollout.ndec_by_head)

                _record_multi_head_rollout_metrics(
                    metrics, rollout, args, head_router, incumbent_before
                )
                best_idx = rollout.best_idx
                best_cost_iter = rollout.best_cost
                best_seen = min(best_seen, best_cost_iter)
                if not args.train_deepaco:
                    with torch.no_grad():
                        aco.update_pheromone(rollout.flats[best_idx], best_cost_iter)
                avg_cost_last = float(torch.cat(rollout.costs_by_head).mean().item())
                continue

            # Sample from ACO
            res = aco.sample(require_prob=True, prior=current_prior, parallel_traced=args.parallel_traced)
            costs, flats, _, _, traces, costs_raw, flats_raw, new_edges, survival = res
            
            if survival is not None:
                metrics.add("survival", survival.mean().item())

            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            if costs_raw is not None:
                costs_raw_t = torch.as_tensor(costs_raw, device=args.device, dtype=torch.float32)
            else:
                costs_raw_t = costs_t
            
            tau_nk = aco.tau_nk_torch().detach()

            # Compute old log probabilities
            with torch.no_grad():
                log_prob_old = log_prob_sparse_from_tau_eta_prior(
                    tau_nk, eta_nk, current_prior,
                    alpha=args.alpha, beta=args.beta, eps=EPS
                )
                logp_old, ndec = replay_logp_from_cpp_batch_trace(traces, log_prob_old)
                ndec_f = ndec.to(torch.float32).clamp_min(1.0)
                logp_old = (logp_old / ndec_f).detach()

            # Store for PPO update
            tau_list.append(tau_nk.detach())
            traces_list.append(traces)
            flats_list.append(None)
            costs_list.append(costs_t.detach())
            logp_old_list.append(logp_old)
            ndec_list.append(ndec.detach())
            
            if new_edges is not None:
                metrics.add("new_edges", new_edges.astype(np.float32).mean())

            metrics.add("ndec", ndec.float().mean().item())
            entropy = (-logp_old / ndec.float().clamp_min(1.0)).mean().item()
            metrics.add("entropy", entropy)

            best_idx = int(costs_t.argmin().item())
            best_cost_iter = float(costs[best_idx])
            best_seen = min(best_seen, best_cost_iter)
            
            if not args.train_deepaco:
                with torch.no_grad():
                    aco.update_pheromone(flats[best_idx], best_cost_iter)

            avg_cost_last = float(costs_t.mean().item())

        t_aco_total += time.time() - t_aco_start_outer
        t_aco_sampling, t_aco_ls, t_aco_update = _update_aco_timings(
            metrics, aco, t_aco_sampling, t_aco_ls, t_aco_update
        )

        # Skip PPO update during warmup
        if outer < warmup_steps:
            continue

        # PPO Update loop
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            
            t0 = time.time()
            if _multi_head_enabled(args):
                prior_new_base, alloc_logits_new = _model_to_multi_priors_and_alloc(
                    model,
                    pyg_data,
                    aco.n,
                    aco.k,
                    args=args,
                    aco=aco,
                    build_fn=build_fn,
                    pyg_args=pyg_args,
                    problem=args.problem,
                    dynamic=not args.no_dynamic_feats,
                )
                if getattr(args, "log_best_head", False):
                    for key, value in _multi_head_js_diversity_metrics(prior_new_base).items():
                        metrics.add(f"update_{key}", value)
            else:
                prior_new_base = _model_to_prior(model, pyg_data, aco.n, aco.k, args)
                alloc_logits_new = None
            t_neural_total += time.time() - t0
            
            if getattr(args, 'smallvram', False):
                prior_new = prior_new_base.detach().requires_grad_(True)
            else:
                prior_new = prior_new_base
            
            all_param_kl = []
            all_clip_frac = []
            all_losses = []
            total_loss_val_epoch = 0.0
            
            for inner in range(args.mini_H):
                current_prior = prior_new
                if args.train_anneal:
                    factor = compute_annealing_factor(
                        inner, args.mini_H, args.gamma, args.min_gamma
                    )
                    current_prior = prior_new * factor

                tau_nk = tau_list[inner]
                traces = traces_list[inner]
                costs_t = costs_list[inner]
                logp_old = logp_old_list[inner]

                if _multi_head_enabled(args):
                    if len(traces) == 2:
                        traces_obj, head_counts = traces
                        selected_head = None
                    else:
                        traces_obj, head_counts, selected_head = traces
                    loss, approx_kl, clip_frac = _multi_head_ppo_loss(
                        current_prior,
                        tau_nk,
                        eta_nk,
                        traces_obj,
                        head_counts,
                        costs_t,
                        logp_old,
                        args,
                        selected_head=selected_head,
                    )
                    js_loss = _multi_head_js_loss_term(current_prior, args)
                    if js_loss.detach().item() != 0.0:
                        loss = loss + js_loss
                        metrics.add("loss_js", js_loss.detach().item())
                        metrics.add("head_jensen_diversity_nats", _multi_head_jensen_diversity(current_prior).detach().item())
                    all_losses.append(loss)
                    all_param_kl.append(approx_kl.detach().item())
                    all_clip_frac.append(clip_frac.detach().item())
                    total_loss_val_epoch += loss.item()
                    continue

                log_prob_new = log_prob_sparse_from_tau_eta_prior(
                    tau_nk, eta_nk, current_prior,
                    alpha=args.alpha, beta=args.beta, eps=EPS
                )
                logp_new, ndec_new = replay_logp_from_cpp_batch_trace(traces, log_prob_new)
                ndec_f = ndec_new.to(torch.float32).clamp_min(1.0)
                logp_new = logp_new / ndec_f
                
                ratio = torch.exp(logp_new - logp_old)
                
                log_ratio = logp_new - logp_old
                approx_kl = (log_ratio.pow(2) * 0.5).mean()
                all_param_kl.append(approx_kl.detach().item())
                
                clipped = (ratio > 1 + args.ppo_clip) | (ratio < 1 - args.ppo_clip)
                clip_frac = clipped.float().mean()
                all_clip_frac.append(clip_frac.detach().item())
                
                # Advantage calculation
                if args.nls and costs_raw_t is not None:
                    cost_combined = args.nls_beta * costs_t + (1.0 - args.nls_beta) * costs_raw_t
                    baseline = cost_combined.mean()
                    adv = (baseline - cost_combined).detach()
                else:
                    baseline = costs_t.mean()
                    adv = (baseline - costs_t).detach()
                
                if not args.no_adv_norm:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * adv
                loss = -torch.mean(torch.min(surr1, surr2))
                
                if getattr(args, 'smallvram', False):
                    scaled_loss = loss / args.mini_H
                    scaled_loss.backward()
                else:
                    all_losses.append(loss)
                    
                total_loss_val_epoch += loss.item()

            if _multi_head_enabled(args) and _learned_router_enabled(args) and alloc_actions:
                raw_rewards = torch.as_tensor(
                    [a.reward for a in alloc_actions],
                    device=args.device,
                    dtype=torch.float32,
                )
                rewards = raw_rewards
                if not args.no_adv_norm:
                    if rewards.numel() > 1:
                        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
                    else:
                        rewards = rewards - rewards.mean()
                alloc_loss, alloc_kl, alloc_clip, alloc_entropy, alloc_entropy_bonus = _allocation_ppo_loss(
                    alloc_logits_new,
                    alloc_actions,
                    rewards,
                    args,
                )
                alloc_loss = float(getattr(args, "allocator_loss_coef", 1.0)) * alloc_loss
                all_losses.append(alloc_loss)
                all_param_kl.append(float(alloc_kl.item()))
                all_clip_frac.append(float(alloc_clip.item()))
                total_loss_val_epoch += float(alloc_loss.detach().item())
                metrics.add("allocator_loss", float(alloc_loss.detach().item()))
                metrics.add("allocator_kl", float(alloc_kl.item()))
                metrics.add("allocator_clip_frac", float(alloc_clip.item()))
                metrics.add("allocator_policy_entropy_update", float(alloc_entropy.item()))
                metrics.add("allocator_entropy_bonus", float(alloc_entropy_bonus.item()))
                metrics.add("allocator_reward_mean", float(raw_rewards.mean().item()))
                metrics.add("allocator_reward_std", float(raw_rewards.std(unbiased=False).item()))
                metrics.add("allocator_adv_mean", float(rewards.mean().item()))
                metrics.add("allocator_adv_std", float(rewards.std(unbiased=False).item()))
            
            if getattr(args, 'smallvram', False):
                prior_new_base.backward(prior_new.grad)
            else:
                total_loss = torch.stack(all_losses).mean()
                total_loss.backward()
            
            # Compute gradient variance
            if not args.simple_train:
                grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                if grad_norms:
                    metrics.add("grad_var", np.var(grad_norms))
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            metrics.add("loss", total_loss_val_epoch / args.mini_H)
            metrics.add("approx_kl", np.mean(all_param_kl))
            metrics.add("clip_frac", np.mean(all_clip_frac))

    # Finalize metrics
    out_metrics = metrics.get_all_means()
    out_metrics["time_neural"] = t_neural_total
    out_metrics["time_aco"] = t_aco_total
    if t_aco_sampling > 0:
        out_metrics["time_sampling"] = t_aco_sampling
    if t_aco_ls > 0:
        out_metrics["time_ls"] = t_aco_ls
    if t_aco_update > 0:
        out_metrics["time_update"] = t_aco_update

    # Cleanup
    del traces_list, flats_list, tau_list, logp_old_list, ndec_list, costs_list
    gc.collect()
    torch.cuda.empty_cache()

    return avg_cost_last, best_seen, out_metrics




# =============================================================================
# EPOCH TRAINING & INFERENCE
# =============================================================================

def train_epoch(
    net: Net,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    args: argparse.Namespace
) -> Tuple[int, float, float, float, float]:
    """
    Train for one epoch across multiple instances.
    
    Args:
        net: Neural network model
        optimizer: Optimizer
        global_step: Current global step
        epoch: Current epoch number
        args: Training arguments
    
    Returns:
        Tuple of (global_step, avg_cost, t_neural, t_aco, epoch_train_time)
    """
    logger = get_logger()
    
    sum_avg_cost = 0
    steps = args.steps_per_epoch
    
    gen_func = (utils.generate_tsp_instance if args.problem == 'tsp' 
                else utils.gen_cvrp_instance)

    epoch_train_time = 0.0
    epoch_time_neural = 0.0
    epoch_time_aco = 0.0
    
    for step in tqdm(
        range(steps),
        desc=f"Epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    ):
        # Generate instance
        if args.problem == 'tsp':
            instance_data = np.random.rand(args.n_node, 2).astype(np.float32)
        else:
            coords_t, demand_t, capacity = gen_func(
                args.n_node,
                device=args.device,
                capacity=args.capacity_override,
            )
            instance_data = (
                coords_t.detach().cpu().numpy().astype(np.float32),
                demand_t.detach().cpu().numpy().astype(np.float32),
                capacity
            )
        
        # Train on instance
        t_start_instance = time.time()
        if args.algo == 'ppo':
            avg_cost, best_cost, metrics = train_instance_ppo(
                net, optimizer, instance_data, args
            )
        else:
            avg_cost, best_cost, metrics = train_instance_reinforce(
                net, optimizer, instance_data, args
            )
        epoch_train_time += time.time() - t_start_instance
            
        sum_avg_cost += avg_cost
        
        if "time_neural" in metrics:
            epoch_time_neural = metrics["time_neural"]
        if "time_aco" in metrics:
            epoch_time_aco = metrics["time_aco"]

        # Log step metrics
        logger.set_step(global_step)
        logger.log_train_step(avg_cost, best_cost, epoch, metrics, global_step)
        global_step += 1
    
    return global_step, sum_avg_cost / steps, epoch_time_neural, epoch_time_aco, epoch_train_time


def infer_instance(
    net: Net,
    instance_data: Any,
    k: int,
    n_ants: int,
    dynamic: bool,
    args: argparse.Namespace,
    collect_metrics: bool = True
) -> Tuple[float, float, Dict[str, float], Dict[str, List[float]]]:
    """
    Run inference on a single instance.
    
    Args:
        net: Neural network model
        instance_data: Problem instance (coords for TSP, tuple for CVRP)
        k: Sparse neighbor count
        n_ants: Number of ants
        dynamic: Whether to use dynamic features
        args: Arguments
        collect_metrics: Whether to collect detailed metrics
    
    Returns:
        Tuple of (avg_cost, best_cost, timings, metrics_log)
    """
    if args.problem == 'tsp':
        aco, pyg_args = setup_aco(args, instance_data, 'tsp')
        build_fn = utils.build_pyg_data_tsp
    else:
        aco, pyg_args = setup_aco(args, instance_data, 'cvrp')
        build_fn = utils.build_pyg_data_cvrp

    best_seen = float("inf")
    if net is not None:
        net.eval()
    head_router = None
    if net is not None and _multi_head_enabled(args) and not _learned_router_enabled(args):
        head_router = utils.make_head_router(args, int(getattr(args, "num_heads", 1)), args.n_ants)
    
    # Initialize metrics
    metrics_log: Dict[str, List[float]] = {}
    if collect_metrics:
        metrics_log = {
            'new_edges': [], 'prior_mean': [], 'prior_std': [],
            'prior_l2_drift': [], 'prior_kl': [], 'prior_turnover': [], 'prior_flip': [],
            'prior_eta_corr': [], 'survival': []
        }
        eta_nk = get_heuristic_tensor(aco, args.problem, args.device)
    
    # Normalize coordinates for model input (scale to [0, 1] while preserving aspect ratio)
    norm_coords = instance_data
    if args.problem == 'tsp':
        coords = instance_data
    else:
        coords = instance_data[0]

    if net is not None:
        if torch.is_tensor(coords):
             c_min = coords.min(dim=0)[0]
             c_max = coords.max(dim=0)[0]
             c_diff = c_max - c_min
             scale = c_diff.max()
             if scale < 1e-6: scale = 1.0
             norm_coords = (coords - c_min) / scale
        else:
             c_min = coords.min(axis=0)
             c_max = coords.max(axis=0)
             c_diff = c_max - c_min
             scale = c_diff.max()
             if scale < 1e-6: scale = 1.0
             norm_coords = (coords - c_min) / scale
        
        # If CVRP, repackage norm_coords into the tuple structure expected by build_fn
        if args.problem == 'cvrp':
            # instance_data is (coords, demand, capacity)
            # norm_coords becomes the new coords
            norm_coords = (norm_coords, instance_data[1], instance_data[2])


    prior_prev_outer = None
    warmup_steps = compute_warmup_steps(args, use_train=False)
    best_seen_before_ls = float("inf")
    avg_cost_before_ls = None

    for outer in range(args.H):
        pyg_data = build_fn(aco, *pyg_args, dynamic=dynamic)
        
        # Use normalized coordinates for the network to match training distribution
        if net is not None:
            if args.problem == 'tsp':
                pyg_args_net = (
                    norm_coords,
                    args.device,
                    args.ablation_pheromone_features,
                    args.ablation_incumbent_features,
                    args.edge_feature_set,
                )
                pyg_data_net = build_fn(
                    aco,
                    *pyg_args_net,
                    dynamic=dynamic,
                )
            else:
                # Pass normalized coordinates and original demand for network input
                pyg_args_net = (
                    norm_coords[0],
                    norm_coords[1],
                    args.device,
                    args.ablation_pheromone_features,
                    args.ablation_incumbent_features,
                    args.edge_feature_set,
                )
                pyg_data_net = build_fn(
                    aco,
                    *pyg_args_net,
                    dynamic=dynamic,
                )
        else:
            pyg_data_net = pyg_data

        guidance = None
        guidance_multi = None
        alloc_logits = None
        alloc_action = None
        guidance_head_counts = None
        if net is not None and outer >= warmup_steps:
            with torch.no_grad():
                if _multi_head_enabled(args):
                    guidance_multi, alloc_logits = _model_to_multi_priors_and_alloc(
                        net,
                        pyg_data_net,
                        aco.n,
                        aco.k,
                        args=args,
                        aco=aco,
                        build_fn=build_fn,
                        pyg_args=pyg_args_net,
                        problem=args.problem,
                        dynamic=dynamic,
                    )
                    guidance = guidance_multi.mean(dim=0)
                    if _learned_router_enabled(args) and alloc_logits is not None:
                        guidance_head_counts, alloc_action, _ = _sample_learned_allocation(
                            alloc_logits, args, deterministic=True
                        )
                        if collect_metrics:
                            policy_probs = alloc_action.probs.detach().cpu().numpy().astype(np.float64)
                            policy_logits = alloc_action.logits.detach().cpu().numpy().astype(np.float64)
                            count_probs = (
                                np.asarray(alloc_action.counts, dtype=np.float64)
                                / max(float(sum(alloc_action.counts)), 1e-12)
                            )
                            metrics_log.setdefault("allocator_entropy", []).append(
                                float(-(count_probs * np.log(count_probs + 1e-12)).sum())
                            )
                            metrics_log.setdefault("allocator_policy_entropy", []).append(
                                float(alloc_action.entropy_old.item())
                            )
                            metrics_log.setdefault("allocator_prob_max", []).append(float(policy_probs.max()))
                            metrics_log.setdefault("allocator_prob_min", []).append(float(policy_probs.min()))
                            metrics_log.setdefault("allocator_logit_max", []).append(float(policy_logits.max()))
                            metrics_log.setdefault("allocator_logit_min", []).append(float(policy_logits.min()))
                            metrics_log.setdefault("allocator_logit_spread", []).append(
                                float(policy_logits.max() - policy_logits.min())
                            )
                            for h, c in enumerate(alloc_action.counts):
                                metrics_log.setdefault(f"allocator_count_{h}", []).append(float(c))
                            for h, (p, z) in enumerate(zip(policy_probs, policy_logits)):
                                metrics_log.setdefault(f"allocator_prob_{h}", []).append(float(p))
                                metrics_log.setdefault(f"allocator_logit_{h}", []).append(float(z))
                else:
                    prior_output = net(pyg_data_net)
                    guidance = dynaco_net.output_to_sparse_prior(prior_output, aco.n, aco.k)
        
        # Track prior metrics
        if collect_metrics and guidance is not None:
            metrics_log['prior_mean'].append(guidance.mean().item())
            metrics_log['prior_std'].append(guidance.std().item())
            
            if prior_prev_outer is not None:
                metrics_log['prior_l2_drift'].append(
                    rel_l2_drift(prior_prev_outer, guidance)
                )
                metrics_log['prior_kl'].append(mean_row_kl(prior_prev_outer, guidance))
                metrics_log['prior_turnover'].append(top_turnover(prior_prev_outer, guidance))
                metrics_log['prior_flip'].append(top1_flip_rate(prior_prev_outer, guidance))
            
            metrics_log['prior_eta_corr'].append(safe_corr(guidance, eta_nk))
            prior_prev_outer = guidance.detach().clone()
        
        # Inner loop
        for inner in range(args.mini_H):
            current_prior = guidance_multi if guidance_multi is not None else guidance
            if current_prior is not None and not args.no_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = current_prior * factor
            
            incumbent_before = best_seen
            if current_prior is not None and getattr(current_prior, "dim", lambda: 0)() == 3:
                if not (_learned_router_enabled(args) and guidance_head_counts is not None):
                    guidance_head_counts = utils.head_counts_for_router(
                        args, int(current_prior.shape[0]), args.n_ants, head_router
                    )
                costs, flats, _, _, _, costs_raw, _, new_edges, survival = aco.sample_mixed_priors(
                    current_prior,
                    require_prob=False,
                    parallel_traced=True,
                    head_counts=guidance_head_counts,
                )
            else:
                costs, flats, _, _, _, costs_raw, _, new_edges, survival = aco.sample(
                    prior=current_prior, require_prob=False
                )
            if costs_raw is not None:
                costs_before_ls = np.asarray(costs_raw, dtype=np.float32)
            else:
                costs_before_ls = np.asarray(costs, dtype=np.float32)
            
            if collect_metrics:
                if new_edges is not None:
                    metrics_log['new_edges'].append(new_edges.astype(np.float32).mean())
                if survival is not None:
                    metrics_log['survival'].append(survival.mean().item())

            best_idx = np.argmin(costs)
            best_val = costs[best_idx]
            best_seen = min(best_seen, best_val)
            best_seen_before_ls = min(best_seen_before_ls, float(costs_before_ls.min()))
            if current_prior is not None and getattr(current_prior, "dim", lambda: 0)() == 3:
                costs_np = np.asarray(costs, dtype=np.float32)
                head_costs = np.split(costs_np, np.cumsum(guidance_head_counts)[:-1])
                head_best_values = [float(x.min()) for x in head_costs]
                selected_head = int(np.argmin(head_best_values))
                for h, (count, vals) in enumerate(zip(guidance_head_counts, head_costs)):
                    head_best = float(vals.min())
                    metrics_log.setdefault(f"head_count_{h}", []).append(float(count))
                    metrics_log.setdefault(f"head_mean_cost_{h}", []).append(float(vals.mean()))
                    metrics_log.setdefault(f"head_best_cost_{h}", []).append(head_best)
                    metrics_log.setdefault(f"head_win_{h}", []).append(1.0 if h == selected_head else 0.0)
                    if np.isfinite(incumbent_before):
                        metrics_log.setdefault(f"head_improvement_{h}", []).append(
                            float(incumbent_before - head_best)
                        )
                utils.update_head_router(
                    head_router,
                    guidance_head_counts,
                    costs,
                    incumbent_before=incumbent_before,
                    best_idx=int(best_idx),
                )
                router_metrics = utils.head_router_metrics(head_router, guidance_head_counts)
                for key, value in router_metrics.items():
                    if not isinstance(value, list):
                        metrics_log.setdefault(key, []).append(float(value))
            
            aco.update_pheromone(flats[best_idx], best_val)
    
    avg_cost = float(np.mean(costs))
    avg_cost_before_ls = float(np.mean(costs_before_ls))
    metrics_log['average_before_ls'] = [avg_cost_before_ls]
    metrics_log['best_before_ls'] = [float(best_seen_before_ls)]
    
    # Get timings
    timings = {}
    if hasattr(aco, "get_timings"):
        t = aco.get_timings()
        timings = {k: v / 1000.0 for k, v in t.items()}

    return avg_cost, best_seen, timings, metrics_log

# =============================================================================
# VALIDATION
# =============================================================================

def validation(
    net: Net,
    val_dataset: List[Any],
    args: argparse.Namespace,
    baseline_values: Optional[np.ndarray] = None
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Run validation on a dataset.
    
    Args:
        net: Neural network model
        val_dataset: Validation dataset
        args: Arguments
        baseline_values: Optional baseline costs for gap calculation
    
    Returns:
        Tuple of (avg_last, avg_best, avg_gap, aggregated_metrics)
    """
    logger = get_logger()
    logger.debug(f"Validating on {len(val_dataset)} instances...")
    
    # Create validation args with overrides
    val_args = argparse.Namespace(**vars(args))
    if val_args.val_H is not None:
        val_args.H = val_args.val_H
    if val_args.val_mini_H is not None:
        val_args.mini_H = val_args.val_mini_H

    # Pure MFACO Caching
    if net is None:
        cached_res = utils.load_pure_mfaco_cache(val_args, val_dataset)
        if cached_res is not None:
            logger.debug("Loaded Pure MFACO result from cache.")
            return (
                cached_res['avg_last'], 
                cached_res['avg_best'], 
                cached_res['avg_gap'], 
                cached_res['metrics']
            )
    
    if net is not None:
        net.eval()
    sum_sample_best = 0.0
    sum_aco_best = 0.0
    sum_gap = 0.0
    n_val = len(val_dataset)
    
    iterable = val_dataset
    
    agg_metrics: Dict[str, List[float]] = {}
    
    all_aco_best = []
    
    for idx, item in enumerate(tqdm(iterable, desc="Validating", leave=False)):
        # Preprocess item based on problem type
        item = _preprocess_val_item(item, args.problem)
        
        dynamic = not args.no_dynamic_feats
        avg, best, timings, metrics = infer_instance(
            net, item, args.k_sparse, args.n_ants, dynamic, val_args,
            collect_metrics=not args.simple_train
        )
        
        sum_sample_best += avg
        sum_aco_best += best
        all_aco_best.append(best)
        
        if baseline_values is not None:
            opt = float(baseline_values[idx])
            gap = (best - opt) / opt * 100
            sum_gap += gap
        
        # Aggregate metrics
        for k, v in metrics.items():
            if k not in agg_metrics:
                agg_metrics[k] = []
            if len(v) > 0:
                agg_metrics[k].append(np.mean(v))
    
    avg_last = sum_sample_best / n_val
    avg_aco_best = sum_aco_best / n_val
    avg_gap = sum_gap / n_val if baseline_values is not None else 0.0
    
    out_metrics = {k: float(np.mean(v)) for k, v in agg_metrics.items() if len(v) > 0}
    
    if net is None:
        cache_data = {
            'avg_last': float(avg_last),
            'avg_best': float(avg_aco_best),
            'avg_gap': float(avg_gap),
            'costs': [float(x) for x in all_aco_best],
            'metrics': out_metrics
        }
        utils.save_pure_mfaco_cache(val_args, val_dataset, cache_data)
        logger.debug("Saved Pure MFACO result to cache.")

    return avg_last, avg_aco_best, avg_gap, out_metrics


def _preprocess_val_item(item: Any, problem: str) -> Any:
    """Preprocess validation item to standard format."""
    if problem == 'cvrp':
        if isinstance(item, (tuple, list)):
            item = [item[0], item[1], item[2]]

        def _unbatch_if_needed(x: Any) -> Any:
            if not torch.is_tensor(x):
                return x
            if x.dim() >= 2 and x.shape[0] == 1:
                return x[0]
            return x

        item = [_unbatch_if_needed(x) for x in item]
        if torch.is_tensor(item[0]):
            item[0] = item[0].numpy()
        if torch.is_tensor(item[1]):
            item[1] = item[1].numpy()
        if torch.is_tensor(item[2]):
            item[2] = float(item[2])
            
        # Normalize demand if capacity > 1.0 (indicating raw data)
        # Training data is already normalized (cap=1.0)
        capacity = float(item[2])
        if capacity > 1.0 + 1e-6:
             item[1] = item[1] / capacity
             item[2] = 1.0
    else:  # tsp
        if isinstance(item, (tuple, list)):
            item = item[0]
    
    return item


# =============================================================================
# EXTENDED PROBLEM TRAINING (BPP / MKP / OP)
# =============================================================================

def _train_extended_instance_reinforce(
    model: Any,
    optimizer: torch.optim.Optimizer,
    instance_data: Any,
    args: argparse.Namespace,
) -> Tuple[float, float, Dict[str, float]]:
    """Train on a single extended-problem instance using REINFORCE."""
    model.train()

    data = extract_extended_problem_data(args.problem, instance_data)
    aco, pyg_args = setup_extended_aco(args, instance_data, args.problem, heuristic=None)
    expected_edge_feats = get_extended_model_edge_feats(args)
    dynamic_graph = use_extended_dynamic_edge_features(args)

    static_pyg_data = None
    if args.static_prior:
        static_pyg_data = build_extended_pyg_data(aco, args.problem, *pyg_args, dynamic=False)
        static_pyg_data = align_extended_edge_attr_width(static_pyg_data, expected_edge_feats)

    best_seen = float("-inf")
    avg_cost_last = None
    metrics = MetricsCollector()
    t_neural_total = 0.0
    t_aco_total = 0.0

    outer_pbar = tqdm(range(args.H), desc="Outer", leave=False)
    for outer in outer_pbar:
        t0 = time.time()
        if args.static_prior:
            pyg_data = static_pyg_data
        else:
            pyg_data = build_extended_pyg_data(aco, args.problem, *pyg_args, dynamic=dynamic_graph)
            pyg_data = align_extended_edge_attr_width(pyg_data, expected_edge_feats)

        prior_output = model(pyg_data)
        prior = reshape_extended_prior_output(
            prior_output,
            args.problem,
            **get_extended_prior_kwargs(args.problem, data),
        )
        t_neural_total += time.time() - t0

        all_logp_sums = []
        all_objectives = []

        for inner in range(args.mini_H):
            t_aco_start_inner = time.time()
            current_prior = prior
            if args.train_anneal:
                factor = compute_annealing_factor(inner, args.mini_H, args.gamma, args.min_gamma)
                current_prior = prior * factor

            aco.heuristic = current_prior.to(device=args.device, dtype=torch.float32)
            aco._sync_cpp_inputs()

            costs, paths, logps, _ = aco.sample(require_prob=True, prior=None, parallel_traced=True)
            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            objective_t = extended_raw_values_to_objective(costs_t, args.problem)
            best_cost_obj = select_extended_best_value(costs_t, args.problem)
            best_seen = max(best_seen, best_cost_obj)

            with torch.no_grad():
                aco.update_pheromone(paths, costs)

            avg_cost_last = float(objective_t.mean().item())
            t_aco_total += time.time() - t_aco_start_inner

            if outer >= args.warmup:
                if args.static_prior:
                    logp_steps_new = aco._replay_logp_from_paths(
                        paths,
                        current_prior,
                        pheromone=aco.pheromone,
                        heuristic=current_prior,
                    )
                    if logp_steps_new.size(0) > 0:
                        logp_sum = logp_steps_new.sum(dim=0)
                    else:
                        logp_sum = torch.zeros((paths.size(1),), device=args.device)
                else:
                    logp_sum = logps.sum(dim=0)

                all_logp_sums.append(logp_sum)
                all_objectives.append(objective_t)

        if outer >= args.warmup and all_logp_sums:
            logp_sums_t = torch.cat(all_logp_sums)
            objectives_t = torch.cat(all_objectives)
            baseline = objectives_t.mean()
            adv = (objectives_t - baseline).detach()
            loss = -torch.sum(adv * logp_sums_t) / (aco.n_ants * args.mini_H)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics.add("loss", loss.item())
            metrics.add("grad_norm", grad_norm.item())
            metrics.add("prior_mean", prior.mean().item())
            metrics.add("prior_std", prior.std().item())
            outer_pbar.set_postfix({
                "obj": f"{best_seen:.4f}",
                "avg": f"{avg_cost_last:.4f}",
                "loss": f"{loss.item():.4f}",
            })

    out_metrics = metrics.get_all_means()
    out_metrics["time_neural"] = t_neural_total
    out_metrics["time_aco"] = t_aco_total
    return avg_cost_last, best_seen, out_metrics


def _run_extended_validation_epoch(
    model: Any,
    val_instances: List[Any],
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    """Run the extended validation loop and compare guided vs unguided ACO."""
    val_with_prior = []
    val_without_prior = []
    dynamic_graph = use_extended_dynamic_edge_features(args)
    expected_edge_feats = get_extended_model_edge_feats(args)

    for instance_data in val_instances:
        data = extract_extended_problem_data(args.problem, instance_data)
        aco, pyg_args = setup_extended_aco(args, instance_data, args.problem, heuristic=None)

        model.eval()
        with torch.no_grad():
            if args.static_prior:
                pyg_data = build_extended_pyg_data(aco, args.problem, *pyg_args, dynamic=False)
                pyg_data = align_extended_edge_attr_width(pyg_data, expected_edge_feats)
                prior_output = model(pyg_data)
                prior = reshape_extended_prior_output(
                    prior_output,
                    args.problem,
                    **get_extended_prior_kwargs(args.problem, data),
                )
                aco.heuristic = prior.to(device=args.device, dtype=torch.float32)
                aco._sync_cpp_inputs()

            for outer in range(args.H):
                if not args.static_prior:
                    pyg_data = build_extended_pyg_data(aco, args.problem, *pyg_args, dynamic=dynamic_graph)
                    pyg_data = align_extended_edge_attr_width(pyg_data, expected_edge_feats)
                    prior_output = model(pyg_data)
                    prior = reshape_extended_prior_output(
                        prior_output,
                        args.problem,
                        **get_extended_prior_kwargs(args.problem, data),
                    )

                for inner in range(args.mini_H):
                    current_prior = prior
                    if args.train_anneal:
                        factor = compute_annealing_factor(inner, args.mini_H, args.gamma, args.min_gamma)
                        current_prior = prior * factor

                    aco.heuristic = current_prior.to(device=args.device, dtype=torch.float32)
                    aco._sync_cpp_inputs()
                    costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
                    with torch.no_grad():
                        aco.update_pheromone(paths, costs)

            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            val_with_prior.append(select_extended_best_value(costs_t, args.problem))

        aco_baseline, _ = setup_extended_aco(args, instance_data, args.problem)
        costs_baseline, _, _, _ = aco_baseline.sample(require_prob=False, prior=None, parallel_traced=True)
        costs_baseline_t = torch.as_tensor(costs_baseline, device=args.device, dtype=torch.float32)
        val_without_prior.append(select_extended_best_value(costs_baseline_t, args.problem))
        model.train()

    avg_val_with_prior = float(np.mean(val_with_prior))
    avg_val_without_prior = float(np.mean(val_without_prior))
    improvement = compute_extended_relative_improvement(
        avg_val_with_prior,
        avg_val_without_prior,
        args.problem,
        objective_mode=True,
    )
    return avg_val_with_prior, avg_val_without_prior, 0.0 if improvement is None else float(improvement)


def _train_extended_main(args: argparse.Namespace):
    """Main training loop for BPP, MKP, and OP."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model_name = build_extended_model_name(args)

    if args.wandb_project and not args.no_wandb:
        run_id = wandb.util.generate_id()
        run_name = args.run_name if args.run_name else model_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=run_id,
            group=args.wandb_group,
            config=_serializable_args(args),
            mode="online" if not args.dry_run else "disabled",
        )

    edge_feats = get_extended_model_edge_feats(args)
    model = create_extended_model(
        args.problem,
        m=args.m,
        edge_feats=edge_feats,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    val_instances = []
    torch.manual_seed(args.seed + 1000)
    np.random.seed(args.seed + 1000)
    for _ in range(args.val_size):
        val_instances.append(utils.get_problem_data(args.problem, args.n_node, args.device, args.k_sparse, m=args.m))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_costs = []

        for step in range(args.steps_per_epoch):
            instance_data = utils.get_problem_data(args.problem, args.n_node, args.device, args.k_sparse, m=args.m)
            avg_cost, best_cost, metrics = _train_extended_instance_reinforce(model, optimizer, instance_data, args)
            train_costs.append(avg_cost)

            if args.wandb_project and not args.no_wandb:
                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "train_cost": avg_cost,
                    "best_cost": best_cost,
                    **metrics,
                })

        avg_val_with_prior, avg_val_without_prior, improvement = _run_extended_validation_epoch(model, val_instances, args)
        print(f"  Train Cost: {np.mean(train_costs):.4f}")
        print(f"  Val Cost (with prior): {avg_val_with_prior:.4f}")
        print(f"  Val Cost (without prior): {avg_val_without_prior:.4f}")
        print(f"  Improvement: {improvement:.2f}%")

        if args.wandb_project and not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_cost": np.mean(train_costs),
                "val_cost_with_prior": avg_val_with_prior,
                "val_cost_without_prior": avg_val_without_prior,
                "improvement": improvement,
            })

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.save_dir) / f"{model_name}_epoch{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_cost": np.mean(train_costs),
                "val_cost_with_prior": avg_val_with_prior,
                "val_cost_without_prior": avg_val_without_prior,
                "improvement": improvement,
                "config": _serializable_args(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    final_path = Path(args.save_dir) / f"{model_name}_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": _serializable_args(args),
    }, final_path)
    print(f"\nSaved final model: {final_path}")

    if args.wandb_project and not args.no_wandb:
        wandb.finish()


def _parse_extended_train_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse training arguments for BPP/MKP/OP."""
    parser = argparse.ArgumentParser(description="Train NGFACO for BPP, MKP, OP", allow_abbrev=False)

    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")

    add_extended_problem_args(parser)
    parser.add_argument("--H", type=int, default=5, help="Number of outer iterations (replaces T)")
    parser.add_argument("--mini_H", type=int, default=5, help="Number of inner iterations")
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants")
    parser.add_argument("--k_sparse", type=int, default=32, help="K-NN size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=64, help="Steps per epoch")
    parser.add_argument("--val_size", type=int, default=16, help="Validation size")
    parser.add_argument("--aco_iters_per_sample", type=int, default=20, help="ACO iterations per sample")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rho", type=float, default=0.9, help="Pheromone retention")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Heuristic weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Prior scaling")
    parser.add_argument("--min_gamma", type=float, default=0.0, help="Minimum prior scaling")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup steps")
    parser.add_argument("--train_anneal", action="store_true", help="Use annealing")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="Disable dynamic features")
    parser.add_argument("--smallvram", action="store_true", help="Use small VRAM mode")
    parser.add_argument(
        "--static_prior",
        action="store_true",
        help="Train a DeepACO-style static-prior baseline (1 edge feature, no dynamic prior)",
    )
    parser.add_argument("--wandb_project", type=str, default="ngfaco_extended", help="WandB project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints_extended", help="Save directory")
    parser.add_argument("--save_interval", type=int, default=1, help="Save interval")
    parser.add_argument("--dry_run", action="store_true", help="Dry run")
    return parser.parse_args(argv)

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def _parse_base_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the original TSP/CVRP training flow."""
    parser = argparse.ArgumentParser(
        description="Train neural-guided ACO for TSP/CVRP",
        allow_abbrev=False,
    )

    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")

    # Problem configuration
    parser.add_argument("--problem", type=str, default=None,
                        choices=['tsp', 'cvrp'])
    parser.add_argument("--n_node", type=int, default=1000)
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--algo", choices=["reinforce", "ppo"], default="ppo")
    parser.add_argument("--alg", choices=["faco", "mmas"], default="faco",
                        help="Algorithm type")

    # PPO hyperparameters
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_clip", type=float, default=0.1)
    parser.add_argument("--no_adv_norm", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    
    # Training configuration
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ppo_lr", type=float, default=5e-6)
    parser.add_argument("--reinforce_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # ACO configuration
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--min_new_edges", type=int, default=None)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_smooth_mmas", action="store_true")
    parser.add_argument("--no_extend_ls", action="store_true")
    parser.add_argument("--ls_scope", choices=["localized", "global"], default="localized")
    parser.add_argument("--ls_budget", choices=["truncated", "full"], default="truncated")
    parser.add_argument("--ls_max_opt", type=int, default=0,
                        help="Maximum accepted LS improving moves per call; 0 uses floor(N/4)")
    parser.add_argument("--no_normalized_heuristic", action="store_true")
    
    # Ablation studies
    parser.add_argument("--ablation_pheromone_features", action="store_true",
                        help="Remove pheromone-based features (tau_cv, log_tau_rel)")
    parser.add_argument("--ablation_incumbent_features", action="store_true", 
                        help="Remove incumbent-based features (source_succ, source_pred, new_edge)")
    parser.add_argument(
        "--edge_feature_set",
        "--edge-feature-set",
        choices=["full", "compact3"],
        default="full",
        help="Edge feature layout: full=6 channels, compact3=dist_norm/log_tau_rel/is_in_incumbent",
    )


    # Traced sampling parallelism
    # CVRP wrapper defaults to parallel_traced=False (single-thread traced sampling) unless passed explicitly.
    pt_group = parser.add_mutually_exclusive_group()
    pt_group.add_argument(
        "--parallel_traced",
        dest="parallel_traced",
        action="store_true",
        help="Parallelize traced sampling across ants (faster; different RNG usage)",
    )
    pt_group.add_argument(
        "--no_parallel_traced",
        dest="parallel_traced",
        action="store_false",
        help="Force single-thread traced sampling (slower; legacy behavior)",
    )
    parser.set_defaults(parallel_traced=True)

    # Optimization

    
    # Neural Local Search
    parser.add_argument("--nls", action="store_true",
                        help="Enable Neural Local Search")
    parser.add_argument("--nls_beta", type=float, default=0.2, help="Weight for post-LS cost in advantage")
    parser.add_argument("--T_nls", type=int, default=5, help="Number of NLS iterations")
    parser.add_argument("--no_logit_net", action="store_true")
    parser.add_argument("--smallvram", action="store_true", help="Fix OOM mathematically perfectly via graph detachment inner loop")
    parser.add_argument("--grad_checkpointing", action="store_true", help="Enable PyTorch gradient checkpointing to save VRAM at the cost of compute")

    # Multi-head prediction
    parser.add_argument("--multi_head", action="store_true",
                        help="Use a Multi-Prediction style K-head edge-prior decoder")
    parser.add_argument("--num_heads", type=int, default=1,
                        help="Number of prediction heads; values >1 enable multi-head training")
    parser.add_argument("--head_zdim", type=int, default=16,
                        help="Head-code width for --head_decoder_type lowrank; unused by the LoRA decoder")
    parser.add_argument("--head_decoder_type", "--head-decoder-type", dest="head_decoder_type",
                        choices=["lora", "lowrank"], default="lora",
                        help="Multi-head decoder type: true LoRA final layer, or legacy head-code low-rank residual")
    parser.add_argument("--lora_rank", "--lora-rank", dest="lora_rank", type=int, default=8,
                        help="Rank of each head-specific LoRA adapter in the multi-head decoder")
    parser.add_argument("--lora_alpha", "--lora-alpha", dest="lora_alpha", type=float, default=1.0,
                        help="LoRA scaling alpha; the adapter scale is alpha / lora_rank")
    parser.add_argument("--freeze_lora_base", "--freeze-lora-base", dest="freeze_lora_base", action="store_true",
                        help="Freeze the shared final decoder linear layer and train only LoRA adapters there")
    parser.add_argument("--log_best_head", "--log-best-head", dest="log_best_head", action="store_true",
                        help="Log expensive multi-head diagnostics, including JS diversity, to WandB")
    parser.add_argument("--loss_js", "--loss-js", dest="loss_js", type=float, default=0.0,
                        help="Coefficient for Jensen-to-mean PolyNet head-diversity reward; 0 disables it")
    parser.add_argument("--head_ant_weights", type=str, default=None,
                        help="Comma-delimited ant allocation weights/counts for PolyNet ant groups, e.g. 50,20,15,15")
    parser.add_argument("--head_router", choices=["static", "ema", "learned"], default="static",
                        help="Ant routing for PolyNet ant groups; static preserves head_ant_weights")
    parser.add_argument("--head_router_alpha", type=float, default=0.25,
                        help="EMA update rate for adaptive head utility routing")
    parser.add_argument("--head_router_min_frac", type=float, default=0.0,
                        help="Minimum allocation fraction blended into each head by adaptive routing")
    parser.add_argument("--allocator_loss_coef", type=float, default=1.0,
                        help="Loss coefficient for learned PolyNet ant-allocation policy")
    parser.add_argument("--allocator_entropy_coef", type=float, default=0.01,
                        help="Entropy bonus coefficient for learned ant allocation")
    parser.add_argument("--allocator_temperature", type=float, default=1.0,
                        help="Softmax temperature for learned ant allocation")
    parser.add_argument(
        "--head_input_transform",
        "--head-input-transform",
        choices=["none", "d4"],
        default="none",
        help="Apply a deterministic distance-preserving coordinate transform per head before encoding",
    )
    parser.add_argument("--init_single_head_checkpoint", type=str, default=None,
                        help="Initialize a multi-head decoder from a trained single-head checkpoint")
    
    # Wandb settings and output
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dynaco")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Optional suffix for checkpoint filenames")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    
    # Baseline configuration
    parser.add_argument("--baseline", type=str, default='default')
    parser.add_argument("--baseline_runs", type=int, default=1)
    parser.add_argument("--baseline_time_limit", type=float, default=0.5)
    
    # Validation configuration
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to validation dataset")
    parser.add_argument("--val_size", type=int, default=16,
                        help="Limit validation set size")
    parser.add_argument("--generate_val", action="store_true",
                        help="Generate validation set")
    parser.add_argument("--save_generated", type=str, default=None,
                        help="Path to save generated validation dataset")
    parser.add_argument("--capacity_override", type=float, default=None,
                        help="Override CVRP capacity during train/validation generation")
    parser.add_argument("--val_H", type=int, default=None)
    parser.add_argument("--val_mini_H", type=int, default=None)

    
    # Warmup and annealing
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Use warmup strategy in validation")
    parser.add_argument("--no-warmup", "--no_warmup", dest="warmup", action="store_false")
    parser.add_argument("--train_warmup", action="store_true",
                        help="Use warmup strategy in training")
    parser.add_argument("--warmup_ratio", type=float, default=0.5)
    parser.add_argument("--train_anneal", action="store_true",
                        help="Enable annealing during training")
    parser.add_argument("--no_anneal", action="store_true",
                        help="Disable annealing during validation")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.0)
    parser.add_argument("--L", type=int, default=0)
    
    # Miscellaneous
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--simple_train", action="store_true",
                        help="Skip expensive metric calculations")
    parser.add_argument("--train_deepaco", action="store_true",
                        help="Disable pheromone updates during training")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. Use without path for auto-detection based on config.")

    return parser.parse_args(argv)


def _normalize_cli_argv(argv: Optional[List[str]] = None) -> List[str]:
    """Return a mutable CLI argument list."""
    if argv is None:
        return list(sys.argv[1:])
    return list(argv)


def _extract_cli_value(argv: List[str], flag: str) -> Optional[str]:
    """Extract a simple --flag value from argv, supporting --flag=value syntax."""
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _infer_train_problem(argv: Optional[List[str]] = None) -> Optional[str]:
    """Infer the requested problem before selecting the parser/implementation."""
    args_list = _normalize_cli_argv(argv)
    return _extract_cli_value(args_list, "--problem")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments for either the base or extended training entrypoint."""
    args_list = _normalize_cli_argv(argv)
    problem = _infer_train_problem(args_list)
    if problem in EXTENDED_PROBLEMS:
        return _parse_extended_train_args(args_list)
    return _parse_base_args(args_list)


def setup_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model_name(args: argparse.Namespace) -> str:
    """Generate descriptive model filename based on parameters."""
    name = (f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}"
            f"_H{args.H}_miniH{args.mini_H}_rho{args.rho}"
            f"_mne{args.min_new_edges}_{args.algo}_lr{args.lr}")
    if args.problem == 'cvrp' and args.capacity_override is not None:
        name += f"_cap{args.capacity_override:g}"
    
    if args.train_anneal:
        name += f"_anneal_g{args.gamma}_mg{args.min_gamma}"
    if args.L > 0:
        name += f"_L{args.L}"
    if args.train_warmup:
        name += f"_warmup{args.warmup_ratio}"
    if args.train_deepaco:
        name += "_deepaco"
    
    # Ablation suffixes
    if args.no_dynamic_feats:
        name += "_static"
    if args.no_smooth_mmas:
        name += "_nosmooth"
    if args.disable_heuristic:
        name += "_noheu"
    if args.no_local_search:
        name += "_nols"
    if args.no_extend_ls:
        name += "_noextls"
    if args.ls_scope != "localized":
        name += f"_ls{args.ls_scope}"
    if args.ls_budget != "truncated":
        name += f"_{args.ls_budget}ls"
    if args.ls_max_opt > 0:
        name += f"_lsmax{args.ls_max_opt}"
    if args.no_normalized_heuristic:
        name += "_nonorm"
    if args.ablation_pheromone_features:
        name += "_noPherFeat"
    if args.ablation_incumbent_features:
        name += "_noIncumbFeat"
    if getattr(args, "edge_feature_set", "full") != "full":
        name += f"_{args.edge_feature_set}"
    if _multi_head_enabled(args):
        name += f"_mh{args.num_heads}_polynet"
        if getattr(args, "head_decoder_type", "lora") == "lora":
            name += f"_lora_r{int(getattr(args, 'lora_rank', 8))}"
        if getattr(args, "head_ant_weights", None):
            alloc = str(args.head_ant_weights).replace(",", "-").replace(" ", "")
            name += f"_ha{alloc}"
        if str(getattr(args, "head_router", "static") or "static").lower() != "static":
            name += f"_hr{getattr(args, 'head_router')}"
            if _learned_router_enabled(args):
                name += f"_alcoef{float(getattr(args, 'allocator_loss_coef', 1.0)):g}"
                name += f"_alent{float(getattr(args, 'allocator_entropy_coef', 0.01)):g}"
                name += f"_alt{float(getattr(args, 'allocator_temperature', 1.0)):g}"
            else:
                name += f"_hra{float(getattr(args, 'head_router_alpha', 0.25)):g}"
        if _head_input_transform_mode(args) != "none":
            name += f"_ht{_head_input_transform_mode(args)}"

    if args.alg == 'mmas':
        name += "_mmas"
    if getattr(args, "run_tag", None):
        tag = str(args.run_tag).strip().replace(" ", "_")
        if tag:
            name += f"_{tag}"

    return name


def load_validation_data(args: argparse.Namespace, logger: Logger):
    """Load or generate validation dataset and baselines."""
    val_dataset = None
    baseline_values = None
    
    if args.generate_val:
        baseline_solver = (args.baseline if args.baseline != 'default' 
                          else ('lkh' if args.problem == 'tsp' else 'hgs'))
        val_dataset = utils.generate_and_save_dataset(
            problem=args.problem,
            n_node=args.n_node,
            n_instances=args.val_size,
            save_path=args.save_generated,
            baseline_solver=baseline_solver,
            baseline_runs=args.baseline_runs,
            time_limit=args.baseline_time_limit,
            device='cpu',
            capacity_override=args.capacity_override if args.problem == 'cvrp' else None,
        )
    elif args.val_dataset:
        logger.info(f"Loading validation dataset from {args.val_dataset}...")
        val_dataset = _load_dataset_from_path(args.val_dataset, args.problem)
    else:
        # Default behavior: load auto dataset (test_set by default for now, can be adjusted)
        # Training typically uses a fixed validation set from file or generates one.
        val_dataset = utils.load_auto_dataset(
            args.n_node, 
            problem=args.problem, 
            data_source='validation_set', # Use validation set (not test set)
            rl_data=False,          # Default assumption unless training needs RL data
            device='cpu'
        )

    # Extract baseline values if embedded in dataset
    baseline_values = _extract_baseline_from_dataset(val_dataset, args.problem)
    
    # Generate fallback dataset if needed
    if val_dataset is None:
        logger.info("Validation dataset not found. Generating 16 instances...")
        val_dataset = _generate_fallback_dataset(args)
        if not args.val_dataset:
            utils.save_val_dataset(val_dataset, args.n_node, problem=args.problem)
    
    # Limit validation set size
    if args.val_size is not None and val_dataset is not None:
        original_len = len(val_dataset)
        val_dataset = val_dataset[:args.val_size]
        if original_len != len(val_dataset):
            logger.info(f"Limited validation dataset from {original_len} to {len(val_dataset)} instances.")
    
    # Compute baselines if needed
    if (baseline_values is None and args.baseline != 'none' 
        and not getattr(args, 'no_baseline', False)):
        baseline_values = _compute_baselines(val_dataset, args)
    
    return val_dataset, baseline_values


def _load_dataset_from_path(path: str, problem: str):
    """Load dataset from file path."""
    if path.endswith(".txt") and problem == 'tsp':
        return utils.load_tsp_txt_dataset(path)
    elif path.endswith(".txt") and problem == 'cvrp':
        return utils.load_cvrp_txt_dataset(path)
    else:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            return data.get("coords", data)
        return data


def _extract_baseline_from_dataset(val_dataset, problem: str):
    """Extract baseline costs if embedded in dataset."""
    if not isinstance(val_dataset, list) or len(val_dataset) == 0:
        return None
    
    try:
        if problem == 'tsp' and isinstance(val_dataset[0], tuple) and len(val_dataset[0]) >= 2:
            costs = [x[1] for x in val_dataset]
            if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) 
                   and c > 1e-6 for c in costs):
                return np.array(costs)
        elif problem == 'cvrp' and isinstance(val_dataset[0], tuple) and len(val_dataset[0]) >= 4:
            costs = [x[3] for x in val_dataset]
            if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) 
                   and c > 1e-6 for c in costs):
                return np.array(costs)
    except Exception:
        pass
    
    return None


def _generate_fallback_dataset(args: argparse.Namespace):
    """Generate fallback validation dataset."""
    val_dataset = []
    gen_fn = (utils.generate_tsp_instance if args.problem == 'tsp' 
              else utils.gen_cvrp_instance)
    
    for _ in range(16):
        if args.problem == 'tsp':
            val_dataset.append(torch.from_numpy(gen_fn(args.n_node)))
        else:
            c, d, cap = gen_fn(args.n_node, device='cpu', capacity=args.capacity_override)
            val_dataset.append((c.cpu(), d.cpu(), cap))
    
    return val_dataset


def _compute_baselines(val_dataset, args: argparse.Namespace):
    """Compute baseline values for validation dataset."""
    logger = get_logger()
    logger.info("Computing baseline values...")
    
    # Extract coords if dataset contains tuples
    if (args.problem == 'tsp' and isinstance(val_dataset, list) 
        and len(val_dataset) > 0 and isinstance(val_dataset[0], tuple)):
        val_dataset_coords = [x[0] for x in val_dataset]
        return get_baseline(
            val_dataset_coords, problem=args.problem, n_node=args.n_node,
            runs=args.baseline_runs, time_limit=args.baseline_time_limit
        )
    
    return get_baseline(
        val_dataset, problem=args.problem, n_node=args.n_node,
        runs=args.baseline_runs, time_limit=args.baseline_time_limit
    )


def save_checkpoint(
    net: Net,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    save_path: Path,
    val_cost: Optional[float] = None,
    val_gap: Optional[float] = None
):
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": _serializable_args(args)
    }
    if val_cost is not None:
        checkpoint["val_cost"] = val_cost
    if val_gap is not None:
        checkpoint["val_gap"] = val_gap
    
    torch.save(checkpoint, save_path)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(argv: Optional[List[str]] = None):
    """Main training entry point."""
    args_list = _normalize_cli_argv(argv)
    problem = _infer_train_problem(args_list)
    if problem in EXTENDED_PROBLEMS:
        args = _parse_extended_train_args(args_list)
        return _train_extended_main(args)

    args = _parse_base_args(args_list)

    def _cli_has_flag(key: str) -> bool:
        flag = "--" + key
        flag_hyphen = "--" + key.replace("_", "-")
        return any(
            token == flag
            or token == flag_hyphen
            or token.startswith(f"{flag}=")
            or token.startswith(f"{flag_hyphen}=")
            for token in args_list
        )

    # Load configuration from YAML if provided
    yaml_config_keys = set()
    if hasattr(args, 'config') and args.config:
        print(f"Loading configuration from YAML: {args.config}")
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f) or {}
        yaml_config_keys = set(yaml_config.keys())

        legacy_ignored_config_keys = {
            "head_deploy",
            "head_index",
            "head_training",
            "head_gamma",
            "head_score_mode",
            "head_topq",
            "head_diversity_coef",
            "head_complement_coef",
            "head_anchor_checkpoint",
            "head_anchor_coef",
            "head_router_gamma",
            "head_router_score_mode",
        }

        # Override parser defaults with YAML values unless the CLI explicitly set the flag.
        for key, value in yaml_config.items():
            if key in legacy_ignored_config_keys:
                continue
            if not hasattr(args, key):
                setattr(args, key, value)
            elif not _cli_has_flag(key):
                setattr(args, key, value)

    if args.problem is None:
        raise ValueError("Problem must be specified via --problem or the YAML config")
    if (
        _multi_head_enabled(args)
        and getattr(args, "head_decoder_type", "lora") == "lowrank"
        and not _cli_has_flag("lora_rank")
        and "lora_rank" not in yaml_config_keys
    ):
        args.lora_rank = 16

    # Load configuration from checkpoint if resuming with a specific path
    if args.resume and args.resume != "auto" and os.path.isfile(args.resume):
        print(f"Loading configuration from checkpoint: {args.resume}")
        try:
            # Load checkpoint on CPU to just get config
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                
                # Keys to preserve from current command line (do not overwrite with saved config)
                preserve_keys = {
                    'resume', 'epochs', 'device', 'save_dir', 
                    'no_wandb', 'wandb_entity', 'wandb_project', 'run_name', 
                    'threads', 'val_dataset', 'generate_val', 'save_generated'
                }
                
                # Update args with saved config
                update_count = 0
                for k, v in saved_config.items():
                    if k not in preserve_keys and hasattr(args, k):
                        setattr(args, k, v)
                        update_count += 1
                
                print(f"Restored {update_count} arguments from checkpoint configuration.")
                
        except Exception as e:
            print(f"Warning: Failed to load configuration from checkpoint: {e}")
            print("Continuing with command line arguments.")
    



    import functools
    
    # Setup defaults
    utils.set_seed(args.seed)
    
    if args.lr is None:
        args.lr = args.ppo_lr if args.algo == 'ppo' else args.reinforce_lr
    if _multi_head_enabled(args) and args.algo != "ppo":
        raise ValueError("Multi-head training is currently implemented for --algo ppo")
    
    if args.baseline == 'default':
        args.baseline = 'lkh' if args.problem == 'tsp' else 'hgs'
    
    # Auto-set min_new_edges for CVRP based on capacity
    if args.problem == 'cvrp' and args.min_new_edges is None:
        # Use same capacity logic as utils.gen_cvrp_instance
        if args.capacity_override is not None:
            capacity = float(args.capacity_override)
        elif args.n_node >= 50000:
            capacity = 2000
        elif args.n_node >= 10000:
            capacity = 1000
        elif args.n_node >= 5000:
            capacity = 500
        elif args.n_node >= 1000:
            capacity = 250
        else:
            capacity = 50
        args.min_new_edges = int(capacity / 20)
        print(f"Auto-set min_new_edges to {args.min_new_edges} (capacity={capacity}, capacity/20={capacity/20})")
    elif args.min_new_edges is None:
        # Default for TSP
        args.min_new_edges = 12
        
    if args.threads is None:
        args.threads = psutil.cpu_count(logical=False)
    faco.set_faco_cpp_threads(args.threads)

    # Set seeds
    setup_seeds(args.seed)

    # Initialize logger
    logger = init_logger(
        use_wandb=not args.no_wandb,
        log_dir=Path(args.save_dir) / "logs" if args.save_dir else None,
        verbose=True
    )

    # Build model name
    model_name = build_model_name(args)

    # Create save directory
    save_dir = Path(args.save_dir) / args.problem / f"n{args.n_node}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # NEW: Save experiment configuration if running under experiment wrapper
    if os.getenv("NGFACO_EXPERIMENT_DIR"):
        exp_dir = Path(os.getenv("NGFACO_EXPERIMENT_DIR"))
        save_experiment_config(args, exp_dir)

    # Initialize wandb
    run_id = wandb.util.generate_id()
    if not args.no_wandb:
        run_name = args.run_name if args.run_name else model_name
        wandb_project = args.wandb_project
        wandb_group = args.wandb_group
        if wandb_group is None:
            wandb_group = f"{args.problem}_n{args.n_node}_ls{args.ls_scope}_{args.ls_budget}"
        wandb.init(
            project=wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=run_id,
            group=wandb_group,
            config=_serializable_args(args)
        )

    # Initialize model
    feats = 2 if args.problem == 'tsp' else 4
    
    if args.edge_feature_set == "compact3":
        edge_feats = 3
    else:
        full_edge_feats = 6
        minus_feats = 0
        if args.ablation_pheromone_features:
            minus_feats += 2
        if args.ablation_incumbent_features:
            minus_feats += 3
        edge_feats = full_edge_feats - minus_feats
    train_epoch_fn = train_epoch
    validation_fn = validation

    print(
        f"Using {edge_feats} edge features "
        f"(edge_feature_set={args.edge_feature_set}, "
        f"ablation_pheromone={args.ablation_pheromone_features}, "
        f"ablation_incumbent={args.ablation_incumbent_features})"
    )
    model_cls = MultiHeadNet if _multi_head_enabled(args) else Net
    model_kwargs = {}
    if _multi_head_enabled(args):
        args.multi_head = True
        model_kwargs.update(
            num_heads=args.num_heads,
            head_zdim=args.head_zdim,
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_lora_base=args.freeze_lora_base,
            head_decoder_type=args.head_decoder_type,
        )
        print(
            f"Using {args.head_decoder_type} multi-head decoder: heads={args.num_heads}, rank={args.lora_rank}, ant-group routing"
        )

    net_model = model_cls(
        feats=feats,
        edge_feats=edge_feats,
        logit_net=not args.no_logit_net,
        grad_checkpointing=getattr(args, 'grad_checkpointing', False),
        **model_kwargs,
    ).to(args.device)

    if _multi_head_enabled(args) and getattr(args, "init_single_head_checkpoint", None):
        _copy_single_head_weights_into_multi_head(
            net_model,
            args.init_single_head_checkpoint,
            args.device,
        )
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr)

    # Count model parameters
    total_params = sum(p.numel() for p in net_model.parameters())
    trainable_params = sum(p.numel() for p in net_model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Log memory usage only for CUDA devices
    initial_memory = _gpu_memory_allocated_gb(args.device)
    if initial_memory is not None:
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {initial_memory:.2f} GB")

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        import glob
        
        checkpoint_path = None
        if args.resume == "auto":
            # Auto-detect checkpoint based on current configuration
            expected_checkpoint = save_dir / f"{model_name}_last.pt"
            if expected_checkpoint.exists():
                checkpoint_path = expected_checkpoint
                print(f"Auto-detected checkpoint: {checkpoint_path}")
            else:
                # Try to find any matching checkpoint with the model name prefix
                pattern = str(save_dir / f"{model_name}_*.pt")
                matches = glob.glob(pattern)
                if matches:
                    # Prefer "last" over "best", then "latest", then others
                    for suffix in ["_last.pt", "_latest.pt", "_best.pt"]:
                        for match in matches:
                            if match.endswith(suffix):
                                checkpoint_path = Path(match)
                                print(f"Auto-detected checkpoint: {checkpoint_path}")
                                break
                        if checkpoint_path:
                            break
                    
                    if not checkpoint_path:
                        checkpoint_path = Path(matches[0])
                        print(f"Auto-detected checkpoint: {checkpoint_path}")
                else:
                    print(f"Warning: No checkpoint found matching {model_name}, starting from scratch")
        else:
            # Explicit checkpoint path provided
            checkpoint_path = Path(args.resume)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if checkpoint_path and checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
            
            # Infer input feature sizes from checkpoint model weights for backward compatibility
            state_dict = checkpoint["model_state_dict"]
            if (
                _multi_head_enabled(args)
                and "par_net_heu.head_proj.weight" in state_dict
                and not _cli_has_flag("head_decoder_type")
            ):
                args.head_decoder_type = "lowrank"
                model_kwargs.update(head_decoder_type="lowrank")
                args.lora_rank = int(state_dict["par_net_heu.head_proj.weight"].shape[0])
                model_kwargs.update(rank=args.lora_rank)
                if "head_codes" in state_dict:
                    args.head_zdim = int(state_dict["head_codes"].shape[1])
                    model_kwargs.update(head_zdim=args.head_zdim)
                print("Checkpoint uses legacy head-code decoder; recreating model with head_decoder_type=lowrank")
                net_model = model_cls(
                    feats=feats,
                    edge_feats=edge_feats,
                    logit_net=not args.no_logit_net,
                    grad_checkpointing=getattr(args, 'grad_checkpointing', False),
                    **model_kwargs,
                ).to(args.device)
                optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr)
            if _multi_head_enabled(args) and "par_net_heu.out.lora_A" in state_dict:
                ckpt_lora_a = state_dict["par_net_heu.out.lora_A"]
                ckpt_heads = int(ckpt_lora_a.shape[0])
                ckpt_rank = int(ckpt_lora_a.shape[1])
                if ckpt_heads != args.num_heads or ckpt_rank != args.lora_rank:
                    print(
                        f"Checkpoint LoRA shape heads={ckpt_heads}, rank={ckpt_rank} "
                        f"differs from current heads={args.num_heads}, rank={args.lora_rank}. "
                        "Recreating model..."
                    )
                    args.num_heads = ckpt_heads
                    args.lora_rank = ckpt_rank
                    model_kwargs.update(num_heads=ckpt_heads, rank=ckpt_rank)
                    net_model = model_cls(
                        feats=feats,
                        edge_feats=edge_feats,
                        logit_net=not args.no_logit_net,
                        grad_checkpointing=getattr(args, 'grad_checkpointing', False),
                        **model_kwargs,
                    ).to(args.device)
                    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr)

            if "emb_net.v_lin0.weight" in state_dict:
                ckpt_feats = state_dict["emb_net.v_lin0.weight"].shape[1]
                if ckpt_feats != feats:
                    print(f"Checkpoint feats={ckpt_feats} differs from current={feats}. Recreating model...")
                    feats = ckpt_feats
                    net_model = model_cls(
                        feats=feats,
                        edge_feats=edge_feats,
                        logit_net=not args.no_logit_net,
                        **model_kwargs,
                    ).to(args.device)
                    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr)

            if "emb_net.e_lin0.weight" in state_dict:
                ckpt_edge_feats = state_dict["emb_net.e_lin0.weight"].shape[1]
                if ckpt_edge_feats != edge_feats:
                    print(f"Checkpoint edge_feats={ckpt_edge_feats} differs from current={edge_feats}. Recreating model...")
                    edge_feats = ckpt_edge_feats
                    net_model = model_cls(
                        feats=feats,
                        edge_feats=edge_feats,
                        logit_net=not args.no_logit_net,
                        **model_kwargs,
                    ).to(args.device)
                    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr)
                    
                    # Old checkpoint may have different edge_feats
                    print(f"Warning: Checkpoint edge_feats={ckpt_edge_feats} differs from expected=6. Model may not load correctly.")
            
            # Load model state
            load_result = net.load_multihead_state_dict(net_model, state_dict)
            print("Loaded model state")
            if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
                print(
                    "Checkpoint loaded with architecture migration: "
                    f"missing={list(load_result.missing_keys)}, "
                    f"unexpected={list(load_result.unexpected_keys)}"
                )
            if net.repair_dead_lowrank_head_adapter(net_model):
                print(
                    "Reinitialized dead low-rank head adapter from old checkpoint; "
                    "head-specific residuals can now train."
                )
            
            # Load optimizer state
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded optimizer state")
            
            # Resume from next epoch
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"Resuming from epoch {start_epoch}")
            
            # Log previous validation metrics if available
            if "val_cost" in checkpoint:
                print(f"Previous validation cost: {checkpoint['val_cost']:.4f}")
            if "val_gap" in checkpoint:
                print(f"Previous validation gap: {checkpoint['val_gap']:.4f}%")


    # Load validation data
    val_dataset, baseline_values = load_validation_data(args, logger)
    
    if baseline_values is not None:
        logger.info("Using baseline costs from dataset.")

    # Training loop
    global_step = 0
    best_val_cost = float('inf')
    best_model_state = None
    total_train_time = 0.0
    peak_memory_gb = 0.0

    # Pre-training validation
    if val_dataset is not None:
        # Epoch -1: Pure MFACO check (no model)
        # This ensures we see the performance of the underlying ACO without random weights
        logger.info("Running pre-training validation (Pure MFACO)...")
        avg_last, avg_best, avg_gap, val_metrics = validation_fn(
            None, val_dataset, args, baseline_values
        )
        logger.log_epoch_summary(-1, 0.0, avg_best, avg_gap)
        # logger.log_validation(
        #     avg_last, avg_best, avg_gap, -1, val_metrics,
        #     timing=None, step=global_step
        # )

    for epoch in range(start_epoch, args.epochs):
        # Track peak memory before epoch
        epoch_end_memory = None
        best_saved = False
        if _device_uses_cuda(args.device):
            torch.cuda.synchronize()

        # Train one epoch
        (global_step, avg_train, t_neural, t_aco,
         epoch_train_time) = train_epoch_fn(
            net_model, optimizer, global_step, epoch, args
        )
        total_train_time += epoch_train_time

        # Track peak memory after epoch
        epoch_end_memory = _gpu_memory_allocated_gb(args.device)
        if epoch_end_memory is not None:
            torch.cuda.synchronize()
            peak_memory_gb = max(peak_memory_gb, epoch_end_memory)

        # Validate
        if val_dataset is not None:
            avg_last, avg_best, avg_gap, val_metrics = validation_fn(
                net_model, val_dataset, args, baseline_values
            )

            # Track and save best model
            if avg_best < best_val_cost:
                best_val_cost = avg_best
                best_saved = True
                best_model_state = {
                    "model_state_dict": net_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_cost": avg_best,
                    "val_gap": avg_gap,
                    "config": _serializable_args(args),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "peak_memory_gb": peak_memory_gb,
                }
                best_path = save_dir / f"{model_name}_best.pt"
                torch.save(best_model_state, best_path)
                logger.log_model_saved(best_path, epoch, avg_best, avg_gap)

            logger.log_epoch_summary(
                epoch,
                avg_train,
                avg_best,
                avg_gap,
                gpu_memory_gb=epoch_end_memory,
                peak_memory_gb=peak_memory_gb if epoch_end_memory is not None else None,
                best_saved=best_saved,
            )

            # Log validation metrics
            logger.log_validation(
                avg_last, avg_best, avg_gap, epoch, val_metrics,
                timing={"neural_epoch": t_neural, "aco_epoch": t_aco},
                step=global_step
            )

        # Save "last" checkpoint every epoch
        if args.save_dir:
            save_checkpoint(
                net_model, optimizer, epoch, args,
                save_dir / f"{model_name}_last.pt"
            )
            # Save per-epoch checkpoint
            save_checkpoint(
                net_model, optimizer, epoch, args,
                save_dir / f"{model_name}_epoch{epoch}.pt"
            )

    # Save final model
    logger.info(f"Total Train Time: {total_train_time:.2f}s")
    logger.info(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")
    logger.info(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    if not args.no_wandb:
        wandb.log({"time/total_train_time": total_train_time})
        wandb.log({"memory/peak_gb": peak_memory_gb})
        wandb.log({"model/total_params": total_params})
        wandb.log({"model/trainable_params": trainable_params})

    # No separate "final" checkpoint; "last" is updated each epoch.


if __name__ == "__main__":
    main()
