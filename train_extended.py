#!/usr/bin/env python3
"""
Training script for Neural-Guided Fast ACO (NGFACO) - Extended for BPP, MKP, OP.

This module implements PPO and REINFORCE training for learning neural priors
that guide ant colony optimization for BPP, MKP, and OP problems.
"""

import time
import torch
import os
import argparse
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional, Dict, List, Any, Tuple, Union
import gc

# Shared helpers
import wandb
import functools

# Extended modules
import net_extended
import faco_extended
import utils_extended

# Specific class imports
from net_extended import NetBPP, NetMKP, NetOP
from faco_extended import MFACO_BPP, MFACO_MKP, MFACO_OP

# Import from utils
from utils import (
    row_softmax, mean_row_kl, rel_l2_drift, top_set, top_turnover,
    top1_flip_rate, safe_corr, top_overlap_frac, row_top1_match_rate, EPS,
    Logger, MetricsCollector, get_logger, init_logger
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_model_name(args: argparse.Namespace) -> str:
    """Generate descriptive model filename based on parameters (matching train.py format)."""
    name = (f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}"
            f"_H{args.H}_miniH{args.mini_H}_rho{args.rho}"
            f"_lr{args.lr}")

    if args.train_anneal:
        name += f"_anneal_g{args.gamma}_mg{args.min_gamma}"
    if args.warmup > 0:
        name += f"_warmup{args.warmup}"

    # Ablation suffixes
    if args.no_dynamic_feats:
        name += "_static"
    if args.static_prior:
        name += "_static_prior"

    return name


EPS = 1e-10


def get_model_edge_feats(args: argparse.Namespace) -> int:
    """Return the model edge width for the selected training regime."""
    return 1 if args.static_prior else 3


def use_dynamic_edge_features(args: argparse.Namespace) -> bool:
    """Return whether pheromone-conditioned graph features are enabled."""
    return (not args.no_dynamic_feats) and (not args.static_prior)


def is_maximization_problem(problem_type: str) -> bool:
    """All extended problems are optimized as maximization objectives."""
    return problem_type in {"bpp", "mkp", "op"}


def raw_values_to_objective(values: torch.Tensor, problem_type: str) -> torch.Tensor:
    """Convert raw solver outputs into a higher-is-better objective tensor."""
    if problem_type == "bpp":
        return -values
    if problem_type in {"mkp", "op"}:
        return values
    raise ValueError(f"Unknown problem type: {problem_type}")


def select_best_objective(values: torch.Tensor, problem_type: str) -> float:
    """Select the best objective value for a batch of raw solver outputs."""
    objective = raw_values_to_objective(values, problem_type)
    return float(objective.max().item())


def compute_relative_improvement(candidate: float, baseline: float) -> Optional[float]:
    """Return percent improvement for maximize-style objectives."""
    if abs(baseline) <= EPS:
        return None
    return (candidate - baseline) / abs(baseline) * 100.0


def align_edge_attr_width(pyg_data: Any, expected_edge_feats: int) -> Any:
    """Slice or zero-pad edge attributes to the expected feature width."""
    current_edge_feats = pyg_data.edge_attr.shape[1]
    if current_edge_feats == expected_edge_feats:
        return pyg_data
    if current_edge_feats > expected_edge_feats:
        pyg_data.edge_attr = pyg_data.edge_attr[:, :expected_edge_feats]
        return pyg_data

    pad = torch.zeros(
        (pyg_data.edge_attr.shape[0], expected_edge_feats - current_edge_feats),
        device=pyg_data.edge_attr.device,
        dtype=pyg_data.edge_attr.dtype,
    )
    pyg_data.edge_attr = torch.cat((pyg_data.edge_attr, pad), dim=1)
    return pyg_data


def reshape_prior_output(
    prior_output: torch.Tensor,
    problem_type: str,
    demand: Optional[torch.Tensor] = None,
    prize: Optional[torch.Tensor] = None,
    prizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reshape a flat model output into the dense prior matrix for the problem."""
    if problem_type == 'bpp':
        if demand is None:
            raise ValueError("demand is required for BPP prior reshaping")
        n = len(demand) - 1
    elif problem_type == 'mkp':
        if prize is None:
            raise ValueError("prize is required for MKP prior reshaping")
        n = len(prize)
    elif problem_type == 'op':
        if prizes is None:
            raise ValueError("prizes is required for OP prior reshaping")
        n = len(prizes)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    return prior_output.view(n + 1, n + 1)


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
        magic_mask = (1 << torch.arange(k, device=device))
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
        instance_data: Problem instance data (can be dict or raw data)
        problem_type: 'bpp', 'mkp', or 'op'

    Returns:
        Tuple of (aco_solver, pyg_args)
    """
    # Extract raw data if instance_data is a dictionary
    if isinstance(instance_data, dict):
        if problem_type == 'bpp':
            demand = instance_data['demand']
        elif problem_type == 'mkp':
            prize = instance_data['prize']
            weight = instance_data['weight']
        elif problem_type == 'op':
            distances = instance_data['distances']
            prizes = instance_data['prizes']
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    else:
        # Use raw data directly
        if problem_type == 'bpp':
            demand = instance_data
        elif problem_type == 'mkp':
            prize, weight = instance_data
        elif problem_type == 'op':
            distances, prizes = instance_data
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    if problem_type == 'bpp':
        kwargs = {
            'demand': demand,
            'capacity': args.capacity,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,  # Use False like DeepACO
            'device': args.device,
        }
        pyg_args = (demand, args.device)
        aco = faco_extended.MFACO_BPP(**kwargs)

    elif problem_type == 'mkp':
        kwargs = {
            'prize': prize,
            'weight': weight,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'device': args.device,
        }
        pyg_args = (prize, weight, args.device)
        aco = faco_extended.MFACO_MKP(**kwargs)

    elif problem_type == 'op':
        kwargs = {
            'distances': distances,
            'prizes': prizes,
            'max_len': args.max_len,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'device': args.device,
        }
        pyg_args = (distances, prizes, args.device)
        aco = faco_extended.MFACO_OP(**kwargs)

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return aco, pyg_args


def get_heuristic_tensor(
    aco: Any,
    problem_type: str,
    device: str
) -> torch.Tensor:
    """Get heuristic tensor from ACO solver (unified API)."""
    return aco.heuristic


def build_pyg_data(
    aco: Any,
    *args,
    dynamic: bool = True
):
    """Build PyG data for the given problem."""
    problem_type = args[0]  # First arg is problem type

    if problem_type == 'bpp':
        demand = args[1]
        device = args[2]
        pyg_data = utils_extended.build_pyg_data_bpp(
            demand, aco, device, dynamic=dynamic
        )
        # Ensure all tensors are on the correct device
        pyg_data = pyg_data.to(device)
        return pyg_data

    elif problem_type == 'mkp':
        prize, weight = args[1], args[2]
        device = args[3]
        pyg_data = utils_extended.build_pyg_data_mkp(
            prize, weight, aco, device, dynamic=dynamic
        )
        # Ensure all tensors are on the correct device
        pyg_data = pyg_data.to(device)
        return pyg_data

    elif problem_type == 'op':
        distances, prizes = args[1], args[2]
        device = args[3]
        pyg_data = utils_extended.build_pyg_data_op(
            distances, prizes, aco, device, dynamic=dynamic
        )
        # Ensure all tensors are on the correct device
        pyg_data = pyg_data.to(device)
        return pyg_data

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_instance_reinforce(
    model: Any,
    optimizer: torch.optim.Optimizer,
    instance_data: Any,
    args: argparse.Namespace
) -> Tuple[float, float, Dict[str, float]]:
    """
    Train on a single instance using REINFORCE algorithm with dynamic ACO.

    Args:
        model: Neural network model
        optimizer: Optimizer
        instance_data: Problem instance data
        args: Training arguments

    Returns:
        Tuple of (avg_cost, best_cost, metrics_dict)
    """
    model.train()

    # Extract raw data
    if isinstance(instance_data, dict):
        if args.problem == 'bpp':
            demand = instance_data['demand']
        elif args.problem == 'mkp':
            prize = instance_data['prize']
            weight = instance_data['weight']
        elif args.problem == 'op':
            distances = instance_data['distances']
            prizes = instance_data['prizes']
        else:
            raise ValueError(f"Unknown problem type: {args.problem}")
    else:
        # Use raw data directly
        if args.problem == 'bpp':
            demand = instance_data
        elif args.problem == 'mkp':
            prize, weight = instance_data
        elif args.problem == 'op':
            distances, prizes = instance_data
        else:
            raise ValueError(f"Unknown problem type: {args.problem}")

    # Create ACO solver without heuristic initially
    if args.problem == 'bpp':
        kwargs = {
            'demand': demand,
            'capacity': args.capacity,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,
            'heuristic': None,
            'device': args.device,
        }
        aco = faco_extended.MFACO_BPP(**kwargs)
    elif args.problem == 'mkp':
        kwargs = {
            'prize': prize,
            'weight': weight,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,
            'heuristic': None,
            'device': args.device,
        }
        aco = faco_extended.MFACO_MKP(**kwargs)
    elif args.problem == 'op':
        kwargs = {
            'distances': distances,
            'prizes': prizes,
            'max_len': args.max_len,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,
            'heuristic': None,
            'device': args.device,
        }
        aco = faco_extended.MFACO_OP(**kwargs)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    expected_edge_feats = get_model_edge_feats(args)
    dynamic_graph = use_dynamic_edge_features(args)
    if args.problem == 'bpp':
        pyg_args = (demand, args.device)
    elif args.problem == 'mkp':
        pyg_args = (prize, weight, args.device)
    elif args.problem == 'op':
        pyg_args = (distances, prizes, args.device)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    static_pyg_data = None
    if args.static_prior:
        static_pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=False)
        static_pyg_data = align_edge_attr_width(static_pyg_data, expected_edge_feats)

    best_seen = float("-inf")
    avg_cost_last = None

    metrics = MetricsCollector()

    # Timing accumulators
    t_neural_total = 0.0
    t_aco_total = 0.0

    # Dynamic ACO: regenerate neural prior each outer iteration
    outer_pbar = tqdm(range(args.H), desc="Outer", leave=False)
    for outer in outer_pbar:
        t0 = time.time()
        if args.static_prior:
            pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=False)
            pyg_data = align_edge_attr_width(pyg_data, expected_edge_feats)
            prior_output = model(pyg_data)
            prior = reshape_prior_output(
                prior_output,
                args.problem,
                demand=demand if args.problem == 'bpp' else None,
                prize=prize if args.problem == 'mkp' else None,
                prizes=prizes if args.problem == 'op' else None,
            )
        else:
            pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=dynamic_graph)
            pyg_data = align_edge_attr_width(pyg_data, expected_edge_feats)
            prior_output = model(pyg_data)
            prior = reshape_prior_output(
                prior_output,
                args.problem,
                demand=demand if args.problem == 'bpp' else None,
                prize=prize if args.problem == 'mkp' else None,
                prizes=prizes if args.problem == 'op' else None,
            )
        t_neural_total += time.time() - t0

        # Storage for gradients
        all_logp_sums = []
        all_objectives = []

        for inner in range(args.mini_H):
            t_aco_start_inner = time.time()
            
            # Annealing
            current_prior = prior
            if args.train_anneal:
                factor = compute_annealing_factor(inner, args.mini_H, args.gamma, args.min_gamma)
                current_prior = prior * factor

            # Update ACO solver's heuristic with new neural prior
            aco.heuristic = current_prior.to(device=args.device, dtype=torch.float32)
            aco._sync_cpp_inputs()

            # Sample solutions
            res = aco.sample(require_prob=True, prior=None, parallel_traced=True)
            costs, paths, logps, _ = res

            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            objective_t = raw_values_to_objective(costs_t, args.problem)

            best_idx = int(objective_t.argmax().item())
            best_cost_obj = float(objective_t[best_idx].item())
            best_seen = max(best_seen, best_cost_obj)

            # Update pheromone with the sampled batch
            # The C++ solver handles elitist vs non-elitist based on args.elitist
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
                    logp_sum = logp_steps_new.sum(dim=0) if logp_steps_new.size(0) > 0 else torch.zeros((paths.size(1),), device=args.device)
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

            # Update progress bar
            outer_pbar.set_postfix({
                "obj": f"{best_seen:.4f}",
                "avg": f"{avg_cost_last:.4f}",
                "loss": f"{loss.item():.4f}"
            })

    # Finalize timing metrics
    out_metrics = metrics.get_all_means()
    out_metrics["time_neural"] = t_neural_total
    out_metrics["time_aco"] = t_aco_total

    return avg_cost_last, best_seen, out_metrics


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(args: argparse.Namespace):
    """Main training loop."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model_name = build_model_name(args)

    # Initialize wandb
    if args.wandb_project and not args.no_wandb:
        run_id = wandb.util.generate_id()
        run_name = args.run_name if args.run_name else model_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=run_id,
            group=args.wandb_group,
            config=vars(args),
            mode="online" if not args.dry_run else "disabled"
        )

    # Create model
    edge_feats = get_model_edge_feats(args)
    if args.problem == 'bpp':
        model = NetBPP(feats=1, edge_feats=edge_feats)
    elif args.problem == 'mkp':
        model = NetMKP(m=args.m, feats=args.m + 1, edge_feats=edge_feats)
    elif args.problem == 'op':
        model = NetOP(feats=2, edge_feats=edge_feats)
    else:
        raise ValueError(f"Unknown problem: {args.problem}")

    model = model.to(args.device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # Use AdamW like DeepACO

    # Training loop
    all_train_costs = []
    all_val_costs = []

    # Generate fixed validation instances for consistent evaluation
    val_instances = []
    torch.manual_seed(args.seed + 1000)  # Different seed for validation
    np.random.seed(args.seed + 1000)
    for _ in range(args.val_size):
        val_instances.append(utils_extended.get_problem_data(
            args.problem, args.n_node, args.device, args.k_sparse
        ))
    # Reset seeds for training
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training
        train_costs = []
        for step in range(args.steps_per_epoch):
            # Generate instance
            instance_data = utils_extended.get_problem_data(
                args.problem, args.n_node, args.device, args.k_sparse
            )

            # Train on instance
            avg_cost, best_cost, metrics = train_instance_reinforce(
                model, optimizer, instance_data, args
            )

            train_costs.append(avg_cost)
            all_train_costs.append(avg_cost)

            # Log metrics
            if args.wandb_project and not args.no_wandb:
                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "train_cost": avg_cost,
                    "best_cost": best_cost,
                    **metrics
                })

        # Validation - compare with and without neural guidance
        val_with_prior = []
        val_without_prior = []
        for instance_data in val_instances:

            # With neural guidance (dynamic model)
            # Extract raw data
            if isinstance(instance_data, dict):
                if args.problem == 'bpp':
                    demand = instance_data['demand']
                elif args.problem == 'mkp':
                    prize = instance_data['prize']
                    weight = instance_data['weight']
                elif args.problem == 'op':
                    distances = instance_data['distances']
                    prizes = instance_data['prizes']
                else:
                    raise ValueError(f"Unknown problem type: {args.problem}")
            else:
                # Use raw data directly
                if args.problem == 'bpp':
                    demand = instance_data
                elif args.problem == 'mkp':
                    prize, weight = instance_data
                elif args.problem == 'op':
                    distances, prizes = instance_data
                else:
                    raise ValueError(f"Unknown problem type: {args.problem}")

            # Create ACO solver without heuristic initially
            if args.problem == 'bpp':
                kwargs = {
                    'demand': demand,
                    'capacity': args.capacity,
                    'n_ants': args.n_ants,
                    'decay': args.rho,
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'elitist': False,
                    'heuristic': None,
                    'device': args.device,
                }
                aco = faco_extended.MFACO_BPP(**kwargs)
            elif args.problem == 'mkp':
                kwargs = {
                    'prize': prize,
                    'weight': weight,
                    'n_ants': args.n_ants,
                    'decay': args.rho,
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'elitist': False,
                    'heuristic': None,
                    'device': args.device,
                }
                aco = faco_extended.MFACO_MKP(**kwargs)
            elif args.problem == 'op':
                kwargs = {
                    'distances': distances,
                    'prizes': prizes,
                    'max_len': args.max_len,
                    'n_ants': args.n_ants,
                    'decay': args.rho,
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'elitist': False,
                    'heuristic': None,
                    'device': args.device,
                }
                aco = faco_extended.MFACO_OP(**kwargs)
            else:
                raise ValueError(f"Unknown problem type: {args.problem}")

            dynamic_graph = use_dynamic_edge_features(args)
            expected_edge_feats = get_model_edge_feats(args)
            if args.problem == 'bpp':
                pyg_args = (demand, args.device)
            elif args.problem == 'mkp':
                pyg_args = (prize, weight, args.device)
            elif args.problem == 'op':
                pyg_args = (distances, prizes, args.device)
            else:
                raise ValueError(f"Unknown problem type: {args.problem}")

            model.eval()
            with torch.no_grad():
                if args.static_prior:
                    pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=False)
                    pyg_data = align_edge_attr_width(pyg_data, expected_edge_feats)
                    prior_output = model(pyg_data)
                    prior = reshape_prior_output(
                        prior_output,
                        args.problem,
                        demand=demand if args.problem == 'bpp' else None,
                        prize=prize if args.problem == 'mkp' else None,
                        prizes=prizes if args.problem == 'op' else None,
                    )
                    aco.heuristic = prior.to(device=args.device, dtype=torch.float32)
                    aco._sync_cpp_inputs()

                # Run H outer iterations
                for outer in range(args.H):
                    if not args.static_prior:
                        pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=dynamic_graph)
                        pyg_data = align_edge_attr_width(pyg_data, expected_edge_feats)
                        prior_output = model(pyg_data)
                        prior = reshape_prior_output(
                            prior_output,
                            args.problem,
                            demand=demand if args.problem == 'bpp' else None,
                            prize=prize if args.problem == 'mkp' else None,
                            prizes=prizes if args.problem == 'op' else None,
                        )

                    for inner in range(args.mini_H):
                        current_prior = prior
                        if args.train_anneal:
                            factor = compute_annealing_factor(inner, args.mini_H, args.gamma, args.min_gamma)
                            current_prior = prior * factor
                        
                        aco.heuristic = current_prior.to(device=args.device, dtype=torch.float32)
                        aco._sync_cpp_inputs()

                        # Sample with updated heuristic
                        costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)

                        # Update pheromone with the full batch
                        with torch.no_grad():
                            aco.update_pheromone(paths, costs)

                # Get final sample objective
                costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
                objective_t = raw_values_to_objective(costs_t, args.problem)
                best_cost = float(objective_t.max().item())
                val_with_prior.append(best_cost)

            # Without neural guidance (baseline ACO)
            aco_baseline, _ = setup_aco(args, instance_data, args.problem)
            costs_baseline, _, _, _ = aco_baseline.sample(require_prob=False, prior=None, parallel_traced=True)
            costs_baseline_t = torch.as_tensor(costs_baseline, device=args.device, dtype=torch.float32)
            objective_baseline_t = raw_values_to_objective(costs_baseline_t, args.problem)
            best_cost_baseline = float(objective_baseline_t.max().item())
            val_without_prior.append(best_cost_baseline)

            model.train()

        avg_val_with_prior = np.mean(val_with_prior)
        avg_val_without_prior = np.mean(val_without_prior)

        improvement = compute_relative_improvement(avg_val_with_prior, avg_val_without_prior)
        if improvement is None:
            improvement = 0.0

        all_val_costs.append(avg_val_with_prior)

        print(f"  Train Cost: {np.mean(train_costs):.4f}")
        print(f"  Val Cost (with prior): {avg_val_with_prior:.4f} ± {np.std(val_with_prior):.4f}")
        print(f"  Val Cost (without prior): {avg_val_without_prior:.4f} ± {np.std(val_without_prior):.4f}")
        print(f"  Improvement: {improvement:.2f}%")

        # Log validation metrics
        if args.wandb_project and not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_cost": np.mean(train_costs),
                "val_cost_with_prior": avg_val_with_prior,
                "val_cost_with_prior_std": np.std(val_with_prior),
                "val_cost_with_prior_min": np.min(val_with_prior),
                "val_cost_with_prior_max": np.max(val_with_prior),
                "val_cost_without_prior": avg_val_without_prior,
                "val_cost_without_prior_std": np.std(val_without_prior),
                "val_cost_without_prior_min": np.min(val_without_prior),
                "val_cost_without_prior_max": np.max(val_without_prior),
                "improvement": improvement,
            })

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.save_dir) / f"{model_name}_epoch{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_cost': np.mean(train_costs),
                'val_cost_with_prior': avg_val_with_prior,
                'val_cost_without_prior': avg_val_without_prior,
                'improvement': improvement,
                'config': vars(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = Path(args.save_dir) / f"{model_name}_best.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
    }, final_path)
    print(f"\nSaved final model: {final_path}")

    if args.wandb_project and not args.no_wandb:
        wandb.finish()


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def build_model_name(args: argparse.Namespace) -> str:
    """Generate descriptive model filename based on parameters (matching train.py format)."""
    name = (f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}"
            f"_H{args.H}_miniH{args.mini_H}_rho{args.rho}"
            f"_lr{args.lr}")

    if args.train_anneal:
        name += f"_anneal_g{args.gamma}_mg{args.min_gamma}"
    if args.warmup > 0:
        name += f"_warmup{args.warmup}"

    # Ablation suffixes
    if args.no_dynamic_feats:
        name += "_static"
    if args.static_prior:
        name += "_static_prior"

    return name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NGFACO for BPP, MKP, OP")

    # Problem arguments
    parser.add_argument("--problem", type=str, choices=["bpp", "mkp", "op"], required=True,
                        help="Problem type")
    parser.add_argument("--n_node", type=int, default=50, help="Problem size")
    parser.add_argument("--m", type=int, default=5, help="Number of constraints for MKP")
    parser.add_argument("--capacity", type=float, default=150.0, help="Bin capacity for BPP")
    parser.add_argument("--max_len", type=float, default=4.0, help="Maximum route length for OP")

    # Training arguments (matching DeepACO parameters)
    parser.add_argument("--H", type=int, default=5, help="Number of outer iterations (replaces T)")
    parser.add_argument("--mini_H", type=int, default=5, help="Number of inner iterations")
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants")
    parser.add_argument("--k_sparse", type=int, default=32, help="K-NN size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=64, help="Steps per epoch")
    parser.add_argument("--val_size", type=int, default=16, help="Validation size")  # Use 100 like DeepACO
    parser.add_argument("--aco_iters_per_sample", type=int, default=20, help="ACO iterations per sample")

    # Optimization arguments
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")  # Use 3e-4 like DeepACO
    parser.add_argument("--rho", type=float, default=0.9, help="Pheromone retention (0.9 = 90%% kept, 10%% decayed)")  # Use 0.9 like DeepACO
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Heuristic weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Prior scaling")
    parser.add_argument("--min_gamma", type=float, default=0.0, help="Minimum prior scaling")

    # Training control
    parser.add_argument("--warmup", type=int, default=0, help="Warmup steps")
    parser.add_argument("--train_anneal", action="store_true", help="Use annealing")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="Disable dynamic features")
    parser.add_argument("--smallvram", action="store_true", help="Use small VRAM mode")
    parser.add_argument("--static_prior", action="store_true",
                        help="Train a DeepACO-style static-prior baseline (1 edge feature, no dynamic prior)")

    # Wandb settings and output
    parser.add_argument("--wandb_project", type=str, default="ngfaco_extended", help="WandB project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints_extended", help="Save directory")
    parser.add_argument("--save_interval", type=int, default=1, help="Save interval")
    parser.add_argument("--dry_run", action="store_true", help="Dry run")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
