#!/usr/bin/env python3
"""
Shared helpers for the extended NGFACO problems (BPP, MKP, OP).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple

import torch

# Unified modules
import faco
import utils
from net import NetBPP, NetMKP, NetOP, MultiHeadNet


EXTENDED_PROBLEMS = ("bpp", "mkp", "op")
EPS = 1e-10


def build_model_name(args: argparse.Namespace) -> str:
    """Generate a descriptive checkpoint stem for extended-problem runs."""
    name = (
        f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}"
        f"_H{args.H}_miniH{args.mini_H}_rho{args.rho}"
        f"_lr{args.lr}"
    )

    if getattr(args, "train_anneal", False):
        name += f"_anneal_g{args.gamma}_mg{args.min_gamma}"
    if getattr(args, "warmup", 0) > 0:
        name += f"_warmup{args.warmup}"
    if getattr(args, "no_dynamic_feats", False):
        name += "_static"
    if getattr(args, "static_prior", False):
        name += "_static_prior"
    if getattr(args, "multi_head", False):
        nh = getattr(args, "num_heads", 4)
        dt = getattr(args, "head_decoder_type", "deep_lora")
        name += f"_mh{nh}_{dt}"

    return name


def get_model_edge_feats(args: argparse.Namespace) -> int:
    """Return the model edge width for the selected extended training regime.

    Dynamic runs use the TSP/CVRP "compact3" edge-feature set
    ([base, log_tau_rel, is_in_incumbent]); static-prior runs see only the
    constant ``base`` channel.
    """
    return 1 if getattr(args, "static_prior", False) else 3


def use_dynamic_edge_features(args: argparse.Namespace) -> bool:
    """Return whether pheromone-conditioned graph features are enabled."""
    return (not getattr(args, "no_dynamic_feats", False)) and (not getattr(args, "static_prior", False))


def is_maximization_problem(problem_type: str) -> bool:
    """Return whether larger raw values are better for the problem."""
    return problem_type in {"mkp", "op"}


def raw_values_to_objective(values: torch.Tensor, problem_type: str) -> torch.Tensor:
    """Convert raw solver outputs into a higher-is-better objective tensor."""
    if problem_type == "bpp":
        return -values
    if problem_type in {"mkp", "op"}:
        return values
    raise ValueError(f"Unknown problem type: {problem_type}")


def select_best_value(values: torch.Tensor, problem_type: str) -> float:
    """Select the best value using the problem's optimization direction."""
    objective = raw_values_to_objective(values, problem_type)
    return float(objective.max().item())


def compute_relative_improvement(
    candidate: float,
    baseline: float,
    problem_type: str,
    *,
    objective_mode: bool = False,
) -> Optional[float]:
    """Return percent improvement, positive when candidate is better."""
    if abs(baseline) <= EPS:
        return None
    if objective_mode:
        return (candidate - baseline) / abs(baseline) * 100.0
    if is_maximization_problem(problem_type):
        return (candidate - baseline) / abs(baseline) * 100.0
    return (baseline - candidate) / abs(baseline) * 100.0


def prior_output_to_heuristic(
    prior: torch.Tensor,
    base_heuristic: torch.Tensor,
    *,
    logit_net: bool,
) -> torch.Tensor:
    """Convert model output into positive heuristic weights.

    Extended BPP/MKP/OP solvers consume ``aco.heuristic`` as constructive
    weights and their C++ samplers only read the pheromone/heuristic channels
    (they cannot take an additive ``prior`` term during sampling like the TSP/
    CVRP samplers do).  To reproduce the TSP/CVRP sampling distribution

        log_w = alpha*log(tau) + beta*log(eta) + prior

    we fold the neural output into the heuristic as ``eta * exp(prior)`` so that
    ``beta*log(heuristic) = beta*log(eta) + beta*prior``.  The model output is
    therefore a *raw additive logit* on top of the default problem heuristic,
    exactly like ``output_to_sparse_prior`` for TSP/CVRP (an untrained model
    with prior~0 gives multiplier~1, i.e. plain ACO).
    """
    if logit_net:
        # Additive-logit modulation matching TSP/CVRP: heuristic = eta * exp(prior)
        # so that beta*log(heuristic) = beta*log(eta) + beta*prior.  During training
        # the GNN's batch-norm keeps the prior O(1), so the tanh bound is ~linear
        # (identity) there; it only saturates in the eval-at-init corner case where
        # BN running stats are uninitialized.  For |prior| << LOGIT_BOUND this is the
        # plain additive logit (an untrained model gives multiplier ~ 1, i.e. ACO).
        LOGIT_BOUND = 6.0
        bounded_logit = LOGIT_BOUND * torch.tanh(prior / LOGIT_BOUND)
        multiplier = torch.exp(bounded_logit)
    else:
        multiplier = prior.clamp_min(EPS)
    base = base_heuristic.to(device=prior.device, dtype=prior.dtype)
    return torch.where(base > EPS, base * multiplier, multiplier)


def align_edge_attr_width(pyg_data: Any, expected_edge_feats: int) -> Any:
    """Slice or zero-pad edge attributes to match the expected width."""
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


def extract_problem_data(problem_type: str, instance_data: Any) -> Dict[str, torch.Tensor]:
    """Normalize raw/dict instance payloads into a consistent mapping."""
    if isinstance(instance_data, dict):
        if problem_type == "bpp":
            return {"demand": instance_data["demand"]}
        if problem_type == "mkp":
            return {"prize": instance_data["prize"], "weight": instance_data["weight"]}
        if problem_type == "op":
            return {"distances": instance_data["distances"], "prizes": instance_data["prizes"]}
    else:
        if problem_type == "bpp":
            return {"demand": instance_data}
        if problem_type == "mkp":
            prize, weight = instance_data
            return {"prize": prize, "weight": weight}
        if problem_type == "op":
            distances, prizes = instance_data
            return {"distances": distances, "prizes": prizes}
    raise ValueError(f"Unknown problem type: {problem_type}")


def prior_kwargs(problem_type: str, data: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
    """Build reshape kwargs for the selected problem."""
    return {
        "demand": data.get("demand") if problem_type == "bpp" else None,
        "prize": data.get("prize") if problem_type == "mkp" else None,
        "prizes": data.get("prizes") if problem_type == "op" else None,
    }


def reshape_prior_output(
    prior_output: torch.Tensor,
    problem_type: str,
    demand: Optional[torch.Tensor] = None,
    prize: Optional[torch.Tensor] = None,
    prizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reshape a flat model output into the dense prior matrix for the problem."""
    if problem_type == "bpp":
        if demand is None:
            raise ValueError("demand is required for BPP prior reshaping")
        n = len(demand) - 1
    elif problem_type == "mkp":
        if prize is None:
            raise ValueError("prize is required for MKP prior reshaping")
        n = len(prize)
    elif problem_type == "op":
        if prizes is None:
            raise ValueError("prizes is required for OP prior reshaping")
        n = len(prizes)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    return prior_output.view(n + 1, n + 1)


def get_pyg_args(problem_type: str, data: Dict[str, torch.Tensor], device: str) -> Tuple[Any, ...]:
    """Return the positional graph-builder args for the problem."""
    if problem_type == "bpp":
        return (data["demand"], device)
    if problem_type == "mkp":
        return (data["prize"], data["weight"], device)
    if problem_type == "op":
        return (data["distances"], data["prizes"], device)
    raise ValueError(f"Unknown problem type: {problem_type}")


def get_problem_n(problem_type: str, data: Dict[str, Any]) -> int:
    """Extract the problem size n from the data dict (number of non-depot items)."""
    if problem_type == "bpp":
        return len(data["demand"]) - 1
    if problem_type == "mkp":
        return len(data["prize"])
    if problem_type == "op":
        return len(data["prizes"])
    raise ValueError(f"Unknown problem type: {problem_type}")


def reshape_multihead_extended_prior(
    prior_output: torch.Tensor,
    num_heads: int,
    n: int,
) -> torch.Tensor:
    """Reshape multi-head model output (E, H) → (H, n+1, n+1).

    All three extended problems build fully-connected graphs with (n+1)^2 edges,
    so the model produces (E, H) = ((n+1)^2, H) which we reshape to (H, n+1, n+1).
    """
    return prior_output.T.contiguous().view(num_heads, n + 1, n + 1)


def create_model(
    problem_type: str,
    *,
    m: int = 5,
    feats: Optional[int] = None,
    edge_feats: int,
    multi_head: bool = False,
    num_heads: int = 4,
    rank: int = 8,
    lora_alpha: float = 1.0,
    freeze_lora_base: bool = False,
    head_decoder_type: str = "deep_lora",
    head_zdim: int = 128,
    head_adapter_init: str = "random",
    head_adapter_init_std: float = 0.02,
    **kwargs: Any,
) -> Any:
    """Create the neural model for the selected extended problem."""
    if not multi_head:
        single_head_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"grad_checkpointing", "logit_net"}
        }
        if problem_type == "bpp":
            return NetBPP(feats=1 if feats is None else feats, edge_feats=edge_feats, **single_head_kwargs)
        if problem_type == "mkp":
            resolved_feats = (m + 1) if feats is None else feats
            return NetMKP(m=m, feats=resolved_feats, edge_feats=edge_feats, **single_head_kwargs)
        if problem_type == "op":
            return NetOP(feats=2 if feats is None else feats, edge_feats=edge_feats, **single_head_kwargs)
        raise ValueError(f"Unknown problem: {problem_type}")

    # Multi-head path: use MultiHeadNet with problem-specific feature dimensions
    if problem_type == "bpp":
        resolved_feats = 1 if feats is None else feats
    elif problem_type == "mkp":
        resolved_feats = (m + 1) if feats is None else feats
    elif problem_type == "op":
        resolved_feats = 2 if feats is None else feats
    else:
        raise ValueError(f"Unknown problem: {problem_type}")
    return MultiHeadNet(
        problem_type=problem_type,
        m=m,
        feats=resolved_feats,
        edge_feats=edge_feats,
        num_heads=num_heads,
        rank=rank,
        lora_alpha=lora_alpha,
        freeze_lora_base=freeze_lora_base,
        head_decoder_type=head_decoder_type,
        head_zdim=head_zdim,
        head_adapter_init=head_adapter_init,
        head_adapter_init_std=head_adapter_init_std,
        **kwargs,
    )


def setup_aco(
    args: argparse.Namespace,
    instance_data: Any,
    problem_type: str,
    *,
    heuristic: Optional[torch.Tensor] = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Create the ACO solver and matching PyG args for an extended problem instance."""
    data = extract_problem_data(problem_type, instance_data)

    if problem_type == "bpp":
        aco = faco.ACO_BPP(
            demand=data["demand"],
            capacity=args.capacity,
            n_ants=args.n_ants,
            decay=args.rho,
            alpha=args.alpha,
            beta=args.beta,
            elitist=bool(getattr(args, "elitist", False)),
            heuristic=heuristic,
            device=args.device,
        )
    elif problem_type == "mkp":
        aco = faco.ACO_MKP(
            prize=data["prize"],
            weight=data["weight"],
            n_ants=args.n_ants,
            decay=args.rho,
            alpha=args.alpha,
            beta=args.beta,
            elitist=bool(getattr(args, "elitist", False)),
            heuristic=heuristic,
            device=args.device,
        )
    elif problem_type == "op":
        aco = faco.ACO_OP(
            distances=data["distances"],
            prizes=data["prizes"],
            max_len=args.max_len,
            n_ants=args.n_ants,
            decay=args.rho,
            alpha=args.alpha,
            beta=args.beta,
            elitist=bool(getattr(args, "elitist", False)),
            heuristic=heuristic,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return aco, get_pyg_args(problem_type, data, args.device)


def build_pyg_data(
    aco: Any,
    problem_type: str,
    *pyg_args: Any,
    dynamic: bool = True,
):
    """Build PyG data for the given problem from the live ACO state."""
    if problem_type == "bpp":
        demand, device = pyg_args
        return utils.build_pyg_data_bpp(
            demand=demand,
            aco=aco,
            device=device,
            dynamic=dynamic,
        ).to(device)
    if problem_type == "mkp":
        prize, weight, device = pyg_args
        return utils.build_pyg_data_mkp(
            prize=prize,
            weight=weight,
            aco=aco,
            device=device,
            dynamic=dynamic,
        ).to(device)
    if problem_type == "op":
        distances, prizes, device = pyg_args
        return utils.build_pyg_data_op(
            distances=distances,
            prizes=prizes,
            aco=aco,
            device=device,
            dynamic=dynamic,
        ).to(device)
    raise ValueError(f"Unknown problem type: {problem_type}")


def add_extended_problem_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared BPP/MKP/OP problem configuration flags."""
    parser.add_argument("--problem", type=str, choices=list(EXTENDED_PROBLEMS), required=True, help="Problem type")
    parser.add_argument("--n_node", type=int, default=50, help="Problem size")
    parser.add_argument("--m", type=int, default=5, help="Number of constraints for MKP")
    parser.add_argument("--capacity", type=float, default=150.0, help="Bin capacity for BPP")
    parser.add_argument("--max_len", type=float, default=4.0, help="Maximum route length for OP")
    return parser
