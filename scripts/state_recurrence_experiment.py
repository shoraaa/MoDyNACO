#!/usr/bin/env python3
"""Collect DyNACO search states and test whether instance domains are separable.

The experiment is designed for the manuscript claim "states recur across
instances."  It compares search states from size-matched domains by default:
uniform-1k and clustered-16 with 1,000 TSP nodes or 1,000 CVRP customers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import faco  # noqa: E402
import net  # noqa: E402
import utils  # noqa: E402


DOMAIN_SPECS = {
    "uniform-1k": {"kind": "uniform", "n": 1000, "clusters": None},
    "clustered-16": {"kind": "clustered", "n": 16000, "clusters": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw or learned DyNACO state representations, train a domain "
            "classifier, and visualize whether search states overlap."
        )
    )
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Domain labels to compare. Built-ins: uniform-1k clustered-16")
    parser.add_argument("--domain-config", type=Path, default=None,
                        help="YAML/JSON file with domain_specs and optional compare/domain_order list")
    parser.add_argument("--domain-spec", action="append", default=[],
                        help='Add/override one domain spec as label={"kind":"uniform","n":1000}. Repeatable.')
    parser.add_argument("--domain-specs", type=str, default=None,
                        help='JSON object mapping labels to specs, or @path/to/specs.json.')
    parser.add_argument("--instances-per-domain", type=int, default=10)
    parser.add_argument("--H", type=int, default=10,
                        help="Outer prior-refresh steps. Defaults to --steps for backward compatibility.")
    parser.add_argument("--mini_H", type=int, default=100,
                        help="Inner ACO samples/pheromone updates per outer step")
    parser.add_argument("--steps", type=int, default=12,
                        help="Backward-compatible alias for --H when --H is not set")
    parser.add_argument("--n-node", type=int, default=None,
                        help="Optional override for every domain spec's node/customer count")
    parser.add_argument("--clusters", type=int, default=1,
                        help="Cluster count for clustered domains")
    parser.add_argument("--capacity-override", type=float, default=None,
                        help="Raw CVRP capacity before demand normalization")
    parser.add_argument("--k-sparse", type=int, default=32)
    parser.add_argument("--backup-list-size", type=int, default=32)
    parser.add_argument("--n-ants", type=int, default=32)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-new-edges", type=int, default=12)
    parser.add_argument("--no-local-search", action="store_true")
    parser.add_argument("--extend-ls", action="store_true")
    parser.add_argument("--smooth-mmas", action="store_true")
    parser.add_argument("--disable-heuristic", action="store_true")
    parser.add_argument("--normalized-heuristic", action="store_true")
    parser.add_argument("--edge-feature-set", choices=["full", "compact3"], default="compact3")
    parser.add_argument("--ablation-pheromone-features", action="store_true")
    parser.add_argument("--ablation-incumbent-features", action="store_true")
    parser.add_argument("--representation", choices=["raw", "embedding"], default="raw",
                        help="raw uses PyG edge_attr; embedding uses model.emb_net for rollout priors")
    parser.add_argument("--state-surface", choices=["dynamic", "full"], default="dynamic",
                        help="dynamic classifies only pheromone/incumbent channels; full classifies the requested representation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Required for --representation embedding")
    parser.add_argument("--state-level", choices=["behavior", "graph", "edge", "trajectory"], default="graph",
                        help="behavior uses fixed-bin dynamic summaries; graph pools rows; edge samples edge rows; trajectory summarises per-instance search dynamics across all H steps")
    parser.add_argument("--edge-samples-per-state", type=int, default=256)
    parser.add_argument("--classifier", choices=["linear", "mlp"], default="linear",
                        help="linear = logistic regression; mlp = 2-hidden-layer ReLU network")
    parser.add_argument("--classifier-hidden", type=int, default=64,
                        help="Hidden width for --classifier mlp")
    parser.add_argument("--classifier-epochs", type=int, default=300)
    parser.add_argument("--classifier-lr", type=float, default=0.05)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--permutation-trials", type=int, default=0,
                        help="Run N permutation-test trials to estimate p-value")
    parser.add_argument("--controls", action="store_true",
                        help="Run positive control (early-vs-late) and geometry baseline alongside the main domain classifier")
    parser.add_argument("--behavior-classification", action="store_true",
                        help="With a multi-head checkpoint, classify which head produced each guidance field")
    parser.add_argument("--no-anneal", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=Path("results/state_recurrence"))
    args = parser.parse_args()
    if args.H is None:
        args.H = int(args.steps)
    args.resolved_domain_specs = load_domain_specs(args)
    if args.domains is None:
        args.domains = getattr(args, "domain_order_from_config", None) or ["uniform-1k", "clustered-16"]
    if args.state_level == "trajectory" and args.state_surface != "dynamic":
        parser.error("--state-level trajectory requires --state-surface dynamic")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_structured_file(path: Path) -> Any:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _load_structured_payload(value: str) -> Any:
    if value.startswith("@"):
        return _load_structured_file(Path(value[1:]))
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return yaml.safe_load(value)


def _domain_specs_from_config_payload(payload: Any) -> Tuple[Dict[str, Dict[str, Any]], Optional[List[str]]]:
    if not isinstance(payload, dict):
        raise ValueError("domain config must decode to a mapping")

    order = None
    if "compare" in payload:
        order = [str(x) for x in payload["compare"]]
    elif "domain_order" in payload:
        order = [str(x) for x in payload["domain_order"]]

    if "domain_specs" in payload:
        specs_payload = payload["domain_specs"]
    elif isinstance(payload.get("domains"), dict):
        specs_payload = payload["domains"]
    else:
        reserved = {"compare", "domain_order", "domains"}
        specs_payload = {k: v for k, v in payload.items() if k not in reserved}

    if not isinstance(specs_payload, dict):
        raise ValueError("domain specs must be a mapping")
    specs: Dict[str, Dict[str, Any]] = {}
    for label, spec in specs_payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Domain spec for {label!r} must be a mapping")
        specs[str(label)] = dict(spec)
    return specs, order


def load_domain_specs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    specs = {label: dict(spec) for label, spec in DOMAIN_SPECS.items()}
    if args.domain_config is not None:
        config_specs, order = _domain_specs_from_config_payload(_load_structured_file(args.domain_config))
        specs.update(config_specs)
        if order is not None and args.domains is None:
            args.domain_order_from_config = order
    if args.domain_specs:
        payload = _load_structured_payload(args.domain_specs)
        config_specs, order = _domain_specs_from_config_payload(payload)
        specs.update(config_specs)
        if order is not None and args.domains is None:
            args.domain_order_from_config = order
    for item in args.domain_spec:
        if "=" not in item:
            raise ValueError("--domain-spec must have form label={...json-or-yaml...}")
        label, payload = item.split("=", 1)
        if not label:
            raise ValueError("--domain-spec label cannot be empty")
        spec = _load_structured_payload(payload)
        if not isinstance(spec, dict):
            raise ValueError(f"Domain spec for {label!r} must be a mapping")
        specs[label] = dict(spec)
    return specs


def domain_config(label: str, args: argparse.Namespace) -> Dict[str, Any]:
    if label in args.resolved_domain_specs:
        spec = dict(args.resolved_domain_specs[label])
    elif label.startswith("uniform"):
        spec = {"kind": "uniform", "n": 1000, "clusters": None}
    elif label.startswith("clustered"):
        spec = {"kind": "clustered", "n": 1000, "clusters": args.clusters}
    else:
        raise ValueError(
            f"Unknown domain {label!r}; use a built-in, prefix with uniform/clustered, "
            "or pass --domain-spec/--domain-specs"
        )
    if args.n_node is not None:
        spec["n"] = int(args.n_node)
    elif "n" not in spec:
        spec["n"] = 1000
    if spec["kind"] == "clustered" and spec.get("clusters") is None:
        spec["clusters"] = int(args.clusters)
    if spec["kind"] not in {"uniform", "clustered"}:
        raise ValueError(f"Unsupported domain kind for {label!r}: {spec['kind']!r}")
    return spec


def gen_uniform_tsp(n: int, device: str) -> torch.Tensor:
    return torch.rand((n, 2), device=device, dtype=torch.float32)


def gen_clustered_tsp(n: int, n_clusters: int, device: str) -> torch.Tensor:
    centers = torch.rand((n_clusters, 2), device=device, dtype=torch.float32)
    assignments = torch.arange(n, device=device) % n_clusters
    assignments = assignments[torch.randperm(n, device=device)]
    scale = 0.035 / max(1.0, math.sqrt(float(n_clusters) / 16.0))
    coords = centers[assignments] + torch.randn((n, 2), device=device, dtype=torch.float32) * scale
    return coords.clamp_(0.0, 1.0)


def default_cvrp_capacity(n_customers: int) -> float:
    if n_customers >= 50000:
        return 2000.0
    if n_customers >= 10000:
        return 1000.0
    if n_customers >= 5000:
        return 500.0
    if n_customers >= 1000:
        return 250.0
    return 50.0


def gen_cvrp_from_customer_coords(
    customer_coords: torch.Tensor,
    device: str,
    capacity_override: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    n_customers = int(customer_coords.size(0))
    capacity = float(capacity_override) if capacity_override is not None else default_cvrp_capacity(n_customers)
    depot = torch.rand((1, 2), device=device, dtype=torch.float32)
    raw_demands = torch.randint(1, 10, size=(n_customers,), device=device).to(torch.float32)
    demand = torch.cat((torch.zeros((1,), device=device), raw_demands / capacity))
    return torch.cat((depot, customer_coords), dim=0), demand, 1.0


def gen_instance(label: str, args: argparse.Namespace) -> Any:
    spec = domain_config(label, args)
    if spec["kind"] == "uniform":
        coords = gen_uniform_tsp(spec["n"], args.device)
    elif spec["kind"] == "clustered":
        coords = gen_clustered_tsp(spec["n"], spec["clusters"], args.device)
    else:
        raise AssertionError(spec)

    if args.problem == "tsp":
        return coords
    if args.problem == "cvrp":
        capacity_override = spec.get("capacity_override", args.capacity_override)
        return gen_cvrp_from_customer_coords(coords, args.device, capacity_override)
    raise AssertionError(spec)


def build_aco(instance: Any, args: argparse.Namespace, seed: int) -> Any:
    if args.problem == "tsp":
        n = int(instance.size(0))
        common = dict(
            n_ants=args.n_ants,
            coords=instance,
            cand_list_size=min(args.k_sparse, n - 2),
            backup_list_size=min(args.backup_list_size, n - 2),
            min_new_edges=args.min_new_edges,
            decay=args.rho,
            alpha=args.alpha,
            use_local_search=not args.no_local_search,
            extend_ls=args.extend_ls,
            smooth_mmas=args.smooth_mmas,
            enable_torch_sync=True,
            device=args.device,
            disable_heuristic=args.disable_heuristic,
            normalized_heuristic=args.normalized_heuristic,
            fixed_steps=0,
            ls_scope="localized",
            ls_budget="truncated",
            ls_max_opt=0,
            euc_2d_cost=False,
        )
        aco = faco.MFACO_TSP(**common)
    elif args.problem == "cvrp":
        coords, demand, capacity = instance
        n = int(coords.size(0)) - 1
        common = dict(
            coords=coords,
            demand=demand,
            capacity=float(capacity),
            n_ants=args.n_ants,
            cand_list_size=min(args.k_sparse, n - 1),
            backup_list_size=max(min(args.backup_list_size, n - 1), min(64, n - 1)),
            min_new_edges=args.min_new_edges,
            decay=args.rho,
            p_best=0.05,
            use_local_search=not args.no_local_search,
            extend_ls=args.extend_ls,
            smooth_mmas=args.smooth_mmas,
            enable_torch_sync=True,
            device=args.device,
            disable_heuristic=args.disable_heuristic,
            normalized_heuristic=args.normalized_heuristic,
            fixed_steps=0,
            ls_scope="localized",
            ls_budget="truncated",
            ls_max_opt=0,
            euc_2d_cost=False,
        )
        aco = faco.MFACO_CVRP(**common)
    else:
        raise ValueError(f"Unsupported problem: {args.problem}")
    aco.seed_rng(seed)
    return aco


def infer_model_shape(state_dict: Dict[str, torch.Tensor], fallback_edge_feats: int) -> Tuple[int, int]:
    feats = int(state_dict.get("emb_net.v_lin0.weight", torch.empty(0, 2)).shape[1])
    edge_feats = int(state_dict.get("emb_net.e_lin0.weight", torch.empty(0, fallback_edge_feats)).shape[1])
    return feats, edge_feats


def load_embedding_model(args: argparse.Namespace, edge_feats: int) -> torch.nn.Module:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --representation embedding")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    feats, ckpt_edge_feats = infer_model_shape(state_dict, edge_feats)
    if ckpt_edge_feats != edge_feats:
        raise ValueError(
            f"Checkpoint expects {ckpt_edge_feats} edge features, but --edge-feature-set "
            f"produces {edge_feats}. Adjust --edge-feature-set or use a matching checkpoint."
        )

    is_multi = bool(config.get("multi_head", False) or config.get("num_heads", 1) > 1)
    is_multi = is_multi or any("head_codes" in k or ".lora_" in k or "alloc_net." in k for k in state_dict)
    model_cls = net.MultiHeadNet if is_multi else net.Net
    kwargs: Dict[str, Any] = {"feats": feats, "edge_feats": ckpt_edge_feats, "logit_net": True}
    if is_multi:
        kwargs.update({
            "num_heads": int(config.get("num_heads", 4)),
            "head_decoder_type": config.get("head_decoder_type", "lora"),
            "rank": int(config.get("rank", 8)),
            "head_zdim": int(config.get("head_zdim", 16)),
            "alloc_mode": config.get("alloc_mode", "mlp"),
        })
    model = model_cls(**kwargs).to(args.device)
    net.load_multihead_state_dict(model, state_dict)
    model.eval()
    return model


def edge_feature_count(args: argparse.Namespace) -> int:
    if args.edge_feature_set == "compact3":
        return 3
    count = 1
    if not args.ablation_pheromone_features:
        count += 2
    if not args.ablation_incumbent_features:
        count += 3
    return count


def build_state_tensor(aco: Any, instance: Any, args: argparse.Namespace) -> Any:
    kwargs = dict(
        ablation_pheromone=args.ablation_pheromone_features,
        ablation_incumbent=args.ablation_incumbent_features,
        edge_feature_set=args.edge_feature_set,
        dynamic=True,
    )
    if args.problem == "tsp":
        return utils.build_pyg_data_tsp(aco, instance, args.device, **kwargs)
    if args.problem == "cvrp":
        coords, demand, _ = instance
        return utils.build_pyg_data_cvrp(aco, coords, demand, args.device, **kwargs)
    raise ValueError(f"Unsupported problem: {args.problem}")


def instance_node_count(instance: Any, problem: str) -> int:
    if problem == "tsp":
        return int(instance.size(0))
    coords, _, _ = instance
    return int(coords.size(0))


def sample_and_update(aco: Any, args: argparse.Namespace, prior: Optional[torch.Tensor], prior_scale: float = 1.0) -> None:
    if prior is not None and prior.dim() == 3 and hasattr(aco, "sample_mixed_priors"):
        prior_np = prior.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        n_heads = int(prior.shape[0])
        head_counts = [args.n_ants // n_heads] * n_heads
        for i in range(args.n_ants % n_heads):
            head_counts[i] += 1
        sample_out = aco.sample_mixed_priors(
            prior_np,
            require_prob=False,
            parallel_traced=True,
            head_counts=head_counts,
            prior_scale=prior_scale,
        )
        costs, solution_rows = sample_out[0], sample_out[1]
    else:
        prior_arg = None
        if prior is not None:
            scaled = prior * prior_scale if prior_scale != 1.0 else prior
            prior_arg = scaled.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if args.problem == "cvrp":
            costs, solution_rows, *_ = aco.sample(
                require_prob=False,
                prior=prior_arg,
                return_decoded=False,
                parallel_traced=True,
            )
        else:
            costs, solution_rows, *_ = aco.sample(
                require_prob=False,
                prior=prior_arg,
                parallel_traced=True,
            )
    costs_np = np.asarray(costs)
    best_idx = int(np.argmin(costs_np))
    aco.update_pheromone(solution_rows[best_idx], float(costs_np[best_idx]))


def dynamic_edge_rows(edge_attr: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Return dynamic edge-state channels, excluding static geometry/demand cues."""
    if args.edge_feature_set == "compact3":
        # compact3 = dist_norm, log_tau_rel, is_in_incumbent
        return edge_attr[:, 1:]

    # full = dist_norm, optional pheromone channels, optional incumbent channels
    cols: List[int] = []
    cursor = 1
    if not args.ablation_pheromone_features:
        cols.extend([cursor, cursor + 1])
        cursor += 2
    if not args.ablation_incumbent_features:
        cols.extend([cursor, cursor + 1, cursor + 2])
    if not cols:
        raise ValueError("dynamic state surface has no channels after the selected ablations")
    return edge_attr[:, cols]


def _fixed_hist(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    values = values.clamp(float(edges[0]), float(edges[-1]))
    bucket = torch.bucketize(values, edges[1:-1], right=False)
    counts = torch.bincount(bucket, minlength=int(edges.numel() - 1)).to(dtype=torch.float32)
    return counts / counts.sum().clamp_min(1.0)


def behavior_state_vector(rows: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    if args.state_surface != "dynamic":
        raise ValueError("--state-level behavior requires --state-surface dynamic")

    rows = rows.detach().to(dtype=torch.float32)
    bins_log_tau = torch.tensor(
        [-5.0, -2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0, 5.0],
        device=rows.device,
        dtype=rows.dtype,
    )

    parts: List[torch.Tensor] = []
    if args.edge_feature_set == "compact3":
        # dynamic compact3 rows = log_tau_rel, is_in_incumbent
        log_tau = rows[:, 0].clamp(float(bins_log_tau[0]), float(bins_log_tau[-1]))
        incumbent = rows[:, 1]
        parts.extend([
            _fixed_hist(log_tau, bins_log_tau),
            torch.stack([
                incumbent.mean(),
                incumbent.std(unbiased=False),
                (log_tau > 0.5).to(torch.float32).mean(),
                (log_tau < -0.5).to(torch.float32).mean(),
                log_tau.abs().mean(),
            ]),
        ])
    else:
        cursor = 0
        if not args.ablation_pheromone_features:
            tau_cv = rows[:, cursor]
            log_tau = rows[:, cursor + 1].clamp(float(bins_log_tau[0]), float(bins_log_tau[-1]))
            cursor += 2
            bins_tau_cv = torch.tensor(
                [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                device=rows.device,
                dtype=rows.dtype,
            )
            parts.extend([
                _fixed_hist(log_tau, bins_log_tau),
                _fixed_hist(tau_cv.clamp(float(bins_tau_cv[0]), float(bins_tau_cv[-1])), bins_tau_cv),
                torch.stack([
                    (log_tau > 0.5).to(torch.float32).mean(),
                    (log_tau < -0.5).to(torch.float32).mean(),
                    log_tau.abs().mean(),
                ]),
            ])
        if not args.ablation_incumbent_features:
            incumbent_cols = rows[:, cursor:cursor + 3]
            parts.append(torch.cat((incumbent_cols.mean(dim=0), incumbent_cols.std(dim=0, unbiased=False)), dim=0))

    if not parts:
        raise ValueError("behavior state surface has no dynamic channels after the selected ablations")
    return torch.cat(parts, dim=0).unsqueeze(0).cpu().numpy()


def edge_rows_to_state_vectors(rows: torch.Tensor, state_level: str, args: argparse.Namespace) -> np.ndarray:
    rows = rows.detach().to(dtype=torch.float32)
    if state_level == "behavior":
        return behavior_state_vector(rows, args)
    if state_level == "edge":
        return rows.cpu().numpy()
    mean = rows.mean(dim=0)
    std = rows.std(dim=0, unbiased=False)
    q10 = torch.quantile(rows, 0.10, dim=0)
    q50 = torch.quantile(rows, 0.50, dim=0)
    q90 = torch.quantile(rows, 0.90, dim=0)
    return torch.cat((mean, std, q10, q50, q90), dim=0).unsqueeze(0).cpu().numpy()


def step_snapshot(rows: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    """Compute a fixed-size process snapshot from dynamic edge features at one step.

    Returns 5 scale-free indicators:
      mu_tau, sigma_tau, inc_frac, reinf_frac, suppr_frac
    """
    rows = rows.detach().to(dtype=torch.float32)
    if args.edge_feature_set == "compact3":
        log_tau = rows[:, 0]
        incumbent = rows[:, 1]
    else:
        cursor = 0
        if not args.ablation_pheromone_features:
            log_tau = rows[:, cursor + 1]
            cursor += 2
        else:
            log_tau = torch.zeros(rows.size(0), device=rows.device)
        if not args.ablation_incumbent_features:
            incumbent = rows[:, cursor]
        else:
            incumbent = torch.zeros(rows.size(0), device=rows.device)
    return np.array([
        float(log_tau.mean()),
        float(log_tau.std(unbiased=False)),
        float(incumbent.mean()),
        float((log_tau > 0.5).float().mean()),
        float((log_tau < -0.5).float().mean()),
    ], dtype=np.float32)


def trajectory_vector(snapshots: List[np.ndarray]) -> np.ndarray:
    """Summarise an H-length sequence of step snapshots into one trajectory vector.

    For each of the 5 channels: initial value, final value, OLS slope,
    residual std around the trend, and overall mean — 25 features total.
    """
    mat = np.stack(snapshots, axis=0)  # (H, 5)
    H = mat.shape[0]
    t = np.arange(H, dtype=np.float64)
    t_centered = t - t.mean()
    t_var = (t_centered ** 2).sum()

    parts = []
    for c in range(mat.shape[1]):
        y = mat[:, c].astype(np.float64)
        initial = float(y[0])
        final = float(y[-1])
        mean_val = float(y.mean())
        if H > 1 and t_var > 0:
            slope = float((t_centered * (y - y.mean())).sum() / t_var)
            residual = float(np.std(y - (y.mean() + slope * t_centered)))
        else:
            slope = 0.0
            residual = 0.0
        parts.extend([initial, final, slope, residual, mean_val])
    return np.array(parts, dtype=np.float32)


def model_prior_from_pyg(model: torch.nn.Module, pyg: Any, aco: Any) -> torch.Tensor:
    prior_output = model(pyg)
    if prior_output.dim() == 2:
        return net.output_to_multi_sparse_priors(prior_output, aco.n, aco.k)
    return net.output_to_sparse_prior(prior_output, aco.n, aco.k)


def compute_anneal_factor(inner: int, args: argparse.Namespace) -> float:
    if args.no_anneal:
        return 1.0
    if args.mini_H > 1:
        ratio = inner / float(args.mini_H - 1)
        return args.gamma * (1.0 - ratio) + args.min_gamma * ratio
    return args.gamma


def collect_states(
    args: argparse.Namespace,
    model: Optional[torch.nn.Module],
) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[np.ndarray]]:
    """Returns (state_vectors, metadata_rows, geometry_vectors_or_None)."""
    vectors: List[np.ndarray] = []
    geo_vectors: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    need_geo = bool(getattr(args, "controls", False)) and args.state_level == "edge"

    for domain_idx, domain in enumerate(args.domains):
        spec = domain_config(domain, args)
        for inst_idx in range(args.instances_per_domain):
            inst_seed = args.seed + 100000 * domain_idx + inst_idx
            set_seed(inst_seed)
            instance = gen_instance(domain, args)
            aco = build_aco(instance, args, inst_seed)

            # Normalize coordinates for model input (matching test.py)
            norm_instance = instance
            if model is not None:
                if args.problem == "tsp":
                    coords = instance
                else:
                    coords = instance[0]
                c_min = coords.min(dim=0)[0]
                c_max = coords.max(dim=0)[0]
                scale = (c_max - c_min).max()
                if scale < 1e-6:
                    scale = 1.0
                norm_coords = (coords - c_min) / scale
                if args.problem == "tsp":
                    norm_instance = norm_coords
                else:
                    norm_instance = (norm_coords, instance[1], instance[2])

            is_trajectory = args.state_level == "trajectory"
            traj_snapshots: List[np.ndarray] = []
            build_instance = norm_instance if model is not None else instance

            for outer in range(args.H):
                pyg = build_state_tensor(aco, build_instance, args)
                prior = None
                with torch.no_grad():
                    if model is None:
                        edge_rows = pyg.edge_attr
                    elif args.state_surface == "dynamic":
                        pyg = net.move_pyg_to_module_device(model, pyg)
                        edge_rows = pyg.edge_attr
                        prior = model_prior_from_pyg(model, pyg, aco)
                    else:
                        pyg = net.move_pyg_to_module_device(model, pyg)
                        edge_rows = model.emb_net(pyg.x, pyg.edge_index, pyg.edge_attr)
                        prior = model_prior_from_pyg(model, pyg, aco)

                    if args.state_surface == "dynamic":
                        edge_rows = dynamic_edge_rows(edge_rows, args)
                    elif model is None:
                        edge_rows = pyg.edge_attr

                if is_trajectory:
                    traj_snapshots.append(step_snapshot(edge_rows, args))
                else:
                    idx_t = None
                    if args.state_level == "edge" and edge_rows.size(0) > args.edge_samples_per_state:
                        sample_idx = rng.choice(edge_rows.size(0), size=args.edge_samples_per_state, replace=False)
                        idx_t = torch.as_tensor(sample_idx, device=edge_rows.device, dtype=torch.long)
                        edge_rows = edge_rows[idx_t]

                    if need_geo:
                        src = pyg.edge_index[0]
                        dst = pyg.edge_index[1]
                        if idx_t is not None:
                            src = src[idx_t]
                            dst = dst[idx_t]
                        geo_feat = torch.cat([pyg.x[src], pyg.x[dst]], dim=1)
                        geo_vectors.append(geo_feat.detach().cpu().numpy())

                    state_vectors = edge_rows_to_state_vectors(edge_rows, args.state_level, args)
                    base_meta = {
                        "domain": domain,
                        "domain_id": domain_idx,
                        "domain_spec": json.dumps(spec, sort_keys=True, separators=(",", ":")),
                        "instance_id": inst_idx,
                        "outer": outer,
                        "step": outer,
                        "mini_H": int(args.mini_H),
                        "problem": args.problem,
                        "n_nodes": instance_node_count(instance, args.problem),
                        "representation": args.representation,
                        "state_surface": args.state_surface,
                        "state_level": args.state_level,
                    }
                    for local_idx, vec in enumerate(state_vectors):
                        vectors.append(vec.reshape(1, -1))
                        meta = dict(base_meta)
                        meta["sample_id"] = local_idx
                        rows.append(meta)

                for inner in range(args.mini_H):
                    prior_scale = compute_anneal_factor(inner, args) if prior is not None else 1.0
                    sample_and_update(aco, args, prior, prior_scale=prior_scale)

            if is_trajectory:
                vec = trajectory_vector(traj_snapshots)
                vectors.append(vec.reshape(1, -1))
                rows.append({
                    "domain": domain,
                    "domain_id": domain_idx,
                    "domain_spec": json.dumps(spec, sort_keys=True, separators=(",", ":")),
                    "instance_id": inst_idx,
                    "outer": -1,
                    "step": -1,
                    "mini_H": int(args.mini_H),
                    "problem": args.problem,
                    "n_nodes": instance_node_count(instance, args.problem),
                    "representation": args.representation,
                    "state_surface": args.state_surface,
                    "state_level": args.state_level,
                    "sample_id": 0,
                })

    x = np.concatenate(vectors, axis=0).astype(np.float32)
    x_geo = np.concatenate(geo_vectors, axis=0).astype(np.float32) if geo_vectors else None
    return x, rows, x_geo


def _head_of_ant(ant_idx: int, head_counts: List[int]) -> int:
    cumul = 0
    for h, cnt in enumerate(head_counts):
        cumul += cnt
        if ant_idx < cumul:
            return h
    return len(head_counts) - 1


def collect_behavior_data(
    args: argparse.Namespace,
    model: torch.nn.Module,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Collect per-head guidance fields, per-step wins, and pairwise divergence.

    Returns (x, y_head, groups, win_matrix, divergence) where
      x          – (N, 5) pooled head-column vectors for behavior classification,
      y_head     – head index label per row of x,
      groups     – instance group tag per row of x,
      win_matrix – (H, num_heads) counts of best-ant attribution per macro-step,
      divergence – dict with cosine/top-K overlap matrices and scalar summaries.
    """
    model.eval()
    vectors: List[np.ndarray] = []
    head_ids: List[int] = []
    group_tags: List[str] = []
    n_heads: Optional[int] = None
    win_counts: Optional[np.ndarray] = None  # (H, num_heads)
    cos_sum: Optional[np.ndarray] = None     # (num_heads, num_heads) running sum
    topk_sum: Optional[np.ndarray] = None    # (num_heads, num_heads) running sum
    n_snapshots = 0

    for domain_idx, domain in enumerate(args.domains):
        for inst_idx in range(args.instances_per_domain):
            inst_seed = args.seed + 100000 * domain_idx + inst_idx
            set_seed(inst_seed)
            instance = gen_instance(domain, args)
            aco = build_aco(instance, args, inst_seed)

            if args.problem == "tsp":
                coords = instance
            else:
                coords = instance[0]
            c_min = coords.min(dim=0)[0]
            c_max = coords.max(dim=0)[0]
            scale = (c_max - c_min).max()
            if scale < 1e-6:
                scale = 1.0
            norm_coords = (coords - c_min) / scale
            if args.problem == "tsp":
                norm_instance = norm_coords
            else:
                norm_instance = (norm_coords, instance[1], instance[2])

            for outer in range(args.H):
                pyg = build_state_tensor(aco, norm_instance, args)
                with torch.no_grad():
                    pyg = net.move_pyg_to_module_device(model, pyg)
                    emb = model.emb_net(pyg.x, pyg.edge_index, pyg.edge_attr)
                    head_logits = model._decode_heads(emb)  # (E, H)
                    prior = model_prior_from_pyg(model, pyg, aco)

                if n_heads is None:
                    n_heads = int(head_logits.shape[1])
                    win_counts = np.zeros((args.H, n_heads), dtype=np.int64)

                # --- pairwise head divergence ---
                logits_f = head_logits.detach().float()  # (E, H)

                # Residual cosine: subtract head-mean to isolate per-head signal
                head_mean = logits_f.mean(dim=1, keepdim=True)  # (E, 1)
                residual = logits_f - head_mean  # (E, H)
                res_norms = residual.norm(dim=0, keepdim=True).clamp_min(1e-8)
                res_normed = residual / res_norms
                cos_mat = (res_normed.T @ res_normed).cpu().numpy()  # (H, H)

                # Top-K overlap on raw logits
                top_frac = 0.05
                k_top = max(1, int(logits_f.shape[0] * top_frac))
                topk_mat_step = np.zeros((n_heads, n_heads), dtype=np.float64)
                top_sets = []
                for h in range(n_heads):
                    top_sets.append(set(torch.topk(logits_f[:, h], k_top).indices.cpu().tolist()))
                for hi in range(n_heads):
                    for hj in range(n_heads):
                        topk_mat_step[hi, hj] = len(top_sets[hi] & top_sets[hj]) / k_top

                if cos_sum is None:
                    cos_sum = np.zeros_like(cos_mat)
                    topk_sum = np.zeros_like(topk_mat_step)
                cos_sum += cos_mat
                topk_sum += topk_mat_step
                n_snapshots += 1

                # --- behaviour feature vectors (one per head) ---
                for h in range(n_heads):
                    col = logits_f[:, h]
                    vec = torch.stack([
                        col.mean(),
                        col.std(unbiased=False),
                        torch.quantile(col, 0.10),
                        torch.quantile(col, 0.50),
                        torch.quantile(col, 0.90),
                    ]).cpu().numpy()
                    vectors.append(vec.reshape(1, -1))
                    head_ids.append(h)
                    group_tags.append(f"d{domain_idx}:i{inst_idx}:t{outer}")

                # --- inner loop: sample, track winner, update pheromone ---
                head_counts = [args.n_ants // n_heads] * n_heads
                for i in range(args.n_ants % n_heads):
                    head_counts[i] += 1

                prior_np = (
                    prior.detach().to(device="cpu", dtype=torch.float32)
                    .contiguous().numpy()
                    if prior is not None and prior.dim() == 3
                    else None
                )

                for inner in range(args.mini_H):
                    prior_scale = compute_anneal_factor(inner, args) if prior is not None else 1.0

                    if prior_np is not None and hasattr(aco, "sample_mixed_priors"):
                        out = aco.sample_mixed_priors(
                            prior_np,
                            require_prob=False,
                            parallel_traced=True,
                            head_counts=head_counts,
                            prior_scale=prior_scale,
                        )
                        costs, solution_rows = out[0], out[1]
                    else:
                        sample_and_update(aco, args, prior, prior_scale=prior_scale)
                        continue

                    costs_np = np.asarray(costs)
                    best_idx = int(np.argmin(costs_np))
                    winner = _head_of_ant(best_idx, head_counts)
                    win_counts[outer, winner] += 1
                    aco.update_pheromone(solution_rows[best_idx], float(costs_np[best_idx]))

    x = np.concatenate(vectors, axis=0).astype(np.float32)
    y = np.array(head_ids, dtype=np.int64)
    groups = np.array(group_tags)
    if win_counts is None:
        win_counts = np.zeros((args.H, 1), dtype=np.int64)

    divergence: Dict[str, Any] = {}
    if n_snapshots > 0 and cos_sum is not None and topk_sum is not None:
        cos_mean = cos_sum / n_snapshots
        topk_mean = topk_sum / n_snapshots
        mask = ~np.eye(cos_mean.shape[0], dtype=bool)
        divergence = {
            "cosine_similarity_matrix": cos_mean.tolist(),
            "topk_overlap_matrix": topk_mean.tolist(),
            "mean_offdiag_cosine": float(cos_mean[mask].mean()),
            "mean_offdiag_topk_overlap": float(topk_mean[mask].mean()),
            "n_snapshots": n_snapshots,
            "top_frac": 0.05,
        }
    return x, y, groups, win_counts, divergence


def standardize_train_test(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, int]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for label in np.unique(y):
        label_groups = np.unique(groups[y == label])
        rng.shuffle(label_groups)
        n_test_groups = max(1, int(round(len(label_groups) * test_fraction)))
        if len(label_groups) - n_test_groups < 1:
            n_test_groups = max(0, len(label_groups) - 1)
        test_groups = set(label_groups[:n_test_groups].tolist())
        label_idx = np.flatnonzero(y == label)
        test_idx.extend([int(i) for i in label_idx if groups[i] in test_groups])
        train_idx.extend([int(i) for i in label_idx if groups[i] not in test_groups])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    reused_train_for_test = False
    if not test_idx:
        test_idx = list(train_idx)
        reused_train_for_test = True
    train_group_count = int(len(np.unique(groups[train_idx]))) if train_idx else 0
    test_group_count = int(len(np.unique(groups[test_idx]))) if test_idx else 0
    return (
        x[train_idx],
        y[train_idx],
        x[test_idx],
        y[test_idx],
        reused_train_for_test,
        train_group_count,
        test_group_count,
    )


def _build_classifier(in_dim: int, n_classes: int, args: argparse.Namespace) -> torch.nn.Module:
    if args.classifier == "mlp":
        h = args.classifier_hidden
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, n_classes),
        ).to(args.device)
    return torch.nn.Linear(in_dim, n_classes).to(args.device)


def _fit_and_eval(
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[float, float, np.ndarray]:
    """Returns (train_acc, test_acc, per-sample correctness on test set)."""
    clf = _build_classifier(int(xtr.size(1)), int(ytr.max().item()) + 1, args)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
    for _ in range(args.classifier_epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(clf(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = float((clf(xtr).argmax(1) == ytr).float().mean().item())
        correct = (clf(xte).argmax(1) == yte).cpu().numpy() if yte.numel() else np.array([], dtype=bool)
        test_acc = float(correct.mean()) if correct.size else float("nan")
    return train_acc, test_acc, correct


def _bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap CI for accuracy from a boolean correctness vector."""
    if correct.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot, dtype=np.float64)
    n = len(correct)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[i] = correct[idx].mean()
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def train_domain_classifier(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    (
        x_train,
        y_train,
        x_test,
        y_test,
        reused_train_for_test,
        train_group_count,
        test_group_count,
    ) = standardize_train_test(
        x, y, groups, args.test_fraction, args.seed
    )
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    xtr = torch.as_tensor(x_train, dtype=torch.float32, device=args.device)
    ytr = torch.as_tensor(y_train, dtype=torch.long, device=args.device)
    xte = torch.as_tensor(x_test, dtype=torch.float32, device=args.device)
    yte = torch.as_tensor(y_test, dtype=torch.long, device=args.device)

    train_acc, test_acc, correct = _fit_and_eval(xtr, ytr, xte, yte, args)
    ci_lo, ci_hi = _bootstrap_ci(correct, seed=args.seed)

    counts = np.bincount(y, minlength=int(y.max()) + 1)
    chance = float(counts.max() / counts.sum())

    result: Dict[str, Any] = {
        "classifier": args.classifier,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_accuracy_ci95": [ci_lo, ci_hi],
        "majority_chance": chance,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_instances": train_group_count,
        "n_test_instances": test_group_count,
        "split": "held_out_instances",
        "evaluation_reuses_train": bool(reused_train_for_test),
        "class_counts": counts.astype(int).tolist(),
    }

    if args.permutation_trials > 0:
        perm_accs: List[float] = []
        for _ in range(args.permutation_trials):
            y_perm = ytr[torch.randperm(ytr.size(0), device=ytr.device)]
            _, pa, _ = _fit_and_eval(xtr, y_perm, xte, yte, args)
            perm_accs.append(pa)
        perm_arr = np.array(perm_accs)
        p_value = float((perm_arr >= test_acc).mean())
        result["permutation_trials"] = int(args.permutation_trials)
        result["permutation_mean"] = float(perm_arr.mean())
        result["permutation_std"] = float(perm_arr.std())
        result["permutation_p_value"] = p_value

    return result


def pca_2d(x: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    x0 = x.astype(np.float64)
    x0 = x0 - x0.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x0, full_matrices=False)
    coords = x0 @ vt[:2].T
    denom = float(np.sum(s ** 2))
    explained = ((s[:2] ** 2) / denom).tolist() if denom > 0 else [0.0, 0.0]
    return coords.astype(np.float32), explained


def save_scatter(path: Path, coords: np.ndarray, labels: Sequence[str], title: str) -> None:
    import matplotlib.pyplot as plt

    unique_labels = list(dict.fromkeys(labels))
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=160)
    for label in unique_labels:
        mask = np.asarray([x == label for x in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.72, label=label, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def try_save_umap(path: Path, x: np.ndarray, labels: Sequence[str], seed: int) -> Optional[str]:
    try:
        import umap  # type: ignore
    except Exception as exc:
        return f"UMAP skipped: {exc}"
    reducer = umap.UMAP(n_components=2, random_state=seed)
    coords = reducer.fit_transform(x)
    save_scatter(path, coords, labels, "UMAP of DyNACO search states")
    return None


def write_metadata(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, args: argparse.Namespace, metrics: Dict[str, Any], pca_var: List[float], umap_note: Optional[str], control_results: Optional[Dict[str, Any]] = None, behavior_metrics: Optional[Dict[str, Any]] = None) -> None:
    chance = metrics["majority_chance"]
    acc = metrics["test_accuracy"]
    if abs(acc - chance) <= 0.10:
        verdict = "near chance"
    elif acc > chance:
        verdict = "above chance"
    else:
        verdict = "below chance"
    lines = [
        "# State Recurrence Experiment",
        "",
        f"- Problem: {args.problem}",
        f"- Domains: {', '.join(args.domains)}",
        f"- Domain specs: {json.dumps({label: domain_config(label, args) for label in args.domains}, sort_keys=True)}",
        f"- Search schedule: H={args.H}, mini_H={args.mini_H}",
        f"- Representation: {args.representation} / {args.state_surface} / {args.state_level}",
        f"- Classifier: {metrics.get('classifier', 'linear')}",
        f"- Split: {metrics['split']} ({metrics['n_train_instances']} train instances, {metrics['n_test_instances']} test instances)",
        f"- States: train={metrics['n_train']}, test={metrics['n_test']}",
        f"- Domain-classifier test accuracy: {acc:.4f} [{metrics.get('test_accuracy_ci95', [float('nan'), float('nan')])[0]:.4f}, {metrics.get('test_accuracy_ci95', [float('nan'), float('nan')])[1]:.4f}]",
        f"- Majority-class chance: {chance:.4f}",
        f"- Interpretation: classifier is {verdict}; overlap is stronger when accuracy stays near chance.",
        f"- PCA explained variance: PC1={pca_var[0]:.4f}, PC2={pca_var[1]:.4f}",
    ]
    if "permutation_p_value" in metrics:
        lines.append(
            f"- Permutation test: p={metrics['permutation_p_value']:.4f} "
            f"(mean={metrics['permutation_mean']:.4f}, std={metrics['permutation_std']:.4f}, "
            f"trials={metrics['permutation_trials']})"
        )
    if control_results:
        lines.append("")
        lines.append("## Controls")
        lines.append("All controls use the same instances, search states, and sampled edges; only features or labels differ.")
        if "positive_control_phase" in control_results:
            pc = control_results["positive_control_phase"]
            ci = pc.get("test_accuracy_ci95", [float("nan"), float("nan")])
            lines.append(
                f"- Positive control (extremal search phase, dynamic features): "
                f"accuracy={pc['test_accuracy']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], chance={pc['majority_chance']:.4f}"
            )
        if "geometry_baseline" in control_results:
            gb = control_results["geometry_baseline"]
            ci = gb.get("test_accuracy_ci95", [float("nan"), float("nan")])
            lines.append(
                f"- Geometry baseline (edge-endpoint coordinates, domain labels): "
                f"accuracy={gb['test_accuracy']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], chance={gb['majority_chance']:.4f}"
            )
    if metrics.get("evaluation_reuses_train"):
        lines.append("- Split warning: too few states for a held-out class split; diagnostic accuracy reuses training states.")
    if behavior_metrics is not None:
        lines.append("")
        lines.append("## Behavior classification (head ID)")
        bci = behavior_metrics.get("test_accuracy_ci95", [float("nan"), float("nan")])
        lines.append(
            f"- Behavior classifier: accuracy={behavior_metrics['test_accuracy']:.4f} "
            f"[{bci[0]:.4f}, {bci[1]:.4f}], chance={behavior_metrics['majority_chance']:.4f}"
        )
        if "mean_offdiag_cosine" in behavior_metrics:
            lines.append(
                f"- Head residual cosine similarity (off-diagonal mean): "
                f"{behavior_metrics['mean_offdiag_cosine']:.4f}"
            )
            lines.append(
                f"- Head top-5% edge overlap (off-diagonal mean): "
                f"{behavior_metrics['mean_offdiag_topk_overlap']:.4f}"
            )
        if "cramers_v" in behavior_metrics:
            lines.append(
                f"- Phase-behavior coupling (Cramér's V): {behavior_metrics['cramers_v']:.4f} "
                f"(chi² p={behavior_metrics['chi2_p']:.4g})"
            )
            mws = behavior_metrics["mean_win_step"]
            lines.append(f"- Mean win-step per head: {', '.join(f'{v:.2f}' for v in mws)}")
    if umap_note:
        lines.append(f"- {umap_note}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    model = None
    if args.representation == "embedding" or args.behavior_classification:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --representation embedding or --behavior-classification")
        model = load_embedding_model(args, edge_feature_count(args))

    x, metadata, x_geo = collect_states(args, model)
    y = np.asarray([row["domain_id"] for row in metadata], dtype=np.int64)
    groups = np.asarray([f"{row['domain_id']}:{row['instance_id']}" for row in metadata])
    labels = [row["domain"] for row in metadata]

    metrics = train_domain_classifier(x, y, groups, args)
    pca_coords, pca_var = pca_2d(x)

    # --- Controls -----------------------------------------------------------
    control_results: Dict[str, Any] = {}
    if args.controls and args.state_level == "edge":
        steps = np.asarray([row["outer"] for row in metadata])

        # Positive control: extremal search steps (first 20% vs last 20%,
        # dropping the ambiguous middle).  Step 0 has near-uniform pheromone;
        # the final steps have concentrated pheromone and a stable incumbent.
        lo_cutoff = max(1, int(math.ceil(args.H * 0.2)))
        hi_cutoff = args.H - lo_cutoff
        extremal_mask = (steps < lo_cutoff) | (steps >= hi_cutoff)
        if extremal_mask.sum() > 0:
            x_ext = x[extremal_mask]
            y_ext = (steps[extremal_mask] >= hi_cutoff).astype(np.int64)
            groups_ext = np.asarray([
                f"phase{int(steps[i] >= hi_cutoff)}:{row['domain_id']}:{row['instance_id']}"
                for i, row in enumerate(metadata) if extremal_mask[i]
            ])
            phase_metrics = train_domain_classifier(x_ext, y_ext, groups_ext, args)
            control_results["positive_control_phase"] = phase_metrics

        # Geometry baseline: edge-endpoint coordinates with domain labels.
        # Uses the same instances, same search states, same sampled edges —
        # only the features differ (coordinates instead of pheromone/incumbent).
        if x_geo is not None:
            geo_metrics = train_domain_classifier(x_geo, y, groups, args)
            control_results["geometry_baseline"] = geo_metrics

    # --- Behavior classification -----------------------------------------------
    behavior_metrics: Optional[Dict[str, Any]] = None
    win_matrix: Optional[np.ndarray] = None
    if args.behavior_classification and model is not None:
        bx, by, bg, win_matrix, divergence = collect_behavior_data(args, model)
        behavior_metrics = train_domain_classifier(bx, by, bg, args)
        if divergence:
            behavior_metrics.update(divergence)

        # Phase-behavior coupling: Cramér's V between head and macro-step
        if win_matrix is not None and win_matrix.sum() > 0:
            # Chi-squared test and Cramér's V (no scipy dependency)
            observed = win_matrix.astype(np.float64)
            n_obs = float(observed.sum())
            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            expected = row_sums * col_sums / max(n_obs, 1.0)
            expected = np.maximum(expected, 1e-10)
            chi2 = float(((observed - expected) ** 2 / expected).sum())
            k = min(observed.shape)
            cramers_v = float(np.sqrt(chi2 / (n_obs * max(k - 1, 1))))
            dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
            # p-value approximation: use survival function of chi² distribution
            # For large dof, use normal approximation; otherwise flag as "< 0.001" if chi2 >> dof
            p_chi2 = float(np.exp(-0.5 * max(chi2 - dof, 0))) if dof > 0 else 1.0

            # Per-head mean win-step (weighted average step index)
            step_idx = np.arange(win_matrix.shape[0], dtype=np.float64)
            head_totals = win_matrix.sum(axis=0).astype(np.float64)
            head_totals = np.maximum(head_totals, 1.0)
            mean_win_step = (step_idx[:, None] * win_matrix).sum(axis=0) / head_totals

            behavior_metrics["win_matrix"] = win_matrix.tolist()
            behavior_metrics["cramers_v"] = cramers_v
            behavior_metrics["chi2_p"] = float(p_chi2)
            behavior_metrics["mean_win_step"] = mean_win_step.tolist()

    # --- Print results -------------------------------------------------------
    def _fmt(label: str, m: Dict[str, Any]) -> str:
        ci = m.get("test_accuracy_ci95", [float("nan"), float("nan")])
        return f"  {label:<40s} {m['test_accuracy']:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]"

    print(_fmt("domain_classifier (dynamic)", metrics))
    if "positive_control_phase" in control_results:
        print(_fmt("positive_control (search phase)", control_results["positive_control_phase"]))
    if "geometry_baseline" in control_results:
        print(_fmt("geometry_baseline (coordinates)", control_results["geometry_baseline"]))
    if behavior_metrics is not None:
        print(_fmt("behavior_classifier (head ID)", behavior_metrics))
        if "mean_offdiag_cosine" in behavior_metrics:
            cos = behavior_metrics["mean_offdiag_cosine"]
            topk = behavior_metrics["mean_offdiag_topk_overlap"]
            print(f"  {'head residual cosine (off-diag mean)':<40s} {cos:.4f}")
            print(f"  {'head top-5% edge overlap (off-diag)':<40s} {topk:.4f}")
        if "cramers_v" in behavior_metrics:
            print(f"  {'phase×behavior Cramér V':<40s} {behavior_metrics['cramers_v']:.4f}  (p={behavior_metrics['chi2_p']:.4g})")
            mws = behavior_metrics["mean_win_step"]
            print(f"  {'mean win-step per head':<40s} {', '.join(f'{v:.2f}' for v in mws)}")

    # --- Save ----------------------------------------------------------------
    np.savez_compressed(
        run_dir / "states.npz",
        x=x,
        y=y,
        pca=pca_coords,
        domains=np.asarray(args.domains),
        problem=np.asarray(args.problem),
        groups=groups,
    )
    write_metadata(run_dir / "state_metadata.csv", metadata)
    all_metrics = dict(metrics)
    if control_results:
        all_metrics["controls"] = control_results
    if behavior_metrics is not None:
        all_metrics["behavior_classifier"] = behavior_metrics
    (run_dir / "classifier_metrics.json").write_text(json.dumps(all_metrics, indent=2, sort_keys=True) + "\n")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n")
    save_scatter(run_dir / "pca.png", pca_coords, labels, "PCA of DyNACO search states")
    umap_note = try_save_umap(run_dir / "umap.png", x, labels, args.seed)
    write_summary(run_dir / "SUMMARY.md", args, metrics, pca_var, umap_note, control_results, behavior_metrics)
    print(f"Wrote state recurrence experiment to {run_dir}")


if __name__ == "__main__":
    main()
