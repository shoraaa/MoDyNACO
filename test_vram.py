#!/usr/bin/env python3
import argparse
import csv
import gc
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

import faco
import utils
from net import Net
from utils import infer_instance


DEFAULT_SCALES = [1000, 5000, 10000, 50000, 100000]
DEFAULT_PROBLEMS = ["tsp", "cvrp"]
DEFAULT_METHODS = ["auto"]
DEFAULT_VARIANTS = ["dynamic", "static"]

DEFAULT_CONFIG = {
    "H": 10,
    "mini_H": 100,
    "n_ants": 100,
    "k_sparse": 32,
    "rho": 0.1,
    "min_new_edges": 12,
    "gamma": 1.0,
    "min_gamma": 0.0,
    "alg": "faco",
    "warmup_ratio": 0.5,
    "no_anneal": False,
    "disable_heuristic": False,
    "no_local_search": False,
    "no_smooth_mmas": False,
    "no_extend_ls": False,
    "no_normalized_heuristic": False,
    "no_dynamic_feats": False,
    "no_logit_net": False,
    "L": 0,
    "alpha": 1.0,
    "beta": 1.0,
    "ls_scope": "localized",
    "ls_budget": "truncated",
    "ls_max_opt": 0,
    "seed": 1234,
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    use_model: bool
    use_heuristic_only: bool
    no_anneal: bool
    use_mix: bool


@dataclass(frozen=True)
class VariantSpec:
    name: str
    no_dynamic_feats: bool
    feature_indices: Tuple[int, ...]
    ablation_pheromone: bool
    ablation_incumbent: bool

    @property
    def edge_feats(self) -> int:
        return len(self.feature_indices)


METHOD_SPECS: Dict[str, MethodSpec] = {
    "base": MethodSpec("base", use_model=False, use_heuristic_only=True, no_anneal=False, use_mix=False),
    "model_anneal": MethodSpec("model_anneal", use_model=True, use_heuristic_only=False, no_anneal=False, use_mix=False),
    "model_no_anneal": MethodSpec("model_no_anneal", use_model=True, use_heuristic_only=False, no_anneal=True, use_mix=False),
    "mix_anneal": MethodSpec("mix_anneal", use_model=True, use_heuristic_only=False, no_anneal=False, use_mix=True),
    "mix_no_anneal": MethodSpec("mix_no_anneal", use_model=True, use_heuristic_only=False, no_anneal=True, use_mix=True),
}

VARIANT_SPECS: Dict[str, VariantSpec] = {
    "dynamic": VariantSpec("dynamic", no_dynamic_feats=False, feature_indices=(0, 1, 2, 3, 4, 5), ablation_pheromone=False, ablation_incumbent=False),
    "static": VariantSpec("static", no_dynamic_feats=True, feature_indices=(0,), ablation_pheromone=True, ablation_incumbent=True),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _parse_csv_list(raw: str, caster) -> List[Any]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return [caster(part) for part in parts]


def _maybe_override(base: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        base[key] = value


def _resolve_method_name(problem: str, H: int, requested_method: str) -> str:
    if requested_method != "auto":
        return requested_method
    if problem == "tsp":
        return "model_anneal" if int(H) < 50 else "mix_anneal"
    return "model_no_anneal" if int(H) < 50 else "mix_no_anneal"


def _resolve_checkpoint(problem: str, scale: int, checkpoint_root: Path, checkpoint_scale: Optional[int]) -> Path:
    lookup_scale = int(checkpoint_scale) if checkpoint_scale is not None else int(scale)
    candidate_dirs = [
        checkpoint_root / problem / f"n{lookup_scale}",
        Path("pretrained") / problem / f"n{lookup_scale}",
        Path("checkpoints"),
    ]
    patterns = [
        f"{problem}_n{lookup_scale}_*_best.pt",
        f"{problem}_n{lookup_scale}_best.pt",
        "*best.pt",
    ]

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        for pattern in patterns:
            matches = sorted(glob.glob(str(candidate_dir / pattern)))
            if matches:
                return Path(matches[0]).resolve()

    raise FileNotFoundError(
        f"Could not auto-detect checkpoint for problem={problem}, "
        f"scale={scale}, checkpoint_scale={checkpoint_scale}, root={checkpoint_root}"
    )


def _load_state_dict(path: Path, device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = ckpt.get("config", {})
    return state_dict, config


def _filtered_state_dict_for_variant(
    state_dict: Dict[str, Any],
    variant: VariantSpec,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}
    edge_weight_key = "emb_net.e_lin0.weight"
    edge_bias_key = "emb_net.e_lin0.bias"
    checkpoint_edge_feats = None
    if edge_weight_key in state_dict:
        checkpoint_edge_feats = int(state_dict[edge_weight_key].shape[1])

    for key, value in state_dict.items():
        filtered[key] = value.clone() if torch.is_tensor(value) else value

    if checkpoint_edge_feats is None or checkpoint_edge_feats == variant.edge_feats:
        return filtered

    if checkpoint_edge_feats < max(variant.feature_indices) + 1:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has edge_feats={checkpoint_edge_feats}, "
            f"which cannot satisfy variant {variant.name} with indices {variant.feature_indices}"
        )

    filtered[edge_weight_key] = filtered[edge_weight_key][:, list(variant.feature_indices)].contiguous()
    if edge_bias_key in filtered:
        filtered[edge_bias_key] = filtered[edge_bias_key].contiguous()
    return filtered


def _build_runtime(problem: str, config: Dict[str, Any], device: str, method: MethodSpec, variant: VariantSpec) -> Tuple[Any, Any, Any]:
    model = None
    feats = 2 if problem == "tsp" else 1
    if method.use_model:
        checkpoint = Path(config["checkpoint"])
        state_dict, ckpt_config = _load_state_dict(checkpoint, device)
        config.update(ckpt_config)
        config["checkpoint"] = str(checkpoint)

        edge_feats = variant.edge_feats
        if "emb_net.v_lin0.weight" in state_dict:
            feats = int(state_dict["emb_net.v_lin0.weight"].shape[1])
        state_dict = _filtered_state_dict_for_variant(state_dict, variant, checkpoint)

        model = Net(
            feats=feats,
            edge_feats=edge_feats,
            logit_net=not bool(config.get("no_logit_net", False)),
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

    alg = str(config.get("alg", "faco")).lower()
    if problem == "tsp":
        build_fn = utils.build_pyg_data_tsp
        aco_class = faco.ACO_TSP if alg == "mmas" else faco.MFACO_TSP
    else:
        build_fn = _build_pyg_data_cvrp_compat_4d if feats == 4 else utils.build_pyg_data_cvrp
        aco_class = faco.ACO_CVRP if alg == "mmas" else faco.MFACO_CVRP
    return model, build_fn, aco_class


def _build_pyg_data_cvrp_compat_4d(
    aco,
    coords,
    demand,
    device,
    ablation_pheromone: bool = False,
    ablation_incumbent: bool = False,
    dynamic: bool = True,
):
    if isinstance(coords, np.ndarray):
        coords_t = torch.from_numpy(coords)
    elif isinstance(coords, torch.Tensor):
        coords_t = coords
    else:
        coords_t = torch.as_tensor(coords)

    coords_t = coords_t.to(device=device, dtype=torch.float32)
    demand_t = torch.as_tensor(demand, device=device, dtype=torch.float32)

    nn = aco.nn_torch.to(device=device, dtype=torch.long)
    n, k = nn.shape
    E = n * k

    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(k)
    dst = nn.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    dist = torch.norm(coords_t[src] - coords_t[dst], dim=1).view(n, k)
    dist_mean = dist.mean(dim=1, keepdim=True).clamp_min(1e-12)
    dist_norm = (dist / dist_mean).view(E, 1)

    if dynamic:
        tau = aco.pheromone_sparse.detach().to(device=device, dtype=torch.float32)
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(1e-12)
        tau_rel = (tau / tau_mean).clamp_min(1e-12)
        log_tau_rel = torch.log(tau_rel).clamp(-5.0, 5.0).view(E, 1)
        tau_std = tau.std(dim=1, keepdim=True)
        tau_cv = (tau_std / tau_mean).clamp(0, 10).repeat_interleave(k, dim=0)
    else:
        log_tau_rel = torch.zeros((E, 1), device=device, dtype=torch.float32)
        tau_cv = torch.zeros((E, 1), device=device, dtype=torch.float32)

    try:
        src_route_np = aco.source_route
    except AttributeError:
        src_route_np = aco.source_perm

    src_route = torch.as_tensor(np.asarray(src_route_np, dtype=np.int64), device=device, dtype=torch.long)

    is_source_succ = torch.zeros((E, 1), device=device, dtype=torch.float32)
    is_source_pred = torch.zeros((E, 1), device=device, dtype=torch.float32)

    if src_route.numel() > 1:
        route_u = src_route[:-1]
        route_v = src_route[1:]

        succ_arr = torch.full((n,), -1, device=device, dtype=torch.long)
        pred_arr = torch.full((n,), -1, device=device, dtype=torch.long)

        mask_cust_u = route_u != 0
        succ_arr[route_u[mask_cust_u]] = route_v[mask_cust_u]

        mask_cust_v = route_v != 0
        pred_arr[route_v[mask_cust_v]] = route_u[mask_cust_v]

        depot_succs = route_v[route_u == 0]
        depot_preds = route_u[route_v == 0]

        mask_src_cust = src != 0
        is_source_succ[mask_src_cust] = (dst[mask_src_cust] == succ_arr[src[mask_src_cust]]).float().view(-1, 1)

        mask_src_depot = src == 0
        if mask_src_depot.any():
            is_source_succ[mask_src_depot] = torch.isin(dst[mask_src_depot], depot_succs).float().view(-1, 1)

        is_source_pred[mask_src_cust] = (dst[mask_src_cust] == pred_arr[src[mask_src_cust]]).float().view(-1, 1)
        if mask_src_depot.any():
            is_source_pred[mask_src_depot] = torch.isin(dst[mask_src_depot], depot_preds).float().view(-1, 1)

    is_in_route = (is_source_succ > 0.5) | (is_source_pred > 0.5)
    is_new_edge = (~is_in_route).to(torch.float32).view(E, 1)

    features = [dist_norm]
    if not ablation_pheromone:
        features.extend([tau_cv, log_tau_rel])
    if not ablation_incumbent:
        features.extend([is_source_succ, is_source_pred, is_new_edge])
    edge_attr = torch.cat(features, dim=1)

    depot_flag = torch.zeros((n, 1), device=device, dtype=torch.float32)
    depot_flag[0, 0] = 1.0
    x = torch.cat([coords_t, demand_t.unsqueeze(1), depot_flag], dim=1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _load_eval_items(problem: str, n_node: int, val_size: int, rl_data: bool) -> List[Any]:
    val_list = utils.load_auto_dataset(n_node, problem=problem, rl_data=rl_data, device="cpu")
    if val_list is None:
        val_list = []
        for _ in range(val_size):
            if problem == "tsp":
                val_list.append(torch.from_numpy(utils.generate_tsp_instance(n_node)))
            else:
                coords, demand, capacity = utils.gen_cvrp_instance(n_node, device="cpu")
                val_list.append((coords.cpu(), demand.cpu(), capacity))
    return list(val_list[:val_size])


def _normalize_eval_item(problem: str, item: Any) -> Tuple[Any, Optional[float], str]:
    name = ""
    opt_cost = None

    if problem == "tsp":
        if isinstance(item, tuple):
            coords = item[0]
            if len(item) > 1 and isinstance(item[1], (float, int, np.number)):
                opt_cost = float(item[1])
            if len(item) > 3:
                name = str(item[3])
            item = coords
        if torch.is_tensor(item):
            return item.cpu(), opt_cost, name
        return item, opt_cost, name

    if isinstance(item, tuple) and len(item) >= 5:
        coords, demand, capacity = item[:3]
        if len(item) > 3 and isinstance(item[3], (float, int, np.number)):
            opt_cost = float(item[3])
        if len(item) > 5:
            name = str(item[5])
        item = (coords, demand, capacity)

    if isinstance(item, (list, tuple)):
        coords, demand, capacity = item[:3]
        if torch.is_tensor(coords):
            coords = coords.cpu().numpy()
        if torch.is_tensor(demand):
            demand = demand.cpu().numpy()
        capacity = float(capacity.item()) if torch.is_tensor(capacity) else float(capacity)
        if capacity > 1.0 + 1e-6:
            demand = demand / capacity
            capacity = 1.0
        return (coords, demand, capacity), opt_cost, name

    return item, opt_cost, name


def _device_is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _cuda_peak_begin(device: str) -> Optional[Dict[str, float]]:
    if not _device_is_cuda(device):
        return None
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "start_alloc_gb": float(torch.cuda.memory_allocated(device) / 1024**3),
        "start_reserved_gb": float(torch.cuda.memory_reserved(device) / 1024**3),
    }


def _cuda_peak_end(device: str, start: Optional[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if start is None:
        return {
            "start_alloc_gb": None,
            "start_reserved_gb": None,
            "peak_alloc_gb": None,
            "peak_reserved_gb": None,
            "delta_peak_alloc_gb": None,
            "delta_peak_reserved_gb": None,
        }
    torch.cuda.synchronize(device)
    peak_alloc = float(torch.cuda.max_memory_allocated(device) / 1024**3)
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1024**3)
    return {
        **start,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "delta_peak_alloc_gb": peak_alloc - start["start_alloc_gb"],
        "delta_peak_reserved_gb": peak_reserved - start["start_reserved_gb"],
    }


def _build_infer_args(problem: str, n_node: int, device: str, config: Dict[str, Any], no_anneal: bool) -> argparse.Namespace:
    return argparse.Namespace(
        problem=problem,
        n_node=int(n_node),
        device=device,
        H=int(config["H"]),
        mini_H=int(config["mini_H"]),
        gamma=float(config["gamma"]),
        min_gamma=float(config["min_gamma"]),
        no_anneal=bool(no_anneal),
        disable_heuristic=bool(config["disable_heuristic"]),
        no_local_search=bool(config["no_local_search"]),
        rho=float(config["rho"]),
        no_smooth_mmas=bool(config["no_smooth_mmas"]),
        min_new_edges=int(config["min_new_edges"]),
        no_extend_ls=bool(config["no_extend_ls"]),
        no_normalized_heuristic=bool(config["no_normalized_heuristic"]),
        L=int(config["L"]),
        ls_scope=str(config["ls_scope"]),
        ls_budget=str(config["ls_budget"]),
        ls_max_opt=int(config["ls_max_opt"]),
        timed=True,
        verify=(problem == "cvrp"),
        runtime_limit=None,
        iter_log=False,
        iter_print=False,
        stage_metrics=False,
        alpha=float(config["alpha"]),
        beta=float(config["beta"]),
        no_dynamic_feats=bool(config["no_dynamic_feats"]),
        ablation_pheromone_features=False,
        ablation_incumbent_features=False,
    )


def _safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _safe_max(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(max(valid))


def run_worker(args: argparse.Namespace) -> int:
    if _device_is_cuda(args.device) and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({args.device}) but torch.cuda.is_available() is False")

    variant = VARIANT_SPECS[args.variant]
    config = dict(DEFAULT_CONFIG)
    _maybe_override(config, "H", args.H)
    _maybe_override(config, "mini_H", args.mini_H)
    _maybe_override(config, "n_ants", args.n_ants)
    _maybe_override(config, "k_sparse", args.k_sparse)
    _maybe_override(config, "rho", args.rho)
    _maybe_override(config, "min_new_edges", args.min_new_edges)
    _maybe_override(config, "warmup_ratio", args.warmup_ratio)
    _maybe_override(config, "seed", args.seed)
    if args.alg is not None:
        config["alg"] = args.alg
    config["no_dynamic_feats"] = bool(variant.no_dynamic_feats)

    resolved_method_name = _resolve_method_name(args.problem, int(config["H"]), args.method)
    method = METHOD_SPECS[resolved_method_name]

    checkpoint_path = None
    if method.use_model:
        checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else _resolve_checkpoint(
            args.problem,
            args.n_node,
            Path(args.checkpoint_root),
            args.checkpoint_scale,
        )
        config["checkpoint"] = str(checkpoint_path)
    else:
        config["checkpoint"] = None

    if args.threads is None:
        try:
            import psutil
            args.threads = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
        except Exception:
            args.threads = os.cpu_count() or 1
    faco.set_faco_cpp_threads(int(args.threads))

    utils.set_seed(int(config["seed"]))
    model, build_fn, aco_class = _build_runtime(args.problem, config, args.device, method, variant)
    eval_items = _load_eval_items(args.problem, args.n_node, args.val_size, args.rl_data)

    per_instance: List[Dict[str, Any]] = []
    inject_step = None
    if method.use_mix:
        inject_step = int(int(config["H"]) * float(config["warmup_ratio"]))

    infer_args = _build_infer_args(args.problem, args.n_node, args.device, config, no_anneal=method.no_anneal)

    for idx, raw_item in enumerate(eval_items):
        item, opt_cost, name = _normalize_eval_item(args.problem, raw_item)
        start_stats = _cuda_peak_begin(args.device)
        t0 = time.time()
        _, best_seen, timings, extra = infer_instance(
            args.problem,
            aco_class,
            build_fn,
            model,
            item,
            int(config["k_sparse"]),
            int(config["n_ants"]),
            not bool(config["no_dynamic_feats"]),
            infer_args,
            use_heuristic_only=method.use_heuristic_only,
            collect_metrics=False,
            metrics_every_step=False,
            inject_step=inject_step,
            seed=int(config["seed"]) + idx,
            ablation_pheromone=variant.ablation_pheromone,
            ablation_incumbent=variant.ablation_incumbent,
        )
        elapsed_s = float(time.time() - t0)
        peak_stats = _cuda_peak_end(args.device, start_stats)
        gap_pct = None
        if opt_cost is not None and opt_cost > 1e-9:
            gap_pct = float((float(best_seen) - float(opt_cost)) / float(opt_cost) * 100.0)
        per_instance.append({
            "idx": idx,
            "name": name or f"instance_{idx}",
            "best_cost": float(best_seen),
            "opt_cost": opt_cost,
            "gap_pct": gap_pct,
            "elapsed_s": elapsed_s,
            "timed_out": bool(extra.get("timed_out", False)),
            "time_neural_s": float(timings.get("time_neural", 0.0)) if timings else 0.0,
            **peak_stats,
        })
        gc.collect()
        if _device_is_cuda(args.device):
            torch.cuda.empty_cache()

    summary = {
        "problem": args.problem,
        "n_node": int(args.n_node),
        "variant": args.variant,
        "edge_feature_count": int(variant.edge_feats),
        "requested_method": args.method,
        "method": resolved_method_name,
        "device": args.device,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_scale": int(args.checkpoint_scale) if args.checkpoint_scale is not None else None,
        "val_size": int(args.val_size),
        "rl_data": bool(args.rl_data),
        "config": {
            "alg": str(config["alg"]),
            "H": int(config["H"]),
            "mini_H": int(config["mini_H"]),
            "n_ants": int(config["n_ants"]),
            "k_sparse": int(config["k_sparse"]),
            "rho": float(config["rho"]),
            "min_new_edges": int(config["min_new_edges"]),
            "warmup_ratio": float(config["warmup_ratio"]),
            "no_dynamic_feats": bool(config["no_dynamic_feats"]),
            "edge_feature_count": int(variant.edge_feats),
            "seed": int(config["seed"]),
        },
        "summary": {
            "mean_cost": _safe_mean([row["best_cost"] for row in per_instance]),
            "mean_gap_pct": _safe_mean([row["gap_pct"] for row in per_instance]),
            "mean_elapsed_s": _safe_mean([row["elapsed_s"] for row in per_instance]),
            "max_peak_alloc_gb": _safe_max([row["peak_alloc_gb"] for row in per_instance]),
            "max_peak_reserved_gb": _safe_max([row["peak_reserved_gb"] for row in per_instance]),
            "mean_peak_alloc_gb": _safe_mean([row["peak_alloc_gb"] for row in per_instance]),
            "mean_peak_reserved_gb": _safe_mean([row["peak_reserved_gb"] for row in per_instance]),
            "max_delta_peak_alloc_gb": _safe_max([row["delta_peak_alloc_gb"] for row in per_instance]),
            "max_delta_peak_reserved_gb": _safe_max([row["delta_peak_reserved_gb"] for row in per_instance]),
        },
        "per_instance": per_instance,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(
        f"[worker] {args.problem} n={args.n_node} {args.variant} {resolved_method_name}: "
        f"peak_reserved={summary['summary']['max_peak_reserved_gb']}"
    )
    return 0


def _run_subprocess(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def run_driver(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent
    script_path = Path(__file__).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    bundle: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "problems": args.problems,
        "scales": args.scales,
        "variants": args.variants,
        "methods": args.methods,
        "rows": rows,
    }

    total_runs = len(args.problems) * len(args.scales) * len(args.variants) * len(args.methods)
    run_idx = 0

    for problem in args.problems:
        for scale in args.scales:
            for variant in args.variants:
                for method in args.methods:
                    run_idx += 1
                    temp_json = tmp_dir / f"{problem}_n{scale}_{variant}_{method}.json"
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--worker",
                        "--problem",
                        problem,
                        "--n_node",
                        str(scale),
                        "--variant",
                        variant,
                        "--method",
                        method,
                        "--device",
                        args.device,
                        "--val_size",
                        str(args.val_size),
                        "--output_json",
                        str(temp_json),
                        "--checkpoint_root",
                        str(Path(args.checkpoint_root).resolve()),
                    ]
                    if args.rl_data:
                        cmd.append("--rl_data")
                    if args.checkpoint is not None:
                        cmd.extend(["--checkpoint", str(Path(args.checkpoint).resolve())])
                    if args.checkpoint_scale is not None:
                        cmd.extend(["--checkpoint_scale", str(args.checkpoint_scale)])
                    if args.threads is not None:
                        cmd.extend(["--threads", str(args.threads)])
                    for key in ["H", "mini_H", "n_ants", "k_sparse", "rho", "min_new_edges", "warmup_ratio", "seed", "alg"]:
                        value = getattr(args, key)
                        if value is not None:
                            cmd.extend([f"--{key}", str(value)])

                    print(f"[{run_idx}/{total_runs}] profiling {problem} n={scale} {variant} {method}")
                    proc = _run_subprocess(cmd, repo_root)

                    row: Dict[str, Any] = {
        "problem": problem,
        "n_node": int(scale),
        "variant": variant,
        "edge_feature_count": None,
        "requested_method": method,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
                        "stdout_tail": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
                        "stderr_tail": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
                    }

                    if proc.returncode == 0 and temp_json.exists():
                        payload = json.loads(temp_json.read_text(encoding="utf-8"))
                        summary = payload.get("summary", {})
                        row.update({
                            "method": payload.get("method"),
                            "edge_feature_count": payload.get("edge_feature_count"),
                            "checkpoint": payload.get("checkpoint"),
                            "checkpoint_scale": payload.get("checkpoint_scale"),
                            "val_size": payload.get("val_size"),
                            "mean_cost": summary.get("mean_cost"),
                            "mean_gap_pct": summary.get("mean_gap_pct"),
                            "mean_elapsed_s": summary.get("mean_elapsed_s"),
                            "max_peak_alloc_gb": summary.get("max_peak_alloc_gb"),
                            "max_peak_reserved_gb": summary.get("max_peak_reserved_gb"),
                            "mean_peak_alloc_gb": summary.get("mean_peak_alloc_gb"),
                            "mean_peak_reserved_gb": summary.get("mean_peak_reserved_gb"),
                            "max_delta_peak_alloc_gb": summary.get("max_delta_peak_alloc_gb"),
                            "max_delta_peak_reserved_gb": summary.get("max_delta_peak_reserved_gb"),
                        })
                    rows.append(row)

    csv_path = out_dir / "vram_profile.csv"
    json_path = out_dir / "vram_profile.json"

    fieldnames = [
        "problem",
        "n_node",
        "variant",
        "edge_feature_count",
        "requested_method",
        "method",
        "status",
        "returncode",
        "checkpoint",
        "checkpoint_scale",
        "val_size",
        "mean_cost",
        "mean_gap_pct",
        "mean_elapsed_s",
        "max_peak_alloc_gb",
        "max_peak_reserved_gb",
        "mean_peak_alloc_gb",
        "mean_peak_reserved_gb",
        "max_delta_peak_alloc_gb",
        "max_delta_peak_reserved_gb",
        "stdout_tail",
        "stderr_tail",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(json.dumps(bundle, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile peak CUDA VRAM for DyNACO static (1 edge feature) vs dynamic (6 edge features) across problems and scales."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--problem", choices=DEFAULT_PROBLEMS, default=None)
    parser.add_argument("--n_node", type=int, default=None)
    parser.add_argument("--variant", choices=sorted(VARIANT_SPECS.keys()), default=None)
    parser.add_argument("--method", choices=["auto"] + sorted(METHOD_SPECS.keys()), default=None)
    parser.add_argument("--problems", type=str, default=",".join(DEFAULT_PROBLEMS))
    parser.add_argument("--scales", type=str, default=",".join(str(x) for x in DEFAULT_SCALES))
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))

    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to reuse for model-based methods.")
    parser.add_argument("--checkpoint_root", type=str, default="pretrained", help="Root used for auto checkpoint lookup.")
    parser.add_argument("--checkpoint_scale", type=int, default=None, help="If set, reuse the checkpoint from this scale for all test scales.")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_size", type=int, default=1, help="Instances per run; keep this small for clean VRAM profiling.")
    parser.add_argument("--rl_data", action="store_true", help="Use TSPLIB/CVRPLIB auto datasets instead of standard synthetic test sets.")
    parser.add_argument("--output_dir", type=str, default="output/vram_profile")
    parser.add_argument("--output_json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=None)

    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--mini_H", type=int, default=None)
    parser.add_argument("--n_ants", type=int, default=None)
    parser.add_argument("--k_sparse", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--min_new_edges", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alg", choices=["faco", "mmas"], default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.worker:
        if args.problem is None or args.n_node is None or args.variant is None or args.method is None or args.output_json is None:
            parser.error("--worker requires --problem, --n_node, --variant, --method, and --output_json")
        return run_worker(args)

    args.problems = _parse_csv_list(args.problems, str)
    args.scales = _parse_csv_list(args.scales, int)
    args.variants = _parse_csv_list(args.variants, str)
    args.methods = _parse_csv_list(args.methods, str)

    unknown_variants = [variant for variant in args.variants if variant not in VARIANT_SPECS]
    if unknown_variants:
        parser.error(f"Unknown variants: {unknown_variants}")
    unknown_methods = [method for method in args.methods if method not in METHOD_SPECS]
    unknown_methods = [method for method in args.methods if method != "auto" and method not in METHOD_SPECS]
    if unknown_methods:
        parser.error(f"Unknown methods: {unknown_methods}")

    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
