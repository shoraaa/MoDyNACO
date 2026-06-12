#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_CVRP_1K_CHECKPOINTS = [
    Path("models/cvrp/n1000/cvrp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_nosmooth_best.pt"),
    Path("models/cvrp/n1000/cvrp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt"),
    Path("pretrained/cvrp/n1000/best.pt"),
]

DEMAND_LOW = 1
DEMAND_HIGH = 9
DEFAULT_INSCALE_CHECKPOINT_ROOT = Path("models") / "cvrp"


@dataclass
class RawCVRPInstance:
    name: str
    coords: torch.Tensor
    customer_demands: torch.Tensor


@dataclass
class CapacityInstanceResult:
    capacity: float
    checkpoint: str
    instance: str
    cost: float
    elapsed_s: float
    route_count: int
    customers_per_subroute: float
    subroute_cost: float
    subroute_load: float
    subroute_utilization: float
    mean_guidance: float
    pheromone_correlation: float


@dataclass
class CapacitySummary:
    capacity: float
    checkpoint: str
    mean_cost: float
    std_cost: float
    mean_time_s: float
    std_time_s: float
    mean_route_count: float
    mean_customers_per_subroute: float
    mean_subroute_cost: float
    mean_subroute_load: float
    mean_subroute_utilization: float
    mean_guidance: float
    mean_pheromone_correlation: float


def parse_capacities(value: str) -> list[float]:
    capacities: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        cap = float(part)
        if cap <= 0:
            raise argparse.ArgumentTypeError("Capacities must be positive")
        capacities.append(cap)
    if not capacities:
        raise argparse.ArgumentTypeError("At least one capacity is required")
    return capacities


def resolve_default_checkpoint() -> Path:
    for candidate in DEFAULT_CVRP_1K_CHECKPOINTS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a default CVRP-1K checkpoint. Pass --checkpoint explicitly."
    )


def capacity_tag(capacity: float) -> str:
    return f"{capacity:g}"


def resolve_inscale_checkpoint(
    n_node: int,
    capacity: float,
    no_smooth_mmas: bool = True,
    checkpoint_root: Path | None = None,
) -> Path:
    root = checkpoint_root or DEFAULT_INSCALE_CHECKPOINT_ROOT
    checkpoint_dir = root / f"n{n_node}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"In-scale checkpoint directory not found: {checkpoint_dir}")

    cap = capacity_tag(capacity)
    if no_smooth_mmas:
        preferred_matches = sorted(checkpoint_dir.glob(f"*cap{cap}*nosmooth*_best.pt"))
    else:
        preferred_matches = sorted(
            path for path in checkpoint_dir.glob(f"*cap{cap}*_best.pt")
            if "nosmooth" not in path.name
        )
    if preferred_matches:
        return max(preferred_matches, key=lambda path: path.stat().st_mtime)

    all_matches = sorted(checkpoint_dir.glob(f"*cap{cap}*.pt"))
    if all_matches:
        return max(all_matches, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(
        f"No in-scale checkpoint found for n={n_node}, capacity={cap} in {checkpoint_dir}"
    )


def split_route0(route0: list[int] | np.ndarray | torch.Tensor) -> list[list[int]]:
    if torch.is_tensor(route0):
        route = [int(x) for x in route0.detach().cpu().tolist()]
    else:
        route = [int(x) for x in route0]

    subroutes: list[list[int]] = []
    current: list[int] = []
    for node in route:
        if node == 0:
            if current:
                subroutes.append(current)
                current = []
            continue
        current.append(node)
    if current:
        subroutes.append(current)
    return subroutes


def compute_subroute_stats(
    coords: torch.Tensor | np.ndarray,
    customer_demands: torch.Tensor | np.ndarray,
    capacity: float,
    route0: list[int] | np.ndarray | torch.Tensor,
) -> dict[str, float]:
    coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords, dtype=np.float32)
    demands_np = (
        customer_demands.detach().cpu().numpy()
        if torch.is_tensor(customer_demands)
        else np.asarray(customer_demands, dtype=np.float32)
    )
    subroutes = split_route0(route0)

    if not subroutes:
        return {
            "route_count": 0.0,
            "customers_per_subroute": 0.0,
            "subroute_cost": 0.0,
            "subroute_load": 0.0,
            "subroute_utilization": 0.0,
        }

    subroute_costs: list[float] = []
    subroute_loads: list[float] = []
    subroute_sizes: list[int] = []

    for subroute in subroutes:
        expanded = [0, *subroute, 0]
        cost = 0.0
        for idx in range(len(expanded) - 1):
            u = expanded[idx]
            v = expanded[idx + 1]
            diff = coords_np[u] - coords_np[v]
            cost += float(np.sqrt(np.sum(diff ** 2)))
        load = float(demands_np[np.asarray(subroute, dtype=np.int64) - 1].sum())

        subroute_costs.append(cost)
        subroute_loads.append(load)
        subroute_sizes.append(len(subroute))

    mean_load = float(np.mean(subroute_loads))
    return {
        "route_count": float(len(subroutes)),
        "customers_per_subroute": float(np.mean(subroute_sizes)),
        "subroute_cost": float(np.mean(subroute_costs)),
        "subroute_load": mean_load,
        "subroute_utilization": mean_load / float(capacity),
    }


def generate_raw_instances(n_node: int, n_instances: int, seed: int) -> list[RawCVRPInstance]:
    rng = np.random.default_rng(seed)
    instances: list[RawCVRPInstance] = []
    for idx in range(n_instances):
        coords = torch.tensor(rng.random((n_node + 1, 2), dtype=np.float32), dtype=torch.float32)
        customer_demands = torch.tensor(
            rng.integers(DEMAND_LOW, DEMAND_HIGH + 1, size=(n_node,), endpoint=False),
            dtype=torch.float32,
        )
        instances.append(
            RawCVRPInstance(
                name=f"synthetic_{idx:03d}",
                coords=coords,
                customer_demands=customer_demands,
            )
        )
    return instances


def demand_with_depot(customer_demands: torch.Tensor, capacity: float) -> np.ndarray:
    normalized = (customer_demands / float(capacity)).detach().cpu().numpy().astype(np.float32, copy=False)
    return np.concatenate([np.zeros(1, dtype=np.float32), normalized], axis=0)


def _clone_namespace(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def restore_args_from_checkpoint_config(
    args: argparse.Namespace,
    config: dict[str, Any],
    argv: list[str],
) -> None:
    ignore_args = {
        "checkpoint",
        "device",
        "dataset",
        "visualize",
        "visualize_output",
        "timed",
        "verify",
        "baseline",
        "baseline_runs",
        "baseline_time_limit",
        "threads",
        "seed",
        "save_dir",
        "wandb_project",
        "wandb_entity",
        "no_wandb",
        "warmup",
        "no_baseline",
        "problem",
        "n_node",
        "no_anneal",
    }

    for key, value in config.items():
        if key in ignore_args or not hasattr(args, key):
            continue
        flag_underscore = f"--{key}"
        flag_hyphen = f"--{key.replace('_', '-')}"
        if flag_underscore in argv or flag_hyphen in argv:
            continue
        setattr(args, key, value)


def load_model_and_args(args: argparse.Namespace, argv: list[str]) -> tuple[dict[str, Any], torch.nn.Module]:
    import net

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint.get("config", {})
    restore_args_from_checkpoint_config(args, config, argv)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    feats = 1
    if "emb_net.v_lin0.weight" in state_dict:
        feats = state_dict["emb_net.v_lin0.weight"].shape[1]

    edge_feats = 6
    if "emb_net.e_lin0.weight" in state_dict:
        edge_feats = state_dict["emb_net.e_lin0.weight"].shape[1]

    model = net.Net(feats=feats, edge_feats=edge_feats, logit_net=not args.no_logit_net).to(args.device)
    model.load_state_dict(state_dict)
    model.eval()
    return checkpoint, model


def build_eval_args(parsed: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        problem="cvrp",
        n_node=parsed.n_node,
        k_sparse=parsed.k_sparse,
        alg=parsed.alg,
        checkpoint=str(parsed.checkpoint),
        n_ants=parsed.n_ants,
        H=parsed.H,
        mini_H=parsed.mini_H,
        disable_heuristic=parsed.disable_heuristic,
        no_local_search=parsed.no_local_search,
        no_smooth_mmas=parsed.no_smooth_mmas,
        no_extend_ls=parsed.no_extend_ls,
        ls_scope=parsed.ls_scope,
        ls_budget=parsed.ls_budget,
        ls_max_opt=parsed.ls_max_opt,
        rho=parsed.rho,
        min_new_edges=parsed.min_new_edges,
        no_normalized_heuristic=parsed.no_normalized_heuristic,
        no_logit_net=parsed.no_logit_net,
        no_dynamic_feats=parsed.no_dynamic_feats,
        no_anneal=parsed.no_anneal,
        gamma=parsed.gamma,
        min_gamma=parsed.min_gamma,
        device=parsed.device,
        seed=parsed.seed,
        timed=False,
        runtime_limit=None,
        verify=True,
        verify_final_only=True,
        L=0,
        iter_log=False,
        iter_print=False,
        stage_metrics=False,
        warmup=False,
        ablation_pheromone_features=False,
        ablation_incumbent_features=False,
    )


def default_output_paths(checkpoint: Path) -> tuple[Path, Path]:
    out_dir = Path("output") / "cvrp_capacity_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem
    return out_dir / f"{stem}.json", out_dir / f"{stem}.csv"


def default_markdown_output_path() -> Path:
    return Path("test_cvrp.md")


def summarize_capacity(rows: list[CapacityInstanceResult]) -> CapacitySummary:
    return CapacitySummary(
        capacity=rows[0].capacity,
        checkpoint=rows[0].checkpoint,
        mean_cost=float(np.mean([row.cost for row in rows])),
        std_cost=float(np.std([row.cost for row in rows], ddof=0)),
        mean_time_s=float(np.mean([row.elapsed_s for row in rows])),
        std_time_s=float(np.std([row.elapsed_s for row in rows], ddof=0)),
        mean_route_count=float(np.mean([row.route_count for row in rows])),
        mean_customers_per_subroute=float(np.mean([row.customers_per_subroute for row in rows])),
        mean_subroute_cost=float(np.mean([row.subroute_cost for row in rows])),
        mean_subroute_load=float(np.mean([row.subroute_load for row in rows])),
        mean_subroute_utilization=float(np.mean([row.subroute_utilization for row in rows])),
        mean_guidance=float(np.mean([row.mean_guidance for row in rows])),
        mean_pheromone_correlation=float(np.mean([row.pheromone_correlation for row in rows])),
    )


def safe_metric_mean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def safe_metric_mean_excluding_outer_h1(values: list[float]) -> float:
    return safe_metric_mean(values[1:])


def summarize_outer_h_metrics(
    metric_profiles: list[dict[str, list[float]]],
    mini_h: int,
) -> list[dict[str, float]]:
    if not metric_profiles:
        return []

    n_steps = max(
        max(len(profile.get("prior_mean", [])), len(profile.get("corr", [])))
        for profile in metric_profiles
    )
    rows: list[dict[str, float]] = []
    for step_idx in range(n_steps):
        guidance_vals = [
            profile["prior_mean"][step_idx]
            for profile in metric_profiles
            if step_idx < len(profile.get("prior_mean", []))
        ]
        corr_vals = [
            profile["corr"][step_idx]
            for profile in metric_profiles
            if step_idx < len(profile.get("corr", []))
        ]
        rows.append(
            {
                "outer_step": float(step_idx + 1),
                "iteration": float((step_idx + 1) * int(mini_h)),
                "mean_guidance": safe_metric_mean(guidance_vals),
                "mean_pheromone_correlation": safe_metric_mean(corr_vals),
            }
        )
    return rows


def print_outer_h_table(
    capacity: float,
    checkpoint: str,
    metric_profiles: list[dict[str, list[float]]],
    mini_h: int,
) -> list[dict[str, float]]:
    rows = summarize_outer_h_metrics(metric_profiles, mini_h)
    if not rows:
        return rows

    headers = ["Capacity", "Checkpoint", "OuterH", "Iteration", "GuidanceMean", "PheroCorr"]
    table_rows = [
        [
            f"{capacity:.1f}",
            Path(checkpoint).stem if checkpoint else "-",
            f"{int(row['outer_step'])}",
            f"{int(row['iteration'])}",
            f"{row['mean_guidance']:.4f}",
            f"{row['mean_pheromone_correlation']:.4f}",
        ]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def line(char: str) -> str:
        return "+" + "+".join(char * (width + 2) for width in widths) + "+"

    print()
    print(f"Outer-H metrics for capacity={capacity:.1f}")
    print(line("-"))
    print("| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    print(line("="))
    for row in table_rows:
        print("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    print(line("-"))
    return rows


def print_summary_table(summaries: list[CapacitySummary]) -> None:
    headers = [
        "Capacity",
        "Checkpoint",
        "MeanCost",
        "MeanTime(s)",
        "Routes",
        "Cust/Subroute",
        "SubrouteCost",
        "Load/Subroute",
        "Utilization",
        "GuidanceMean",
        "PheroCorr",
    ]
    rows = [
        [
            f"{summary.capacity:.1f}",
            Path(summary.checkpoint).stem if summary.checkpoint else "-",
            f"{summary.mean_cost:.4f}",
            f"{summary.mean_time_s:.4f}",
            f"{summary.mean_route_count:.2f}",
            f"{summary.mean_customers_per_subroute:.2f}",
            f"{summary.mean_subroute_cost:.4f}",
            f"{summary.mean_subroute_load:.2f}",
            f"{summary.mean_subroute_utilization:.3f}",
            f"{summary.mean_guidance:.4f}",
            f"{summary.mean_pheromone_correlation:.4f}",
        ]
        for summary in summaries
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def line(char: str) -> str:
        return "+" + "+".join(char * (width + 2) for width in widths) + "+"

    print(line("-"))
    print("| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    print(line("="))
    for row in rows:
        print("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    print(line("-"))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown_report(payload: dict[str, Any]) -> str:
    summary_rows = payload.get("summary", [])
    summary_headers = [
        "Capacity",
        "Checkpoint",
        "MeanCost",
        "MeanTime(s)",
        "Routes",
        "Cust/Subroute",
        "SubrouteCost",
        "Load/Subroute",
        "Utilization",
        "GuidanceMean",
        "PheroCorr",
    ]
    summary_table_rows = [
        [
            f"{float(row['capacity']):.1f}",
            Path(row["checkpoint"]).stem if row.get("checkpoint") else "-",
            f"{float(row['mean_cost']):.4f}",
            f"{float(row['mean_time_s']):.4f}",
            f"{float(row['mean_route_count']):.2f}",
            f"{float(row['mean_customers_per_subroute']):.2f}",
            f"{float(row['mean_subroute_cost']):.4f}",
            f"{float(row['mean_subroute_load']):.2f}",
            f"{float(row['mean_subroute_utilization']):.3f}",
            f"{float(row['mean_guidance']):.4f}",
            f"{float(row['mean_pheromone_correlation']):.4f}",
        ]
        for row in summary_rows
    ]

    lines = [
        "# CVRP Capacity Sweep",
        "",
        "## Summary",
        "",
        markdown_table(summary_headers, summary_table_rows),
    ]

    outer_h_metrics = payload.get("outer_h_metrics", {})
    capacity_checkpoints = payload.get("capacity_checkpoints", {})
    for capacity in payload.get("capacities", []):
        cap_key = capacity_tag(float(capacity))
        checkpoint = capacity_checkpoints.get(cap_key, "")
        rows = outer_h_metrics.get(cap_key, [])
        if not rows:
            continue
        lines.extend(
            [
                "",
                f"## Outer-H Metrics: Capacity {float(capacity):.1f}",
                "",
                f"Checkpoint: `{Path(checkpoint).stem if checkpoint else '-'}`",
                "",
                markdown_table(
                    ["OuterH", "Iteration", "GuidanceMean", "PheroCorr"],
                    [
                        [
                            f"{int(row['outer_step'])}",
                            f"{int(row['iteration'])}",
                            f"{float(row['mean_guidance']):.4f}",
                            f"{float(row['mean_pheromone_correlation']):.4f}",
                        ]
                        for row in rows
                    ],
                ),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def run_capacity_sweep(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    import faco
    import utils

    utils.set_seed(args.seed)
    raw_instances = generate_raw_instances(args.n_node, args.n_instances, args.seed)
    rows: list[CapacityInstanceResult] = []
    capacity_checkpoints: dict[float, str] = {}
    outer_h_metrics_by_capacity: dict[str, list[dict[str, float]]] = {}

    capacity_bar = tqdm(args.capacities, desc="Capacities", unit="cap")
    for capacity in capacity_bar:
        run_args = argparse.Namespace(**vars(args))
        if args.in_scale:
            run_args.checkpoint = resolve_inscale_checkpoint(
                args.n_node,
                capacity,
                no_smooth_mmas=args.no_smooth_mmas,
                checkpoint_root=args.checkpoint_root,
            )
        _, model = load_model_and_args(run_args, argv)
        eval_args = build_eval_args(run_args)
        aco_class = faco.ACO_CVRP if run_args.alg == "mmas" else faco.MFACO_CVRP
        checkpoint_str = str(run_args.checkpoint)
        capacity_checkpoints[float(capacity)] = checkpoint_str
        capacity_bar.set_postfix_str(f"capacity={capacity:.1f} ckpt={Path(checkpoint_str).stem}")
        capacity_metric_profiles: list[dict[str, list[float]]] = []
        instance_bar = tqdm(
            raw_instances,
            desc=f"Instances@{capacity:.1f}",
            unit="inst",
            leave=False,
        )
        for instance in instance_bar:
            demand = demand_with_depot(instance.customer_demands, capacity)
            infer_args = _clone_namespace(eval_args, seed=args.seed)
            started = time.perf_counter()
            _, best_cost, _, extra = utils.infer_instance(
                "cvrp",
                aco_class,
                utils.build_pyg_data_cvrp,
                model,
                (instance.coords, demand, 1.0),
                args.k_sparse,
                args.n_ants,
                not args.no_dynamic_feats,
                infer_args,
                use_heuristic_only=False,
                collect_metrics=True,
                metrics_every_step=True,
                seed=args.seed,
            )
            elapsed = time.perf_counter() - started

            route0 = extra.get("best_decoded_route")
            if route0 is None:
                raise RuntimeError("infer_instance did not return best_decoded_route; verify mode must stay enabled")

            metrics = extra.get("metrics", {})
            prior_mean = metrics.get("prior_mean", [])
            corr = metrics.get("corr", [])
            capacity_metric_profiles.append(
                {
                    "prior_mean": list(prior_mean),
                    "corr": list(corr),
                }
            )
            route_stats = compute_subroute_stats(instance.coords, instance.customer_demands, capacity, route0)
            rows.append(
                CapacityInstanceResult(
                    capacity=float(capacity),
                    checkpoint=checkpoint_str,
                    instance=instance.name,
                    cost=float(best_cost),
                    elapsed_s=float(elapsed),
                    route_count=int(route_stats["route_count"]),
                    customers_per_subroute=float(route_stats["customers_per_subroute"]),
                    subroute_cost=float(route_stats["subroute_cost"]),
                    subroute_load=float(route_stats["subroute_load"]),
                    subroute_utilization=float(route_stats["subroute_utilization"]),
                    mean_guidance=safe_metric_mean_excluding_outer_h1(prior_mean),
                    pheromone_correlation=safe_metric_mean_excluding_outer_h1(corr),
                )
            )
            instance_bar.set_postfix(
                cost=f"{float(best_cost):.4f}",
                routes=int(route_stats["route_count"]),
            )
        instance_bar.close()
        outer_h_metrics_by_capacity[capacity_tag(float(capacity))] = print_outer_h_table(
            float(capacity),
            checkpoint_str,
            capacity_metric_profiles,
            args.mini_H,
        )

    grouped: dict[float, list[CapacityInstanceResult]] = {}
    for row in rows:
        grouped.setdefault(row.capacity, []).append(row)

    summaries = [summarize_capacity(grouped[capacity]) for capacity in args.capacities]
    print_summary_table(summaries)

    payload = {
        "checkpoint": None if args.in_scale else str(args.checkpoint),
        "in_scale": bool(args.in_scale),
        "capacity_checkpoints": {capacity_tag(k): v for k, v in capacity_checkpoints.items()},
        "n_node": args.n_node,
        "n_instances": args.n_instances,
        "seed": args.seed,
        "capacities": args.capacities,
        "config": {
            "alg": args.alg,
            "k_sparse": args.k_sparse,
            "n_ants": args.n_ants,
            "H": args.H,
            "mini_H": args.mini_H,
            "rho": args.rho,
            "min_new_edges": args.min_new_edges,
            "no_anneal": args.no_anneal,
            "no_smooth_mmas": args.no_smooth_mmas,
            "no_local_search": args.no_local_search,
            "ls_scope": args.ls_scope,
            "ls_budget": args.ls_budget,
            "ls_max_opt": args.ls_max_opt,
        },
        "summary": [asdict(summary) for summary in summaries],
        "instances": [asdict(row) for row in rows],
        "outer_h_metrics": outer_h_metrics_by_capacity,
    }
    return payload


def write_outputs(payload: dict[str, Any], json_path: Path, csv_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "capacity",
                "checkpoint",
                "instance",
                "cost",
                "elapsed_s",
                "route_count",
                "customers_per_subroute",
                "subroute_cost",
                "subroute_load",
                "subroute_utilization",
                "mean_guidance",
                "pheromone_correlation",
            ],
        )
        writer.writeheader()
        writer.writerows(payload["instances"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a CVRP-1K checkpoint over a sweep of synthetic capacity settings."
    )
    default_checkpoint = resolve_default_checkpoint()

    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--in_scale", action="store_true",
                        help="Resolve and load the matched capacity-specific checkpoint for each capacity.")
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_INSCALE_CHECKPOINT_ROOT,
                        help="Root directory containing per-scale in-scale checkpoints (used with --in_scale).")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_node", type=int, default=1000)
    parser.add_argument("--n_instances", type=int, default=16)
    parser.add_argument(
        "--capacities",
        type=parse_capacities,
        default=parse_capacities("100,150,200,250,300,400,500"),
        help="Comma-separated raw capacities to evaluate",
    )
    parser.add_argument("--alg", choices=["faco", "mmas"], default="faco")
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--min_new_edges", type=int, default=12)
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_smooth_mmas", action="store_true", default=True)
    parser.add_argument("--smooth_mmas", dest="no_smooth_mmas", action="store_false")
    parser.add_argument("--no_extend_ls", action="store_true")
    parser.add_argument("--ls_scope", choices=["localized", "global"], default="localized")
    parser.add_argument("--ls_budget", choices=["truncated", "full"], default="truncated")
    parser.add_argument("--ls_max_opt", type=int, default=0)
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--no_normalized_heuristic", action="store_true")
    parser.add_argument("--no_logit_net", action="store_true")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    parser.add_argument("--no_anneal", action="store_true", default=True)
    parser.add_argument("--anneal", dest="no_anneal", action="store_false")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(cli_argv)
    if args.output_json is None or args.output_csv is None:
        default_ref = args.checkpoint if not args.in_scale else (args.checkpoint_root / f"n{args.n_node}" / "in_scale")
        default_json, default_csv = default_output_paths(default_ref)
        if args.output_json is None:
            args.output_json = default_json
        if args.output_csv is None:
            args.output_csv = default_csv
    if args.output_md is None:
        args.output_md = default_markdown_output_path()

    payload = run_capacity_sweep(args, cli_argv)
    write_outputs(payload, args.output_json, args.output_csv, args.output_md)
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")
    print(f"Wrote MD:   {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
