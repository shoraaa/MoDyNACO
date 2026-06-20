#!/usr/bin/env python3
"""Generate a fixed CVRP-1K validation set from robust synthetic training distribution."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baselines
import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed robust CVRP-1K validation data with optional HGS baselines."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/cvrp/valDataset-1000-robust.pt")
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--n_instances", type=int, default=128)
    parser.add_argument("--n_node", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--baseline", choices=["hgs", "none"], default="hgs")
    parser.add_argument("--hgs_time_limit", type=float, default=2.0)
    parser.add_argument("--hgs_seed", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    args = parse_args()
    if args.n_node != 1000:
        raise ValueError("Robust CVRP validation generation is currently defined only for n_node=1000")
    if args.baseline == "hgs" and baselines.hgs is None:
        raise ImportError("hygese is not installed; rerun with --baseline none or install hygese")

    set_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or args.output.with_suffix(".csv")
    summary_json = args.summary_json or args.output.with_suffix(".json")

    dataset = []
    rows = []
    t0 = time.time()
    for idx in tqdm(range(args.n_instances), desc="Generating robust CVRP validation"):
        coords, demand, capacity = train.gen_robust_cvrp_1k_instance(args.n_node, args.device)
        coords_np = coords.detach().cpu().numpy().astype(np.float32)
        demand_np = demand.detach().cpu().numpy().astype(np.float32)
        capacity_val = float(capacity)

        if args.baseline == "hgs":
            cost = baselines.solve_with_hgs(
                coords_np,
                demand_np,
                capacity_val,
                time_limit=args.hgs_time_limit,
                seed=args.hgs_seed + idx,
            )
            tour = []
        else:
            cost = 0.0
            tour = []

        name = f"RobustCVRP1K_{idx:04d}"
        dataset.append((coords_np, demand_np, capacity_val, float(cost), tour, name))
        rows.append(
            {
                "idx": idx,
                "name": name,
                "n_node": args.n_node,
                "capacity": capacity_val,
                "baseline": args.baseline,
                "baseline_cost": float(cost),
                "demand_mean": float(demand_np[1:].mean()),
                "demand_q95": float(np.quantile(demand_np[1:], 0.95)),
                "demand_max": float(demand_np[1:].max()),
                "demand_sum": float(demand_np[1:].sum()),
            }
        )

    torch.save(dataset, args.output)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "output": str(args.output),
        "n_instances": args.n_instances,
        "n_node": args.n_node,
        "seed": args.seed,
        "baseline": args.baseline,
        "hgs_time_limit": args.hgs_time_limit,
        "elapsed_s": time.time() - t0,
        "baseline_cost_mean": float(np.mean([r["baseline_cost"] for r in rows])),
        "demand_mean_mean": float(np.mean([r["demand_mean"] for r in rows])),
        "demand_q95_mean": float(np.mean([r["demand_q95"] for r in rows])),
        "demand_max_mean": float(np.mean([r["demand_max"] for r in rows])),
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote dataset: {args.output}")
    print(f"Wrote CSV:     {summary_csv}")
    print(f"Wrote JSON:    {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
