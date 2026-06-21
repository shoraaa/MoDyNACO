#!/usr/bin/env python3
"""Benchmark MFACO sample() against sample_mixed_priors().

This is intentionally small and direct: it constructs one synthetic instance,
warms each call path, then times repeated sampling calls without pheromone
updates. Use --require-prob to include traced/log-prob paths.
"""

from __future__ import annotations

import argparse
import sys
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faco


@dataclass
class Timing:
    label: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    ratio_to_sample: float | None = None


def _make_solver(args: argparse.Namespace):
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    coords = torch.rand((args.n, 2), generator=gen, dtype=torch.float32)
    common = dict(
        n_ants=args.ants,
        cand_list_size=args.k,
        backup_list_size=args.backup_k,
        min_new_edges=args.min_new_edges,
        use_local_search=not args.no_local_search,
        ls_scope=args.ls_scope,
        ls_budget=args.ls_budget,
        ls_max_opt=args.ls_max_opt,
        enable_torch_sync=False,
        device=args.device,
    )
    if args.problem == "tsp":
        return faco.MFACO_TSP(coords, decay=args.rho, **common)

    demand = torch.randint(1, 10, (args.n,), generator=gen, dtype=torch.float32)
    demand[0] = 0.0
    demand = demand / args.capacity
    return faco.MFACO_CVRP(coords, demand, capacity=1.0, decay=args.rho, **common)


def _time_call(
    label: str,
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> Timing:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)

    return Timing(
        label=label,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def _parse_head_counts(value: str | None, heads: int, ants: int) -> np.ndarray | None:
    if value is None:
        return None
    counts = np.asarray([int(part) for part in value.split(",")], dtype=np.int32)
    if counts.shape != (heads,):
        raise SystemExit(f"--head-counts must contain exactly {heads} values")
    if np.any(counts < 0):
        raise SystemExit("--head-counts must be non-negative")
    if int(counts.sum()) != ants:
        raise SystemExit(f"--head-counts must sum to --ants ({ants})")
    return counts


def _print_table(rows: list[Timing]) -> None:
    headers = ["case", "mean_ms", "median_ms", "min_ms", "max_ms", "x_none"]
    widths = [max(len(headers[0]), *(len(r.label) for r in rows)), 10, 10, 10, 10, 9]
    print(" ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(" ".join("-" * w for w in widths))
    for r in rows:
        ratio = "-" if r.ratio_to_sample is None else f"{r.ratio_to_sample:.2f}x"
        vals = [
            r.label.ljust(widths[0]),
            f"{r.mean_ms:.3f}".rjust(widths[1]),
            f"{r.median_ms:.3f}".rjust(widths[2]),
            f"{r.min_ms:.3f}".rjust(widths[3]),
            f"{r.max_ms:.3f}".rjust(widths[4]),
            ratio.rjust(widths[5]),
        ]
        print(" ".join(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--backup-k", type=int, default=32)
    parser.add_argument("--ants", type=int, default=100)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--head-counts",
        default=None,
        help="Comma-separated ant counts per head, e.g. 100,0,0,0,0,0,0,0",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--capacity", type=float, default=250.0)
    parser.add_argument("--min-new-edges", type=int, default=12)
    parser.add_argument("--require-prob", action="store_true")
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--parallel-traced", action="store_true", default=True)
    parser.add_argument("--no-local-search", action="store_true")
    parser.add_argument("--ls-scope", choices=["localized", "global"], default="localized")
    parser.add_argument("--ls-budget", choices=["truncated", "full"], default="truncated")
    parser.add_argument("--ls-max-opt", type=int, default=0)
    args = parser.parse_args()

    if args.heads > args.ants:
        raise SystemExit("--heads must be <= --ants")
    head_counts = _parse_head_counts(args.head_counts, args.heads, args.ants)

    faco.set_faco_cpp_threads(args.threads)
    solver = _make_solver(args)
    rng = np.random.default_rng(args.seed)
    prior_np = rng.normal(0.0, 1.0, size=(solver.n, solver.k)).astype(np.float32)
    heads_np = rng.normal(0.0, 1.0, size=(args.heads, solver.n, solver.k)).astype(np.float32)
    heads_one_np = prior_np.reshape(1, solver.n, solver.k)

    prior_t = torch.from_numpy(prior_np).to(args.device)
    heads_t = torch.from_numpy(heads_np).to(args.device)
    heads_one_t = torch.from_numpy(heads_one_np).to(args.device)

    print(
        "Config: "
        f"problem={args.problem}, n={solver.n}, k={solver.k}, ants={args.ants}, "
        f"heads={args.heads}, threads={args.threads}, repeats={args.repeats}, "
        f"require_prob={args.require_prob}, local_search={not args.no_local_search}, "
        f"head_counts={args.head_counts or 'even'}, prior_scale={args.prior_scale}"
    )

    cases = [
        (
            "sample_none",
            lambda: solver.sample(
                require_prob=args.require_prob,
                prior=None,
                parallel_traced=args.parallel_traced,
            ),
        ),
        (
            "sample_numpy",
            lambda: solver.sample(
                require_prob=args.require_prob,
                prior=prior_np,
                parallel_traced=args.parallel_traced,
            ),
        ),
        (
            "mixed_1h_numpy",
            lambda: solver.sample_mixed_priors(
                heads_one_np,
                require_prob=args.require_prob,
                parallel_traced=args.parallel_traced,
            ),
        ),
        (
            f"mixed_{args.heads}h_numpy",
            lambda: solver.sample_mixed_priors(
                heads_np,
                require_prob=args.require_prob,
                parallel_traced=args.parallel_traced,
                head_counts=head_counts,
                prior_scale=args.prior_scale,
            ),
        ),
        (
            "sample_torch",
            lambda: solver.sample(
                require_prob=args.require_prob,
                prior=prior_t,
                parallel_traced=args.parallel_traced,
            ),
        ),
        (
            "mixed_1h_torch",
            lambda: solver.sample_mixed_priors(
                heads_one_t,
                require_prob=args.require_prob,
                parallel_traced=args.parallel_traced,
            ),
        ),
        (
            f"mixed_{args.heads}h_torch",
            lambda: solver.sample_mixed_priors(
                heads_t,
                require_prob=args.require_prob,
                parallel_traced=args.parallel_traced,
                head_counts=head_counts,
                prior_scale=args.prior_scale,
            ),
        ),
    ]

    rows = [
        _time_call(label, fn, warmup=args.warmup, repeats=args.repeats)
        for label, fn in cases
    ]
    baseline = rows[0].mean_ms
    for row in rows[1:]:
        row.ratio_to_sample = row.mean_ms / baseline if baseline > 0.0 else None
    _print_table(rows)


if __name__ == "__main__":
    main()
