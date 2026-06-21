#!/usr/bin/env python3
"""Draw CVRP training/test coordinate PDFs.

The default outputs are:
  - a synthetic CVRP-1K training-style instance
  - the largest parsed CVRPLIB test instance in data/CVRP/data/test_set
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_GLOB = "data/CVRP/data/test_set/CVRPlib*.txt"
KEYWORDS = {"name", "depot", "customer", "demand", "capacity", "cost", "edge_weight_type", "end"}


@dataclass(frozen=True)
class CvrpInstance:
    name: str
    coords: np.ndarray
    demand: np.ndarray | None
    capacity: float | None
    cost: float | None
    source: Path | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_customers(self) -> int:
        return self.n_nodes - 1


def _keyword_indices(parts: list[object]) -> dict[str, int]:
    return {str(part): idx for idx, part in enumerate(parts) if str(part) in KEYWORDS}


def _slice_until_next_keyword(parts: list[object], start_idx: int, indices: dict[str, int]) -> list[object]:
    next_keyword_positions = [idx for idx in indices.values() if idx > start_idx]
    end_idx = min(next_keyword_positions) if next_keyword_positions else len(parts)
    return parts[start_idx + 1 : end_idx]


def _as_float_array(values: Iterable[object]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=np.float64)


def parse_cvrplib_line(line: str, source: Path | None = None, line_idx: int = 0) -> CvrpInstance | None:
    line = line.strip()
    if not line:
        return None

    try:
        parts = ast.literal_eval(line)
    except (ValueError, SyntaxError):
        parts = [item.strip() for item in line.split(",")]

    if not isinstance(parts, list):
        return None

    indices = _keyword_indices(parts)
    if not {"depot", "customer"}.issubset(indices):
        return None

    name = f"Instance_{line_idx}"
    if "name" in indices and indices["name"] + 1 < len(parts):
        name = str(parts[indices["name"] + 1])

    depot_values = _slice_until_next_keyword(parts, indices["depot"], indices)
    if len(depot_values) < 2:
        raise ValueError(f"{source}:{line_idx + 1} has no depot coordinate pair")
    depot = _as_float_array(depot_values[:2]).reshape(1, 2)

    customer_values = _slice_until_next_keyword(parts, indices["customer"], indices)
    customer_flat = _as_float_array(customer_values)
    if customer_flat.size % 2:
        raise ValueError(f"{source}:{line_idx + 1} has an odd number of customer coordinates")
    customers = customer_flat.reshape(-1, 2)
    coords = np.vstack([depot, customers])

    demand = None
    if "demand" in indices:
        demand = _as_float_array(_slice_until_next_keyword(parts, indices["demand"], indices))
        if demand.size == coords.shape[0] - 1:
            demand = np.concatenate([[0.0], demand])

    capacity = None
    if "capacity" in indices and indices["capacity"] + 1 < len(parts):
        capacity = float(parts[indices["capacity"] + 1])

    cost = None
    if "cost" in indices and indices["cost"] + 1 < len(parts):
        cost = float(parts[indices["cost"] + 1])

    return CvrpInstance(name=name, coords=coords, demand=demand, capacity=capacity, cost=cost, source=source)


def load_cvrplib_instances(paths: Iterable[Path]) -> list[CvrpInstance]:
    instances: list[CvrpInstance] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh):
                parsed = parse_cvrplib_line(line, source=path, line_idx=line_idx)
                if parsed is not None:
                    instances.append(parsed)
    return instances


def generate_training_cvrp_1k(seed: int) -> CvrpInstance:
    rng = np.random.default_rng(seed)
    n_customers = 1000
    capacity = 250.0
    coords = rng.random((n_customers + 1, 2), dtype=np.float64)
    demand = np.concatenate([[0.0], rng.integers(1, 10, size=n_customers).astype(np.float64) / capacity])
    return CvrpInstance(
        name=f"Synthetic-CVRP-1K-seed{seed}",
        coords=coords,
        demand=demand,
        capacity=1.0,
        cost=None,
        source=None,
    )


def _normalize_for_display(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    span = float(np.max(coords.max(axis=0) - mins))
    if span <= 0:
        return coords - mins
    return (coords - mins) / span


def _square_limits(coords: np.ndarray, pad_frac: float = 0.025) -> tuple[tuple[float, float], tuple[float, float]]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    side = float(np.max(maxs - mins))
    side = max(side, 1e-9)
    half = side * (0.5 + pad_frac)
    return (float(center[0] - half), float(center[0] + half)), (float(center[1] - half), float(center[1] + half))


def draw_instance(instance: CvrpInstance, output_path: Path, normalize: bool) -> None:
    coords = _normalize_for_display(instance.coords) if normalize else instance.coords
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    customers = coords[1:]
    demand = instance.demand[1:] if instance.demand is not None and len(instance.demand) == instance.n_nodes else None
    large_instance = instance.n_customers > 5000
    point_size = 3.8 if large_instance else 13.0
    point_alpha = 0.82 if large_instance else 0.90

    if demand is None:
        ax.scatter(
            customers[:, 0],
            customers[:, 1],
            s=point_size,
            c="#145ea8",
            alpha=point_alpha,
            linewidths=0,
            rasterized=large_instance,
        )
    else:
        ax.scatter(
            customers[:, 0],
            customers[:, 1],
            s=point_size,
            c=demand,
            cmap="plasma",
            alpha=point_alpha,
            linewidths=0,
            rasterized=large_instance,
        )

    depot_size = 666
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        s=depot_size * 1.28,
        marker="*",
        c="#111111",
        edgecolors="#111111",
        linewidths=0,
        zorder=4,
    )
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        s=depot_size,
        marker="*",
        c="#ff0000",
        edgecolors="black",
        linewidths=1.35,
        zorder=5,
    )

    ax.set_aspect("equal", adjustable="box")
    xlim, ylim = _square_limits(coords)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#222222")
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def select_test_instance(instances: list[CvrpInstance], test_name: str | None, test_rank: int) -> CvrpInstance:
    if test_name:
        needle = test_name.lower()
        matches = [item for item in instances if needle in item.name.lower()]
        if not matches:
            available = ", ".join(item.name for item in sorted(instances, key=lambda x: x.n_customers, reverse=True)[:10])
            raise ValueError(f"No CVRPLIB instance name contains {test_name!r}. Largest available: {available}")
        if len(matches) > 1:
            matches = sorted(matches, key=lambda item: item.n_customers, reverse=True)
            print(
                f"Matched {len(matches)} instances for --test-name {test_name!r}; "
                f"using largest match {matches[0].name}."
            )
        return matches[0]

    ranked = sorted(instances, key=lambda item: item.n_customers, reverse=True)
    if test_rank < 1 or test_rank > len(ranked):
        raise ValueError(f"--test-rank must be in [1, {len(ranked)}], got {test_rank}")
    return ranked[test_rank - 1]


def print_test_instances(instances: list[CvrpInstance]) -> None:
    for rank, item in enumerate(sorted(instances, key=lambda x: x.n_customers, reverse=True), start=1):
        source = item.source.relative_to(REPO_ROOT) if item.source and item.source.is_relative_to(REPO_ROOT) else item.source
        print(f"{rank:3d}  {item.name:32s}  {item.n_customers:7,d} customers  {source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for the synthetic training instance.")
    parser.add_argument(
        "--test-glob",
        default=DEFAULT_TEST_GLOB,
        help="Glob, relative to repo root unless absolute, used to find CVRPLIB-style test files.",
    )
    parser.add_argument(
        "--synthetic-output",
        type=Path,
        default=REPO_ROOT / "paper/figures/cvrp_synthetic_train_1k.pdf",
        help="Output PDF path for the synthetic CVRP-1K training instance.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=REPO_ROOT / "paper/figures/cvrplib_largest_test_instance.pdf",
        help="Output PDF path for the largest CVRPLIB test instance.",
    )
    parser.add_argument(
        "--raw-test-scale",
        action="store_true",
        help="Plot the CVRPLIB instance in raw coordinates instead of normalizing for display.",
    )
    parser.add_argument(
        "--test-name",
        default=None,
        help="Case-insensitive substring of the CVRPLIB instance name to draw, e.g. Flanders1 or X-n1001.",
    )
    parser.add_argument(
        "--test-rank",
        type=int,
        default=1,
        help="Size rank to draw when --test-name is not set: 1 is largest, 2 is second largest, etc.",
    )
    parser.add_argument(
        "--list-test-instances",
        action="store_true",
        help="List parsed CVRPLIB instances by descending customer count and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = Path(args.test_glob)
    paths = sorted(pattern.glob("*") if pattern.is_absolute() else REPO_ROOT.glob(args.test_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched --test-glob {args.test_glob!r}")

    synthetic = generate_training_cvrp_1k(args.seed)
    instances = load_cvrplib_instances(paths)
    if not instances:
        raise ValueError(f"No CVRPLIB instances could be parsed from {len(paths)} matched files")
    if args.list_test_instances:
        print_test_instances(instances)
        return
    test_instance = select_test_instance(instances, args.test_name, args.test_rank)

    draw_instance(
        synthetic,
        args.synthetic_output,
        normalize=False,
    )
    draw_instance(
        test_instance,
        args.test_output,
        normalize=not args.raw_test_scale,
    )

    print(f"Wrote {args.synthetic_output}")
    print(
        f"Wrote {args.test_output} from {test_instance.source} "
        f"({test_instance.name}, {test_instance.n_customers:,} customers)"
    )


if __name__ == "__main__":
    main()
