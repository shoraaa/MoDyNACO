#!/usr/bin/env python3
"""Build a DyNACO-compatible STAR TSP <1000-node evaluation file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_tsp(path: Path) -> tuple[str, list[tuple[float, float]]]:
    name = path.stem.split("_")[-1]
    coords: list[tuple[float, float]] = []
    in_coords = False

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NAME"):
            name = line.replace(":", " ").split()[-1]
        elif upper == "NODE_COORD_SECTION":
            in_coords = True
        elif upper == "EOF":
            break
        elif in_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))

    if not coords:
        raise ValueError(f"No NODE_COORD_SECTION parsed from {path}")
    return name, coords


def load_reference_costs(path: Path) -> dict[str, float]:
    refs: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or row.get("problem") != "tsp":
                continue
            final_cost = float(row["final_cost"])
            gap_pct = float(row["gap"])
            opt = final_cost / (1.0 + gap_pct / 100.0)
            rounded = round(opt)
            refs[row["instance"]] = float(rounded) if abs(opt - rounded) < 1e-3 else opt
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--star-root",
        type=Path,
        default=Path("../STAR"),
        help="Path to the STAR checkout.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("../STAR/results_server/results/small-main/results.csv"),
        help="STAR result CSV used to recover reference costs.",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("../STAR/survey/0_data_survey/survey_bench_tsp"),
        help="STAR TSP benchmark directory containing .tsp files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/TSP/data/test_set/STAR_TSP_lt1000.txt"),
        help="Output DyNACO text dataset.",
    )
    parser.add_argument("--max-nodes", type=int, default=999)
    args = parser.parse_args()

    refs = load_reference_costs(args.results_csv)
    rows: list[tuple[str, float, list[tuple[float, float]]]] = []

    for tsp_path in sorted(args.bench_dir.glob("*.tsp")):
        name, coords = parse_tsp(tsp_path)
        if len(coords) > args.max_nodes:
            continue
        if name not in refs:
            continue
        rows.append((name, refs[name], coords))

    if not rows:
        raise SystemExit("No STAR TSP instances matched the requested filter.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for name, opt, coords in rows:
            flat = [str(v) for xy in coords for v in xy]
            f.write(repr([name, f"{opt:.10g}", *flat]) + "\n")

    print(f"Wrote {len(rows)} instances to {args.output}")
    print(f"Node range: {min(len(c) for _, _, c in rows)}-{max(len(c) for _, _, c in rows)}")


if __name__ == "__main__":
    main()
