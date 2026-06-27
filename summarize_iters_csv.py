#!/usr/bin/env python3
"""Summarize per-iteration solver CSV traces at iteration budgets."""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_METHOD_ORDER = [
    "Base",
    "Model(no_anneal)",
    "Model(anneal)",
    "Mix(no_anneal)",
    "Mix(anneal)",
]

DEFAULT_METHOD_LABELS = {
    "Base": "base",
    "Model(no_anneal)": "model(no_anneal)",
    "Model(anneal)": "model(anneal)",
    "Mix(no_anneal)": "mix(no_anneal)",
    "Mix(anneal)": "mix(anneal)",
}

ALL_GROUP = "Overall"
GROUP_LABELS = {
    "<1K": "Small (<1K nodes)",
    "[1K,10K)": "Medium (>=1K and <10K nodes)",
    ">=10K": "Large (>=10K nodes)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize an *_iters.csv trace by total iteration budget, reporting "
            "average optimality gaps and average elapsed times."
        )
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help=(
            "Path to the *_iters.csv file. A filename prefix is also accepted; "
            "the newest matching csv/*_iters.csv is used."
        ),
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=None,
        help="CSV with reference costs. Defaults to sibling *_instances.csv.",
    )
    parser.add_argument(
        "--ref-col",
        default="opt",
        help="Reference-cost column in --refs, usually opt or baseline.",
    )
    parser.add_argument(
        "--iterations",
        default="1000:10000:1000",
        help=(
            "Total-iteration budgets. Use comma list such as 1000,2000,5000 "
            "or range start:stop:step. Default: 1000:10000:1000."
        ),
    )
    parser.add_argument("--idx-col", default="idx", help="Instance id column")
    parser.add_argument(
        "--group-col",
        default="size_group",
        help="Instance size-group column in --refs. Default: size_group.",
    )
    parser.add_argument("--method-col", default="method", help="Method column")
    parser.add_argument(
        "--anneal-col",
        default="anneal",
        help=(
            "Annealing-state column. When present, non-base methods are reported "
            "as Method(anneal) or Method(no_anneal)."
        ),
    )
    parser.add_argument("--iter-col", default="iter", help="Zero-indexed iter column")
    parser.add_argument("--cost-col", default="best", help="Solution-cost column")
    parser.add_argument(
        "--time-col",
        default="elapsed_s",
        help="Elapsed-time column to average. Default: elapsed_s.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        metavar="RAW=LABEL",
        help=(
            "Method mapping to include. Can be repeated. "
            "Use derived keys such as Base, Model(anneal), Model(no_anneal), "
            "Mix(anneal), or Mix(no_anneal). Default: include observed variants."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format. Default: markdown.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the summary table to.",
    )
    parser.add_argument(
        "--no-groups",
        action="store_true",
        help="Only print the overall table, without size-group tables.",
    )
    return parser.parse_args()


def parse_iterations(spec: str) -> list[int]:
    if ":" in spec:
        parts = [int(part) for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError("--iterations range must be start:stop:step")
        start, stop, step = parts
        if step <= 0:
            raise ValueError("--iterations step must be positive")
        return list(range(start, stop + 1, step))
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("--iterations did not contain any budgets")
    return values


def parse_method_labels(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    labels: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"method mapping must be RAW=LABEL, got {item!r}")
        raw, label = item.split("=", 1)
        raw = raw.strip()
        label = label.strip()
        if not raw or not label:
            raise ValueError(f"method mapping must be RAW=LABEL, got {item!r}")
        labels[raw] = label
    return labels


def normalize_anneal(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"on", "true", "1", "yes", "anneal"}:
        return "anneal"
    if normalized in {"off", "false", "0", "no", "no_anneal"}:
        return "no_anneal"
    return None


def method_key(method: str, anneal: str | None) -> str:
    method = method.strip()
    if not method:
        return method
    if method.lower() == "base":
        return method
    anneal_key = normalize_anneal(anneal)
    if anneal_key:
        return f"{method}({anneal_key})"
    return method


def default_method_label(key: str) -> str:
    return DEFAULT_METHOD_LABELS.get(key, key[:1].lower() + key[1:])


def ordered_methods(observed: set[str], labels: dict[str, str]) -> list[tuple[str, str]]:
    ordered = [key for key in DEFAULT_METHOD_ORDER if key in observed or key in labels]
    ordered.extend(sorted(key for key in observed if key not in set(ordered)))
    ordered.extend(key for key in labels if key not in set(ordered))
    return [(key, labels.get(key, default_method_label(key))) for key in ordered]


def sibling_instances_path(csv_file: Path) -> Path:
    name = csv_file.name
    if name.endswith("_iters.csv"):
        return csv_file.with_name(name[: -len("_iters.csv")] + "_instances.csv")
    return csv_file.with_name(csv_file.stem + "_instances.csv")


def sibling_log_path(csv_file: Path) -> Path:
    name = csv_file.name
    if name.endswith("_iters.csv"):
        return csv_file.parent.parent / (name[: -len("_iters.csv")] + ".txt")
    return csv_file.parent.parent / (csv_file.stem + ".txt")


def sibling_summary_path(csv_file: Path) -> Path:
    name = csv_file.name
    if name.endswith("_iters.csv"):
        return csv_file.with_name(name[: -len("_iters.csv")] + "_summary.csv")
    return csv_file.with_name(csv_file.stem + "_summary.csv")


def resolve_csv_file(path: Path) -> Path:
    if path.exists():
        return path

    candidates: list[Path] = []
    candidates.extend(Path("csv").glob(f"{path.name}*_iters.csv"))
    candidates.extend(Path(".").glob(f"{path.name}*_iters.csv"))
    candidates.extend(path.parent.glob(f"{path.name}*_iters.csv"))

    if not candidates:
        raise FileNotFoundError(f"No CSV file found for {path}")

    def candidate_key(candidate: Path) -> tuple[int, float]:
        has_refs = sibling_instances_path(candidate).exists() or sibling_log_path(
            candidate
        ).exists()
        return (1 if has_refs else 0, candidate.stat().st_mtime)

    return max(candidates, key=candidate_key)


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def size_group_from_name(name: str) -> str | None:
    match = re.search(r"(\d+)", name)
    if not match:
        return None
    size = int(match.group(1))
    if size < 1000:
        return "<1K"
    if size < 10000:
        return "[1K,10K)"
    return ">=10K"


def load_refs(
    path: Path, idx_col: str, ref_col: str, group_col: str
) -> tuple[dict[str, float], dict[str, str]]:
    refs: dict[str, float] = {}
    groups: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {idx_col, ref_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        has_group = group_col in set(reader.fieldnames or [])
        for row in reader:
            ref = as_float(row.get(ref_col))
            if ref is not None and ref != 0:
                idx = row[idx_col]
                refs[idx] = ref
                if has_group and row.get(group_col):
                    groups[idx] = row[group_col]
    if not refs:
        raise ValueError(f"No usable reference costs found in {path}:{ref_col}")
    return refs, groups


def load_refs_from_log(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    refs: dict[str, float] = {}
    groups: dict[str, str] = {}
    pattern = re.compile(
        r"\[(?P<pos>\d+)/\d+\]\s+"
        r"(?P<name>[^|]+?)\s+\|\s+"
        r"opt\s+(?P<opt>[0-9]+(?:\.[0-9]+)?)"
    )
    with path.open(errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            idx = str(int(match.group("pos")) - 1)
            refs[idx] = float(match.group("opt"))
            group = size_group_from_name(match.group("name").strip())
            if group:
                groups[idx] = group
    if not refs:
        raise ValueError(f"No opt references found in {path}")
    return refs, groups


def summarize(args: argparse.Namespace) -> list[dict[str, object]]:
    args.csv_file = resolve_csv_file(args.csv_file)
    budgets = parse_iterations(args.iterations)
    wanted_totals = set(budgets)
    explicit_method_labels = parse_method_labels(args.method)
    refs_path = args.refs or sibling_instances_path(args.csv_file)
    if refs_path.exists():
        refs, groups = load_refs(refs_path, args.idx_col, args.ref_col, args.group_col)
    elif args.refs:
        raise FileNotFoundError(f"Reference CSV not found: {refs_path}")
    else:
        log_path = sibling_log_path(args.csv_file)
        if not log_path.exists():
            raise FileNotFoundError(
                f"Neither {refs_path} nor sibling log {log_path} exists"
            )
        refs, groups = load_refs_from_log(log_path)

    stats: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "gap_sum": 0.0,
            "time_n": 0.0,
            "time_sum": 0.0,
            "cost_sum": 0.0,
        }
    )

    with args.csv_file.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            args.idx_col,
            args.method_col,
            args.iter_col,
            args.cost_col,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.csv_file} is missing required columns: {sorted(missing)}")
        has_time = args.time_col in set(reader.fieldnames or [])
        has_anneal = args.anneal_col in set(reader.fieldnames or [])
        observed_methods: set[str] = set()

        for row in reader:
            method = method_key(
                row[args.method_col],
                row.get(args.anneal_col) if has_anneal else None,
            )
            if not method:
                continue
            if explicit_method_labels is not None and method not in explicit_method_labels:
                continue
            observed_methods.add(method)

            try:
                total_iter = int(row[args.iter_col]) + 1
            except ValueError:
                continue
            if total_iter not in wanted_totals:
                continue

            idx = row[args.idx_col]
            ref = refs.get(idx)
            cost = as_float(row.get(args.cost_col))
            elapsed = as_float(row.get(args.time_col)) if has_time else None
            if ref is None or cost is None:
                continue

            gap = (cost - ref) / ref * 100.0
            row_groups = [ALL_GROUP]
            if not args.no_groups and idx in groups:
                row_groups.append(groups[idx])
            for group in row_groups:
                key = (group, total_iter, method)
                stats[key]["n"] += 1
                stats[key]["gap_sum"] += gap
                if elapsed is not None:
                    stats[key]["time_n"] += 1
                    stats[key]["time_sum"] += elapsed
                stats[key]["cost_sum"] += cost

    rows: list[dict[str, object]] = []
    method_labels = explicit_method_labels or {
        method: default_method_label(method) for method in observed_methods
    }
    method_items = ordered_methods(observed_methods, method_labels)
    observed_groups = {group for group, _, _ in stats}
    group_order = [ALL_GROUP]
    if not args.no_groups:
        group_order.extend(label for label in GROUP_LABELS if label in observed_groups)
        group_order.extend(
            sorted(group for group in observed_groups if group not in set(group_order))
        )

    for group in group_order:
        for total_iter in budgets:
            for raw_method, label in method_items:
                s = stats.get((group, total_iter, raw_method))
                group_label = GROUP_LABELS.get(group, group)
                if not s or s["n"] == 0:
                    rows.append(
                        {
                            "group": group_label,
                            "iterations": total_iter,
                            "raw_method": raw_method,
                            "method": label,
                            "n": 0,
                            "avg_gap_pct": "",
                            "avg_time_s": "",
                            "time_n": 0,
                            "avg_cost": "",
                        }
                    )
                    continue
                n = int(s["n"])
                rows.append(
                    {
                        "group": group_label,
                        "iterations": total_iter,
                        "raw_method": raw_method,
                        "method": label,
                        "n": n,
                        "avg_gap_pct": s["gap_sum"] / n,
                        "avg_time_s": (
                            s["time_sum"] / s["time_n"] if s["time_n"] else ""
                        ),
                        "time_n": int(s["time_n"]),
                        "avg_cost": s["cost_sum"] / n,
                    }
                )
    return rows


def format_number(value: object, digits: int) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def render_markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Iterations | Method | N | Avg Gap % | Avg Time s | Avg Cost |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {iterations:,} | {method} | {n} | {gap} | {time} | {cost} |".format(
                iterations=int(row["iterations"]),
                method=row["method"],
                n=int(row["n"]),
                gap=format_number(row["avg_gap_pct"], 3),
                time=format_number(row["avg_time_s"], 3),
                cost=format_number(row["avg_cost"], 3),
            )
        )
    return lines


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    groups = []
    seen = set()
    for row in rows:
        group = str(row["group"])
        if group not in seen:
            groups.append(group)
            seen.add(group)
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        if lines:
            lines.append("")
        lines.append(f"## {group}")
        lines.extend(render_markdown_table(group_rows))
    return "\n".join(lines)


def render_csv(rows: list[dict[str, object]]) -> str:
    fieldnames = [
        "group",
        "iterations",
        "method",
        "n",
        "avg_gap_pct",
        "avg_time_s",
        "avg_cost",
    ]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(fieldnames)
    for row in rows:
        writer.writerow(
            [
                row["group"],
                row["iterations"],
                row["method"],
                row["n"],
                format_number(row["avg_gap_pct"], 6),
                format_number(row["avg_time_s"], 6),
                format_number(row["avg_cost"], 6),
            ]
        )
    return output.getvalue().rstrip()


def main() -> None:
    args = parse_args()
    rows = summarize(args)
    rendered = render_csv(rows) if args.format == "csv" else render_markdown(rows)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
