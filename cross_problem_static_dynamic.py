#!/usr/bin/env python3
"""
Cross-problem static-vs-dynamic experiment driver for BPP, MKP, and OP.

This script addresses the reviewer concern around "static heatmap -> dynamic
policy" novelty by training and evaluating paired variants:

- dynamic: default extended model with dynamic pheromone-derived edge features
- static: identical model architecture with dynamic channels zeroed via
  ``--no_dynamic_feats``
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "H": 5,
    "mini_H": 10,
    "n_ants": 100,
    "k_sparse": 32,
    "epochs": 10,
    "steps_per_epoch": 32,
    "val_size": 16,
    "test_size": 16,
    "T": 5,
    "lr": 5e-6,
    "rho": 0.5,
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "min_gamma": 0.0,
}

BPP_CONFIG: Dict[str, Any] = {
    "problem": "bpp",
    "capacity": 150.0,
}

MKP_CONFIG: Dict[str, Any] = {
    "problem": "mkp",
    "m": 5,
}

OP_CONFIG: Dict[str, Any] = {
    "problem": "op",
    "max_len": 4.0,
}

VARIANTS: Dict[str, Dict[str, Any]] = {
    "dynamic": {},
    "static": {"no_dynamic_feats": True},
}

ROOT_DIR = Path("cross_problem_experiments") / "static_dynamic"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"
PROGRESS_FILE = ROOT_DIR / "experiment_progress.json"
PER_VARIANT_CSV = RESULTS_DIR / "per_variant.csv"
COMPARISON_CSV = RESULTS_DIR / "dynamic_vs_static.csv"

for _path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    _path.mkdir(parents=True, exist_ok=True)


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_progress() -> Dict[str, Any]:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        with open(PROGRESS_FILE, "r") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def save_progress(key: str, payload: Dict[str, Any]) -> None:
    progress = load_progress()
    progress[key] = payload
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as handle:
        json.dump(progress, handle, indent=2)


def run_command(cmd: List[str], log_file: Path, dry_run: bool = False) -> tuple[int, str, str]:
    cmd_str = " ".join(cmd)
    print(f"[CMD] {cmd_str}")
    if dry_run:
        return 0, "DRY_RUN", cmd_str

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as handle:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = ""
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            output += line
        process.wait()
    return process.returncode, output, cmd_str


def build_problem_config(problem: str, n_node: int) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if problem == "bpp":
        config.update(BPP_CONFIG)
    elif problem == "mkp":
        config.update(MKP_CONFIG)
    elif problem == "op":
        config.update(OP_CONFIG)
    else:
        raise ValueError(f"Unknown problem: {problem}")
    config["n_node"] = n_node
    return config


def apply_debug_overrides(config: Dict[str, Any], test_debug: bool) -> Dict[str, Any]:
    if not test_debug:
        return config
    updated = config.copy()
    updated.update({
        "H": 5,
        "mini_H": 5,
        "n_ants": 5,
        "k_sparse": 5,
        "epochs": 2,
        "steps_per_epoch": 2,
        "val_size": 1,
        "test_size": 2,
        "T": 3,
    })
    return updated


def apply_dry_run_overrides(config: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    if not dry_run:
        return config
    updated = config.copy()
    updated.update({
        "H": 2,
        "mini_H": 2,
        "n_ants": min(int(config.get("n_ants", 10)), 2),
        "k_sparse": min(int(config.get("k_sparse", 10)), max(1, int(config["n_node"]) - 1)),
        "epochs": 1,
        "steps_per_epoch": 1,
        "val_size": 1,
        "test_size": 1,
        "T": 2,
    })
    return updated


def get_model_path(problem: str, variant: str, n_node: int) -> Path:
    return MODELS_DIR / variant / f"{problem}_n{n_node}_best.pt"


def get_result_json_path(problem: str, variant: str, n_node: int) -> Path:
    return RESULTS_DIR / variant / f"{problem}_n{n_node}_results.json"


def train_variant(
    config: Dict[str, Any],
    variant: str,
    *,
    force: bool,
    dry_run: bool,
    only_test: bool,
) -> Optional[Path]:
    problem = config["problem"]
    n_node = int(config["n_node"])
    model_path = get_model_path(problem, variant, n_node)

    if only_test:
        return model_path if model_path.exists() else None

    if model_path.exists() and not force:
        print(f"[SKIP] Existing model: {model_path}")
        return model_path

    save_dir = MODELS_DIR / variant
    cmd = [sys.executable, "train_extended.py"]
    for key, value in config.items():
        if key == "test_size":
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    for key, value in VARIANTS[variant].items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    cmd.extend(["--save_dir", str(save_dir)])
    cmd.extend(["--wandb_project", ""])
    cmd.extend(["--run_name", f"{problem}_n{n_node}_{variant}"])

    log_file = LOGS_DIR / f"train_{problem}_n{n_node}_{variant}.log"
    code, _, _ = run_command(cmd, log_file, dry_run=dry_run)
    return model_path if code == 0 else None


def test_variant(
    model_path: Path,
    config: Dict[str, Any],
    variant: str,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    problem = config["problem"]
    n_node = int(config["n_node"])
    result_json = get_result_json_path(problem, variant, n_node)

    cmd = [
        sys.executable,
        "test_extended.py",
        "--problem", problem,
        "--n_node", str(n_node),
        "--checkpoint", str(model_path),
        "--save_results",
        "--save_dir", str(result_json.parent),
        "--test_size", str(config["test_size"]),
        "--T", str(config["T"]),
        "--n_ants", str(config["n_ants"]),
        "--k_sparse", str(config["k_sparse"]),
        "--rho", str(config["rho"]),
        "--alpha", str(config["alpha"]),
        "--beta", str(config["beta"]),
    ]

    if problem == "mkp":
        cmd.extend(["--m", str(config["m"])])
    elif problem == "bpp":
        cmd.extend(["--capacity", str(config["capacity"])])
    elif problem == "op":
        cmd.extend(["--max_len", str(config["max_len"])])

    for key, value in VARIANTS[variant].items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    log_file = LOGS_DIR / f"test_{problem}_n{n_node}_{variant}.log"
    code, output, cmd_str = run_command(cmd, log_file, dry_run=dry_run)

    metrics: Dict[str, Any] = {
        "problem": problem,
        "variant": variant,
        "n_node": n_node,
        "status": "ok" if code == 0 else "failed",
        "returncode": code,
        "command": cmd_str,
        "model_path": str(model_path),
        "result_json": str(result_json),
        "log_file": str(log_file),
        "timestamp": datetime.now().isoformat(),
    }

    if dry_run and code == 0:
        metrics["dry_run"] = True
        return metrics

    if result_json.exists():
        with open(result_json, "r") as handle:
            payload = json.load(handle)
        metrics.update({
            "avg_cost_mean": payload.get("avg_cost_mean"),
            "avg_cost_std": payload.get("avg_cost_std"),
            "best_cost_mean": payload.get("best_cost_mean"),
            "best_cost_std": payload.get("best_cost_std"),
            "aco_cost_mean": payload.get("aco_cost_mean"),
            "aco_cost_std": payload.get("aco_cost_std"),
        })
    else:
        metrics["output_excerpt"] = output[-500:]

    return metrics


def run_problem(
    problem: str,
    n_node: int,
    *,
    dry_run: bool,
    only_test: bool,
    force: bool,
    test_debug: bool,
) -> None:
    print(f"\n=== {problem.upper()} static-vs-dynamic (n={n_node}) ===")
    config = build_problem_config(problem, n_node)
    config = apply_debug_overrides(config, test_debug)
    config = apply_dry_run_overrides(config, dry_run)

    for variant in ("dynamic", "static"):
        key = f"{problem}_n{config['n_node']}_{variant}"
        print(f"\n--- {variant.upper()} ---")
        model_path = train_variant(config, variant, force=force, dry_run=dry_run, only_test=only_test)
        if model_path is None:
            payload = {
                "problem": problem,
                "variant": variant,
                "n_node": int(config["n_node"]),
                "status": "missing_model" if only_test else "train_failed",
                "timestamp": datetime.now().isoformat(),
            }
            save_progress(key, payload)
            continue

        metrics = test_variant(model_path, config, variant, dry_run=dry_run)
        save_progress(key, metrics)


def write_summary_tables() -> None:
    progress = load_progress()
    variant_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, int], Dict[str, Dict[str, Any]]] = {}

    for key, payload in progress.items():
        problem = payload.get("problem")
        variant = payload.get("variant")
        n_node = payload.get("n_node")
        if problem is None or variant is None or n_node is None:
            continue
        variant_rows.append({
            "problem": problem,
            "variant": variant,
            "n_node": n_node,
            "status": payload.get("status"),
            "avg_cost_mean": payload.get("avg_cost_mean"),
            "best_cost_mean": payload.get("best_cost_mean"),
            "aco_cost_mean": payload.get("aco_cost_mean"),
            "model_path": payload.get("model_path"),
            "result_json": payload.get("result_json"),
            "log_file": payload.get("log_file"),
            "timestamp": payload.get("timestamp"),
        })
        grouped.setdefault((problem, int(n_node)), {})[variant] = payload

    variant_rows.sort(key=lambda row: (row["problem"], int(row["n_node"]), row["variant"]))
    write_csv_rows(
        PER_VARIANT_CSV,
        [
            "problem", "variant", "n_node", "status",
            "avg_cost_mean", "best_cost_mean", "aco_cost_mean",
            "model_path", "result_json", "log_file", "timestamp",
        ],
        variant_rows,
    )

    comparison_rows: List[Dict[str, Any]] = []
    for (problem, n_node), variants in sorted(grouped.items()):
        dynamic = variants.get("dynamic")
        static = variants.get("static")
        if dynamic is None or static is None:
            continue
        comparison_rows.append({
            "problem": problem,
            "n_node": n_node,
            "dynamic_status": dynamic.get("status"),
            "static_status": static.get("status"),
            "dynamic_avg_cost_mean": dynamic.get("avg_cost_mean"),
            "static_avg_cost_mean": static.get("avg_cost_mean"),
            "avg_cost_delta_dynamic_minus_static": _subtract(dynamic.get("avg_cost_mean"), static.get("avg_cost_mean")),
            "dynamic_best_cost_mean": dynamic.get("best_cost_mean"),
            "static_best_cost_mean": static.get("best_cost_mean"),
            "best_cost_delta_dynamic_minus_static": _subtract(dynamic.get("best_cost_mean"), static.get("best_cost_mean")),
            "dynamic_aco_cost_mean": dynamic.get("aco_cost_mean"),
            "static_aco_cost_mean": static.get("aco_cost_mean"),
            "aco_cost_delta_dynamic_minus_static": _subtract(dynamic.get("aco_cost_mean"), static.get("aco_cost_mean")),
            "timestamp": datetime.now().isoformat(),
        })

    write_csv_rows(
        COMPARISON_CSV,
        [
            "problem", "n_node",
            "dynamic_status", "static_status",
            "dynamic_avg_cost_mean", "static_avg_cost_mean", "avg_cost_delta_dynamic_minus_static",
            "dynamic_best_cost_mean", "static_best_cost_mean", "best_cost_delta_dynamic_minus_static",
            "dynamic_aco_cost_mean", "static_aco_cost_mean", "aco_cost_delta_dynamic_minus_static",
            "timestamp",
        ],
        comparison_rows,
    )


def _subtract(lhs: Any, rhs: Any) -> Optional[float]:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-problem static-vs-dynamic comparison for BPP, MKP, and OP"
    )
    parser.add_argument("--phase", choices=["all", "bpp", "mkp", "op"], default="all")
    parser.add_argument("--n_node", type=int, default=100, help="Problem size for the chosen phase")
    parser.add_argument("--dry-run", action="store_true", help="Run a minimal end-to-end command set")
    parser.add_argument("--test", action="store_true", help="Only evaluate existing checkpoints")
    parser.add_argument("--force", action="store_true", help="Retrain even if a checkpoint already exists")
    parser.add_argument("--test-debug", action="store_true", help="Use smaller configs for quick validation")
    args = parser.parse_args()

    print("=" * 80)
    print("Cross-Problem Static-vs-Dynamic Pipeline")
    print("=" * 80)
    print(f"Phase: {args.phase}")
    print(f"Problem size: {args.n_node}")
    print(f"Dry run: {args.dry_run}")
    print(f"Test only: {args.test}")
    print("=" * 80)

    if args.phase in ("all", "bpp"):
        run_problem("bpp", args.n_node, dry_run=args.dry_run, only_test=args.test, force=args.force, test_debug=args.test_debug)
    if args.phase in ("all", "mkp"):
        run_problem("mkp", args.n_node, dry_run=args.dry_run, only_test=args.test, force=args.force, test_debug=args.test_debug)
    if args.phase in ("all", "op"):
        run_problem("op", args.n_node, dry_run=args.dry_run, only_test=args.test, force=args.force, test_debug=args.test_debug)

    write_summary_tables()

    print("\n" + "=" * 80)
    print("Static-vs-dynamic experiments completed")
    print(f"Per-variant CSV: {PER_VARIANT_CSV}")
    print(f"Comparison CSV: {COMPARISON_CSV}")
    print(f"Progress file: {PROGRESS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
