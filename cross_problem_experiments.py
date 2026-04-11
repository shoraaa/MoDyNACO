#!/usr/bin/env python3
"""
Cross-Problem Experiments for BPP, MKP, OP

This module orchestrates experiments for BPP, MKP, and OP problems
to demonstrate the broader applicability of the learning-guided ACO framework.
"""

import argparse
import subprocess
import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# =============================================================================
# Configuration & Constants
# =============================================================================

# Default Hyperparameters for small-scale testing
DEFAULT_CONFIG = {
    "H": 5,
    "mini_H": 10,
    "n_ants": 100,
    "k_sparse": 32,
    "epochs": 10,
    "steps_per_epoch": 32,
    "val_size": 16,
    "T": 5,
    "lr": 5e-6,
    "rho": 0.5,
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "min_gamma": 0.0,
    "save_dir": "cross_problem_experiments/models",
}

# Problem specific defaults
BPP_CONFIG = {
    "problem": "bpp",
    "capacity": 150.0,
}

MKP_CONFIG = {
    "problem": "mkp",
    "m": 5,
}

OP_CONFIG = {
    "problem": "op",
    "max_len": 4.0,
}

# Progress file for resumption
PROGRESS_FILE = "cross_problem_experiments/experiment_progress.json"

# Results directory
RESULTS_DIR = Path("cross_problem_experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
TRAIN_ARTIFACTS_DIR = ARTIFACTS_DIR / "train"
TEST_ARTIFACTS_DIR = ARTIFACTS_DIR / "test"
for _dir in [ARTIFACTS_DIR, TRAIN_ARTIFACTS_DIR, TEST_ARTIFACTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Functions
# =============================================================================

def write_json(path: Path, payload: Dict[str, Any]):
    """Write JSON to file."""
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    """Write rows to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_progress() -> Dict[str, Any]:
    """Load experiment progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
                return {k: v for k, v in data.items() if v is not None}
        except json.JSONDecodeError:
            print(f"[WARNING] Could not decode {PROGRESS_FILE}. Starting fresh.")
            return {}
    return {}


def save_progress(key: str, data: Any):
    """Save a single experiment result to the progress file."""
    if data is None:
        print(f"[PROGRESS] Skipping save for '{key}' (No data)")
        return

    progress = load_progress()
    progress[key] = data
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)
    print(f"[PROGRESS] Saved result for '{key}'")


def run_command(cmd: List[str], log_file: Path = None, dry_run: bool = False) -> tuple:
    """Execute a shell command and returns (returncode, output, cmd_str)."""
    cmd_str = " ".join(cmd)
    print(f"[CMD] {cmd_str}")

    if dry_run:
        return 0, "DRY_RUN", cmd_str

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = ""
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                output += line
            process.wait()
            return process.returncode, output, cmd_str
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] Command failed with code {result.returncode}")
            print(result.stderr)
        return result.returncode, result.stdout, cmd_str


def get_model_path(config: Dict[str, Any], suffix: str = "_best.pt") -> Path:
    """Constructs model path based on config."""
    problem = config.get("problem", "unknown")
    n_node = config.get("n_node", 20)
    name = f"{problem}_n{n_node}_best.pt"

    save_dir = Path(config.get("save_dir", "cross_problem_experiments/models")) / problem
    return save_dir / name


def train_model(config: Dict[str, Any], dry_run: bool = False, force: bool = False,
               only_test: bool = False) -> Optional[Path]:
    """Runs training if checkpoint doesn't exist."""
    model_path = get_model_path(config)

    if only_test:
        if model_path.exists():
            print(f"[TEST-ONLY] Found model: {model_path}")
            return model_path
        else:
            print(f"[TEST-ONLY] Model not found, skipping: {model_path}")
            return None

    if model_path.exists() and not force:
        print(f"[SKIP] Model exists: {model_path}")
        return model_path

    cmd = [sys.executable, "train_extended.py"]

    # Add arguments
    for k, v in config.items():
        if k in ["save_dir"]:
            cmd.extend(["--" + k, str(v)])
        elif isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    if dry_run:
        cmd.extend(["--epochs", "1", "--steps_per_epoch", "1"])

    log_file = Path("cross_problem_experiments/logs") / f"train_{config['problem']}_n{config['n_node']}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    returncode, output, cmd_str = run_command(cmd, log_file, dry_run)

    if returncode != 0:
        print(f"[ERROR] Training failed for {config['problem']}")
        return None

    return model_path


def test_model(
    model_path: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
    summary_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs evaluation."""
    cmd = [sys.executable, "test_extended.py"]
    cmd.extend(["--problem", config["problem"]])
    cmd.extend(["--n_node", str(config["n_node"])])
    cmd.extend(["--checkpoint", str(model_path)])

    # Add relevant args
    for k, v in config.items():
        if k in ["n_ants", "k_sparse", "T", "rho", "alpha", "beta", "m", "capacity", "max_len", "test_size"]:
            cmd.extend([f"--{k}", str(v)])
        elif isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")

    if summary_json is not None:
        cmd.extend(["--save_results"])
        cmd.extend(["--save_dir", str(summary_json.parent)])

    if dry_run:
        cmd.extend(["--test_size", "1"])

    log_file = Path("cross_problem_experiments/logs") / f"test_{config['problem']}_n{config['n_node']}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    code, output, cmd_str = run_command(cmd, log_file, dry_run)

    metrics = {
        "status": "ok" if code == 0 else "failed",
        "returncode": code,
        "command": cmd_str,
        "log_file": str(log_file),
        "model_path": str(model_path),
    }

    # Try to parse results from output
    if "Avg Cost:" in output:
        lines = output.split("\n")
        for line in lines:
            if "Avg Cost:" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    metrics["avg_cost"] = float(parts[1].strip().split()[0])
            elif "Best Cost:" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    metrics["best_cost"] = float(parts[1].strip().split()[0])
            elif "ACO Cost:" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    metrics["aco_cost"] = float(parts[1].strip().split()[0])

    if summary_json is not None and summary_json.exists():
        with open(summary_json, "r") as f:
            metrics["summary"] = json.load(f)

    return metrics


# =============================================================================
# Experiment Functions
# =============================================================================

def run_bpp_experiments(n_node: int = 20, dry_run: bool = False, only_test: bool = False):
    """
    Run BPP experiments.

    Args:
        n_node: Problem size
        dry_run: Dry run mode
        only_test: Test only mode
    """
    print(f"\n=== BPP Experiments (n={n_node}) ===")

    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(BPP_CONFIG)
    base_cfg["n_node"] = n_node

    results = load_progress()
    csv_path = RESULTS_DIR / "bpp_experiments.csv"
    fieldnames = ["problem", "n_node", "avg_cost", "best_cost", "aco_cost", "timestamp"]

    key = f"bpp_n{n_node}"

    if key in results and not dry_run:
        print(f"[SKIP] {key} already completed")
        return

    # Train model
    print(f"\n--- Training BPP (n={n_node}) ---")
    model_path = train_model(base_cfg, dry_run, only_test=only_test)

    if model_path is None:
        print(f"[SKIP] Model not found for BPP")
        return

    # Test model
    print(f"\n--- Testing BPP (n={n_node}) ---")
    metrics = test_model(model_path, base_cfg, dry_run)

    row = {
        "problem": "bpp",
        "n_node": n_node,
        "avg_cost": metrics.get("avg_cost", ""),
        "best_cost": metrics.get("best_cost", ""),
        "aco_cost": metrics.get("aco_cost", ""),
        "timestamp": datetime.now().isoformat(),
    }
    write_csv_rows(csv_path, fieldnames, [row])
    save_progress(key, metrics)

    print(f"\n[RESULTS] BPP experiments saved to {csv_path}")


def run_mkp_experiments(n_node: int = 20, dry_run: bool = False, only_test: bool = False):
    """
    Run MKP experiments.

    Args:
        n_node: Problem size
        dry_run: Dry run mode
        only_test: Test only mode
    """
    print(f"\n=== MKP Experiments (n={n_node}) ===")

    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(MKP_CONFIG)
    base_cfg["n_node"] = n_node

    results = load_progress()
    csv_path = RESULTS_DIR / "mkp_experiments.csv"
    fieldnames = ["problem", "n_node", "avg_cost", "best_cost", "aco_cost", "timestamp"]

    key = f"mkp_n{n_node}"

    if key in results and not dry_run:
        print(f"[SKIP] {key} already completed")
        return

    # Train model
    print(f"\n--- Training MKP (n={n_node}) ---")
    model_path = train_model(base_cfg, dry_run, only_test=only_test)

    if model_path is None:
        print(f"[SKIP] Model not found for MKP")
        return

    # Test model
    print(f"\n--- Testing MKP (n={n_node}) ---")
    metrics = test_model(model_path, base_cfg, dry_run)

    row = {
        "problem": "mkp",
        "n_node": n_node,
        "avg_cost": metrics.get("avg_cost", ""),
        "best_cost": metrics.get("best_cost", ""),
        "aco_cost": metrics.get("aco_cost", ""),
        "timestamp": datetime.now().isoformat(),
    }
    write_csv_rows(csv_path, fieldnames, [row])
    save_progress(key, metrics)

    print(f"\n[RESULTS] MKP experiments saved to {csv_path}")


def run_op_experiments(n_node: int = 20, dry_run: bool = False, only_test: bool = False):
    """
    Run OP experiments.

    Args:
        n_node: Problem size
        dry_run: Dry run mode
        only_test: Test only mode
    """
    print(f"\n=== OP Experiments (n={n_node}) ===")

    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(OP_CONFIG)
    base_cfg["n_node"] = n_node

    results = load_progress()
    csv_path = RESULTS_DIR / "op_experiments.csv"
    fieldnames = ["problem", "n_node", "avg_cost", "best_cost", "aco_cost", "timestamp"]

    key = f"op_n{n_node}"

    if key in results and not dry_run:
        print(f"[SKIP] {key} already completed")
        return

    # Train model
    print(f"\n--- Training OP (n={n_node}) ---")
    model_path = train_model(base_cfg, dry_run, only_test=only_test)

    if model_path is None:
        print(f"[SKIP] Model not found for OP")
        return

    # Test model
    print(f"\n--- Testing OP (n={n_node}) ---")
    metrics = test_model(model_path, base_cfg, dry_run)

    row = {
        "problem": "op",
        "n_node": n_node,
        "avg_cost": metrics.get("avg_cost", ""),
        "best_cost": metrics.get("best_cost", ""),
        "aco_cost": metrics.get("aco_cost", ""),
        "timestamp": datetime.now().isoformat(),
    }
    write_csv_rows(csv_path, fieldnames, [row])
    save_progress(key, metrics)

    print(f"\n[RESULTS] OP experiments saved to {csv_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Problem Experiments - Comprehensive experiment pipeline"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "bpp", "mkp", "op"],
        default="all",
        help="Which phase to run"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal steps to verify pipeline")
    parser.add_argument("--test", action="store_true", help="Only test existing models, do not train")
    parser.add_argument("--n_node", type=int, default=100, help="Problem size for experiments")
    parser.add_argument("--test-debug", action="store_true", help="Run with minimal parameters for quick end-to-end testing")

    args = parser.parse_args()

    # Apply debug parameters if requested
    if args.test_debug:
        print("[DEBUG] Applying minimal parameters for testing...")
        args.n_node = 20
        DEFAULT_CONFIG.update({
            "H": 5,
            "mini_H": 5,
            "n_ants": 5,
            "k_sparse": 5,
            "epochs": 2,
            "steps_per_epoch": 2,
            "val_size": 1,
            "T": 3,
        })

    print("=" * 80)
    print("Cross-Problem Experiments Pipeline")
    print("=" * 80)
    print(f"Phase: {args.phase}")
    print(f"Problem size: {args.n_node}")
    print(f"Dry run: {args.dry_run}")
    print(f"Test only: {args.test}")
    print("=" * 80)

    # Run the requested phase(s)
    if args.phase == "all":
        run_bpp_experiments(args.n_node, args.dry_run, args.test)
        run_mkp_experiments(args.n_node, args.dry_run, args.test)
        run_op_experiments(args.n_node, args.dry_run, args.test)

    elif args.phase == "bpp":
        run_bpp_experiments(args.n_node, args.dry_run, args.test)

    elif args.phase == "mkp":
        run_mkp_experiments(args.n_node, args.dry_run, args.test)

    elif args.phase == "op":
        run_op_experiments(args.n_node, args.dry_run, args.test)

    print("\n" + "=" * 80)
    print("Experiments completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Progress file: {PROGRESS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
