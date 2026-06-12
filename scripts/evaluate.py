#!/usr/bin/env python3
"""
Evaluate a trained experiment.

Usage:
    python scripts/evaluate.py --experiment experiments/tsp_n1000_ppo_20250420
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(exp_dir: Path) -> dict:
    """Load experiment configuration."""
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_checkpoint(exp_dir: Path) -> Path:
    """Find the best checkpoint."""
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Prefer best.pt, then last.pt, then any .pt
    candidates = ["best.pt", "last.pt"]
    for name in candidates:
        # Search recursively
        matches = list(ckpt_dir.rglob(name))
        if matches:
            return matches[0]

    # Find any .pt file recursively
    pt_files = list(ckpt_dir.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint files in {ckpt_dir}")

    # Pick the one with "best" in name, or the first
    for f in pt_files:
        if "best" in f.name:
            return f
    return pt_files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment directory path")
    parser.add_argument("--test-size", type=int, default=16,
                       help="Number of test instances")
    parser.add_argument("--no-compare", action="store_true",
                       help="Skip comparison with baselines")
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        sys.exit(1)

    # Load experiment config
    try:
        config = load_config(exp_dir)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Find checkpoint
    try:
        checkpoint = find_checkpoint(exp_dir)
        print(f"Using checkpoint: {checkpoint}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Build test command
    result_dir = exp_dir / "results"
    result_dir.mkdir(exist_ok=True)

    # Determine problem type
    problem = config['problem']

    # Base command
    cmd = ["uv", "run", "test.py"]

    # Add problem-specific arguments
    if problem in ['tsp', 'cvrp']:
        cmd.extend([
            f"--problem={problem}",
            f"--n_node={config['n_node']}",
        ])
    else:  # extended problems
        cmd.extend([
            f"--problem={problem}",
            f"--n_node={config['n_node']}",
        ])
        # Add extended problem-specific parameters
        if problem == 'bpp':
            if 'capacity' in config:
                cmd.append(f"--capacity={config['capacity']}")
        elif problem == 'mkp':
            if 'm' in config:
                cmd.append(f"--m={config['m']}")
        elif problem == 'op':
            if 'max_len' in config:
                cmd.append(f"--max_len={config['max_len']}")

    # Common arguments
    cmd.extend([
        f"--checkpoint={checkpoint}",
        f"--H={config.get('H', 10 if problem in ['tsp', 'cvrp'] else 5)}",
        f"--mini_H={config.get('mini_H', 100 if problem in ['tsp', 'cvrp'] else 5)}",
        f"--n_ants={config.get('n_ants', 100 if problem in ['tsp', 'cvrp'] else 20)}",
        f"--k_sparse={config.get('k_sparse', 32)}",
        f"--rho={config.get('rho', 0.1 if problem in ['tsp', 'cvrp'] else 0.9)}",
    ])

    # Add seed if present
    if 'seed' in config:
        cmd.append(f"--seed={config['seed']}")

    # Add device if specified
    if 'device' in config:
        cmd.append(f"--device={config['device']}")

    # Result saving - different flags for base vs extended problems
    if problem in ['tsp', 'cvrp']:
        # Base problems use --summary_json
        summary_path = result_dir / f"{problem}_n{config['n_node']}_summary.json"
        cmd.extend([
            f"--summary_json={summary_path}",
        ])
    else:
        # Extended problems use --test_size, --save_results, --save_dir
        cmd.extend([
            f"--test_size={args.test_size}",
            f"--save_results",
            f"--save_dir={result_dir}",
        ])

    # Skip baselines if requested
    if args.no_compare:
        cmd.append("--no_baselines" if problem in ['tsp', 'cvrp'] else "--no_baseline")

    print(f"Running evaluation...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run evaluation
    process = subprocess.run(cmd, capture_output=False, text=True)

    if process.returncode != 0:
        print(f"Evaluation failed with exit code {process.returncode}")
        sys.exit(process.returncode)

    # The test.py should have saved results
    if problem in ['tsp', 'cvrp']:
        expected_results = result_dir / f"{problem}_n{config['n_node']}_summary.json"
        final_results = result_dir / "final.json"
        if expected_results.exists():
            import shutil
            shutil.copy2(expected_results, final_results)
            print(f"\nResults saved to {final_results}")
        else:
            print(f"\nWarning: Expected results file not found at {expected_results}")
            print("Check test.py output for errors.")
    else:
        expected_results = result_dir / f"{problem}_n{config['n_node']}_results.json"
        final_results = result_dir / "final.json"
        if expected_results.exists():
            import shutil
            shutil.copy2(expected_results, final_results)
            print(f"\nResults saved to {final_results}")
        else:
            print(f"\nWarning: Expected results file not found at {expected_results}")
            print("Check test.py output for errors.")

    print(f"\nEvaluation complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
