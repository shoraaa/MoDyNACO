#!/usr/bin/env python3
"""
Training wrapper with YAML config support.

Usage:
    python scripts/train.py --config configs/train/tsp_n1000_ppo.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(yaml_path: str) -> dict:
    """Load YAML configuration."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def config_to_args(config: dict) -> list:
    """Convert config dict to command-line arguments, excluding metadata."""
    # Metadata and W&B fields that should NOT be passed to train.py
    EXCLUDE_KEYS = {
        'experiment_name', 'description', 'author', 'use_wandb',
        'wandb_project', 'wandb_entity', 'wandb_group', 'timestamp',
        'git_commit', 'git_diff_stat'
    }

    args = []
    for key, value in config.items():
        if key in EXCLUDE_KEYS:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            args.extend([f"--{key}"] + [str(v) for v in value])
        else:
            args.extend([f"--{key}", str(value)])
    return args


def create_experiment_dir(exp_name: str) -> Path:
    """Create experiment directory structure."""
    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    return exp_dir


def save_experiment_config(config: dict, exp_dir: Path):
    """Save experiment configuration."""
    config_path = exp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Saved config to {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--experiment-name", type=str, default=None, 
                       help="Override experiment directory name")
    parser.add_argument("--no-index", action="store_true",
                       help="Skip updating experiment index after training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Determine experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        stem = Path(args.config).stem
        timestamp = subprocess.check_output(['date', '+%Y%m%d_%H%M%S']).decode().strip()
        exp_name = f"{stem}_{timestamp}"
    
    # Create experiment directory
    exp_dir = create_experiment_dir(exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Save config
    save_experiment_config(config, exp_dir)
    
    # Set environment variable so train.py knows to save additional metadata
    os.environ["NGFACO_EXPERIMENT_DIR"] = str(exp_dir)
    
    # Build command-line arguments
    cli_args = config_to_args(config)

    # Add save_dir pointing to experiment's checkpoints folder
    cli_args.extend([f"--save_dir={exp_dir / 'checkpoints'}"])

    # Build command
    cmd = ["uv", "run", "train.py"] + cli_args
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Logs: {exp_dir / 'logs' / 'stdout.txt'}")
    
    # Run training with output tee'd to log file in real-time
    log_file = exp_dir / "logs" / "stdout.txt"
    with open(log_file, 'w') as log_f:
        # Use unbuffered output for real-time streaming
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   env=env)

        # Stream output in real-time
        for line in process.stdout:
            log_f.write(line)
            print(line, end='', flush=True)

        process.wait()
        returncode = process.returncode

    # Update experiment index in background (non-blocking, ignore errors)
    if not args.no_index:
        print("\nUpdating experiment index...")
        subprocess.Popen(["uv", "run", "scripts/indexExperiments.py", "--rebuild"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
