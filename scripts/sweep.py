#!/usr/bin/env python3
"""
Hyperparameter sweep runner.

Usage:
    python scripts/sweep.py --config configs/tsp_n1000_ppo.yaml --params "lr:0.0001,0.0005;rho:0.1,0.5"
"""

import argparse
import itertools
from pathlib import Path
import yaml
import subprocess
import sys


def parse_params(param_str: str) -> dict:
    """Parse parameter specification string.
    
    Format: "lr:0.0001,0.0005;rho:0.1,0.5"
    Returns: {'lr': [0.0001, 0.0005], 'rho': [0.1, 0.5]}
    """
    params = {}
    for pair in param_str.split(';'):
        if ':' not in pair:
            continue
        key, values = pair.split(':', 1)
        values_list = [v.strip() for v in values.split(',')]
        # Try to convert to float/int if numeric
        converted = []
        for v in values_list:
            try:
                if '.' in v:
                    converted.append(float(v))
                else:
                    converted.append(int(v))
            except ValueError:
                converted.append(v)
        params[key.strip()] = converted
    return params


def generate_grid(base_config: dict, param_grid: dict):
    """Generate all combinations of parameters."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    for combo in itertools.product(*values):
        new_config = base_config.copy()
        param_dict = dict(zip(keys, combo))
        new_config.update(param_dict)
        yield param_dict, new_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Base configuration YAML")
    parser.add_argument("--params", type=str, required=True,
                       help="Parameters to sweep (format: 'lr:0.1,0.01;rho:0.1,0.5')")
    parser.add_argument("--dry-run", action="store_true",
                       help="Just print commands without executing")
    args = parser.parse_args()
    
    # Load base config
    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    
    # Parse sweep parameters
    param_grid = parse_params(args.params)
    if not param_grid:
        print("Error: No valid parameters to sweep")
        sys.exit(1)
    
    print(f"Base config: {args.config}")
    print(f"Sweeping over: {param_grid}")
    print(f"Total combinations: {itertools.prod(len(v) for v in param_grid.values())}")
    print()
    
    if args.dry_run:
        print("DRY RUN - commands will be printed, not executed\n")
    
    # Generate experiment name base
    base_name = Path(args.config).stem
    combo_num = 0
    
    for param_combo, modified_config in generate_grid(base_config, param_grid):
        combo_num += 1
        
        # Create experiment identifier
        param_parts = []
        for key, value in param_combo.items():
            param_parts.append(f"{key}{value}")
        exp_suffix = "_".join(param_parts)
        exp_name = f"{base_name}_sweep_{exp_suffix}"
        
        # Save modified config to experiments directory
        exp_dir = Path("experiments") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(modified_config, f, sort_keys=False)
        
        print(f"[{combo_num}] Experiment: {exp_name}")
        print(f"  Config: {config_path}")
        
        # Build command
        cmd = ["uv", "run", "scripts/train.py", "--config", str(config_path)]
        print(f"  Command: {' '.join(cmd)}")
        
        if not args.dry_run:
            # Optionally: could run in parallel or submit to job scheduler
            # For now, run sequentially
            print(f"  Running...")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  FAILED with exit code {result.returncode}")
                # Optionally continue or break
        else:
            print(f"  (dry-run, not executing)")
        
        print()
    
    print(f"Completed {combo_num} experiments.")


if __name__ == "__main__":
    main()
