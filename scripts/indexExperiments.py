#!/usr/bin/env python3
"""
Generate master index of all experiments.

Scans experiments/*/results/final.json and creates experiments/index.csv
"""

import json
import argparse
from pathlib import Path
import csv
from datetime import datetime


def load_results(exp_dir: Path):
    """Load final results JSON for an experiment."""
    results_path = exp_dir / "results" / "final.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def scan_experiments(experiments_dir: Path = Path("experiments")):
    """Scan all experiments and yield result dicts."""
    if not experiments_dir.exists():
        return []
    
    rows = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        results = load_results(exp_dir)
        if results is None:
            continue
        
        # Extract config info
        config_path = exp_dir / "config.yaml"
        config_meta = {}
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                # Extract key hyperparameters for index
                for key in ['lr', 'rho', 'H', 'mini_H', 'n_ants', 'k_sparse', 
                           'algo', 'seed', 'epochs']:
                    if key in config:
                        config_meta[key] = config[key]
        
        row = {
            'experiment': exp_dir.name,
            'problem': results.get('problem', 'unknown'),
            'n_node': results.get('n_node', ''),
            'test_cost_mean': results.get('eval', {}).get('test_cost_mean', ''),
            'test_cost_std': results.get('eval', {}).get('test_cost_std', ''),
            'aco_improvement_pct_mean': results.get('eval', {}).get('aco_improvement_pct_mean', ''),
            'static_gap_pct_mean': results.get('eval', {}).get('static_gap_pct_mean', ''),
            'checkpoint': results.get('checkpoint', ''),
            'git_commit': results.get('config', {}).get('git_commit', ''),
            'timestamp': exp_dir.stat().st_mtime,
        }
        # Merge config metadata
        row.update(config_meta)
        rows.append(row)
    
    return rows


def write_index_csv(rows, output_path: Path = Path("experiments/index.csv")):
    """Write experiment index to CSV."""
    if not rows:
        print("No experiments found.")
        return
    
    # Determine fieldnames
    fieldnames = ['experiment', 'problem', 'n_node', 'test_cost_mean', 
                  'test_cost_std', 'aco_improvement_pct_mean', 
                  'static_gap_pct_mean', 'checkpoint', 'git_commit', 'timestamp']
    # Add any extra config fields
    extra_fields = set()
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                extra_fields.add(k)
    fieldnames.extend(sorted(extra_fields))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} experiments to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                       help="Rebuild index from scratch")
    parser.add_argument("--output", type=str, default="experiments/index.csv",
                       help="Output CSV path")
    args = parser.parse_args()
    
    rows = scan_experiments()
    write_index_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
