#!/usr/bin/env python3
"""
Compare experiments and generate comparison tables.

Usage:
    python scripts/compare.py --pattern "experiments/tsp_*"
"""

import argparse
import json
from pathlib import Path
import csv
from collections import defaultdict


def load_results(exp_dir: Path):
    """Load final results JSON for an experiment."""
    results_path = exp_dir / "results" / "final.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def scan_experiments(pattern: str = "experiments/*"):
    """Scan experiments matching pattern."""
    from glob import glob
    exp_dirs = []
    for pattern_match in glob(pattern):
        p = Path(pattern_match)
        if p.is_dir():
            exp_dirs.append(p)
    return sorted(exp_dirs)


def collect_results(exp_dirs):
    """Collect results from all experiments into a table."""
    rows = []
    
    for exp_dir in exp_dirs:
        results = load_results(exp_dir)
        if results is None:
            print(f"Warning: No results found for {exp_dir.name}")
            continue
        
        # Load config for hyperparams
        config_path = exp_dir / "config.yaml"
        config = {}
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
        # Build row
        row = {
            'experiment': exp_dir.name,
            'problem': results.get('problem', ''),
            'n_node': results.get('n_node', ''),
            'test_cost_mean': results.get('eval', {}).get('test_cost_mean', ''),
            'test_cost_std': results.get('eval', {}).get('test_cost_std', ''),
            'best_cost_mean': results.get('eval', {}).get('best_cost_mean', ''),
            'aco_gap_pct_mean': results.get('eval', {}).get('aco_improvement_pct_mean', ''),
            'static_gap_pct_mean': results.get('eval', {}).get('static_gap_pct_mean', ''),
            'train_time_hrs': results.get('train', {}).get('total_time_hrs', ''),
            'git_commit': results.get('config', {}).get('git_commit', ''),
        }
        
        # Add hyperparameters from config
        for key in ['lr', 'rho', 'H', 'mini_H', 'n_ants', 'k_sparse', 
                    'algo', 'seed', 'epochs', 'ablation_pheromone_features',
                    'ablation_incumbent_features', 'no_logit_net']:
            if key in config:
                row[f'config_{key}'] = config[key]
        
        rows.append(row)
    
    return rows


def write_csv(rows, output_path: Path):
    """Write results table to CSV."""
    if not rows:
        print("No results to write.")
        return
    
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} rows to {output_path}")


def write_latex(rows, output_path: Path):
    """Generate LaTeX comparison table."""
    if not rows:
        return
    
    # Group by problem
    problems = sorted(set(r['problem'] for r in rows if r['problem']))
    
    latex = "% Auto-generated comparison table\n"
    latex += "\\begin{table}[ht]\n"
    latex += "  \\centering\n"
    latex += "  \\begin{tabular}{l" + "c" * len(problems) + "}\n"
    latex += "    \\toprule\n"
    
    # Header row with problem names
    latex += "    Model & " + " & ".join(p.upper() for p in problems) + " \\\\\n"
    latex += "    \\midrule\n"
    
    # For each experiment, output its cost/std for each problem
    for row in rows:
        label = row['experiment']
        values = []
        for prob in problems:
            if row['problem'] == prob:
                mean = row.get('test_cost_mean', '')
                std = row.get('test_cost_std', '')
                if mean and std:
                    values.append(f"{mean:.4f}$\\pm${std:.4f}")
                elif mean:
                    values.append(str(mean))
                else:
                    values.append("-")
            else:
                values.append("-")
        latex += f"    {label} & {' & '.join(values)} \\\\\n"
    
    latex += "    \\bottomrule\n"
    latex += "  \\end{tabular}\n"
    latex += "  \\caption{Comparison of DyNACO variants across problem types.}\n"
    latex += "  \\label{tab:comparison}\n"
    latex += "\\end{table}\n"
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Wrote LaTeX table to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="experiments/*",
                       help="Glob pattern for experiment directories")
    parser.add_argument("--output", type=str, default="comparison",
                       help="Output basename (without extension)")
    args = parser.parse_args()
    
    exp_dirs = scan_experiments(args.pattern)
    print(f"Found {len(exp_dirs)} experiments")
    
    rows = collect_results(exp_dirs)
    print(f"Collected results from {len(rows)} experiments")
    
    # Write CSV
    csv_path = Path(f"{args.output}.csv")
    write_csv(rows, csv_path)
    
    # Write LaTeX
    tex_path = Path(f"{args.output}.tex")
    write_latex(rows, tex_path)
    
    # Write markdown
    md_path = Path(f"{args.output}.md")
    with open(md_path, 'w') as f:
        f.write("# Comparison Results\n\n")
        if rows:
            # Simple markdown table
            headers = ['experiment', 'problem', 'n_node', 'test_cost_mean', 
                      'test_cost_std', 'aco_gap_pct_mean']
            f.write("| " + " | ".join(h.replace('_', ' ') for h in headers) + " |\n")
            f.write("| " + " | ".join(['---']*len(headers)) + " |\n")
            for row in rows:
                values = [str(row.get(h, '')) for h in headers]
                f.write("| " + " | ".join(values) + " |\n")
    print(f"Wrote markdown to {md_path}")


if __name__ == "__main__":
    main()
