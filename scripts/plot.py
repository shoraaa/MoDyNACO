#!/usr/bin/env python3
"""
Generate plots for experiment results.

Usage:
    python scripts/plot.py --experiment experiments/tsp_n1000_ppo_20250420 --type train_iters
    python scripts/plot.py --type compare --pattern "experiments/tsp_*"
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curve(exp_dir: Path, output_path: Path = None):
    """Plot training metrics over iterations."""
    log_dir = exp_dir / "logs"
    csv_path = log_dir / "train_metrics.csv"
    
    if not csv_path.exists():
        print(f"No training metrics found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training Progress: {exp_dir.name}")
    
    # Plot reward/cost over iterations
    if 'reward' in df.columns:
        axes[0, 0].plot(df['iteration'], df['reward'])
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward')
    
    if 'cost' in df.columns:
        axes[0, 1].plot(df['iteration'], df['cost'])
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].set_title('Cost')
    
    if 'loss' in df.columns:
        axes[1, 0].plot(df['iteration'], df['loss'])
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss')
    
    if 'grad_norm' in df.columns:
        axes[1, 1].plot(df['iteration'], df['grad_norm'])
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Norm')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(pattern: str = "experiments/*", output_path: Path = None):
    """Create bar chart comparing experiments."""
    import json
    
    from glob import glob
    exp_dirs = [Path(p) for p in glob(pattern) if Path(p).is_dir()]
    
    data = []
    for exp_dir in exp_dirs:
        results_path = exp_dir / "results" / "final.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            data.append({
                'experiment': exp_dir.name,
                'problem': results.get('problem', ''),
                'test_cost_mean': results.get('eval', {}).get('test_cost_mean', None),
                'test_cost_std': results.get('eval', {}).get('test_cost_std', None),
            })
    
    if not data:
        print("No results found.")
        return
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=['test_cost_mean'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['experiment'], df['test_cost_mean'], yerr=df['test_cost_std'], capsize=5)
    ax.set_ylabel('Test Cost')
    ax.set_title('Experiment Comparison')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None,
                       help="Experiment directory for single-plot")
    parser.add_argument("--type", type=str, choices=['train_iters', 'compare'],
                       default='train_iters', help="Plot type")
    parser.add_argument("--pattern", type=str, default="experiments/*",
                       help="Glob pattern for compare mode")
    parser.add_argument("--output", type=str, default=None,
                       help="Output image path (default: experiments/{exp}/plots/{type}.png)")
    args = parser.parse_args()
    
    sns.set_theme(style="whitegrid")
    
    if args.type == 'train_iters':
        if not args.experiment:
            parser.error("--experiment required for train_iters type")
        exp_dir = Path(args.experiment)
        
        if args.output:
            output_path = Path(args.output)
        else:
            plots_dir = exp_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            output_path = plots_dir / "training_curve.png"
        
        plot_training_curve(exp_dir, output_path)
    
    elif args.type == 'compare':
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path("experiments/comparison.png")
        
        plot_comparison(args.pattern, output_path)


if __name__ == "__main__":
    main()
