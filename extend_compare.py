#!/usr/bin/env python3
"""
Comparison script for Extended Problems (BPP, MKP, OP)
Comparing Static Prior (DeepACO style) vs Dynamic Prior (NGFACO style).
"""

import torch
import argparse
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import train_extended
import utils_extended
import faco_extended
import net_extended

def get_default_args(problem, n_node=50):
    """Get default arguments for a problem."""
    parser = argparse.ArgumentParser()
    # We'll manually populate a Namespace to avoid full parsing
    args = parser.parse_args([])
    
    args.problem = problem
    args.n_node = n_node
    args.m = 5
    args.capacity = 150.0
    args.max_len = 4.0 if n_node <= 100 else 6.0
    
    args.H = 5
    args.mini_H = 5
    args.n_ants = 20
    args.k_sparse = 32
    args.epochs = 5 # Reduced for quick comparison
    args.steps_per_epoch = 32 # Reduced for quick comparison
    args.val_size = 16
    args.aco_iters_per_sample = 20
    
    args.lr = 3e-4
    args.rho = 0.9
    args.alpha = 1.0
    args.beta = 1.0
    args.gamma = 1.0
    args.min_gamma = 0.0
    
    args.warmup = 0
    args.train_anneal = False
    args.no_dynamic_feats = False
    args.smallvram = False
    args.static_prior = False
    
    args.wandb_project = None # Disable wandb for comparison script
    args.no_wandb = True
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.seed = 1234
    args.save_dir = "checkpoints_compare"
    args.save_interval = 100 # Don't save intermediate
    args.dry_run = False
    args.run_name = None
    args.wandb_entity = None
    args.wandb_group = None
    
    return args

def run_experiment(args):
    """Run training and return best validation score."""
    print(f"\n>>> Running experiment: Problem={args.problem}, StaticPrior={args.static_prior}")
    
    # Ensure save dir exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # We call train_extended.train but we'll need to capture the results.
    # Since train_extended.train prints and saves to disk, we'll read the final result.
    model_name = train_extended.build_model_name(args)
    final_path = Path(args.save_dir) / f"{model_name}_best.pt"
    
    # Run training
    train_extended.train(args)
    
    # Load results from final checkpoint
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location='cpu')
        # We need the best validation score. 
        # Note: train_extended.train saves a list of val scores in the last epoch's checkpoint if we modify it, 
        # but here we'll just return what's available.
        # Actually, let's look at what's in 'best.pt'
        # It has 'model_state_dict' and 'config'.
        # We might need to run a final evaluation.
        return evaluate_model(args, final_path)
    else:
        print(f"Error: Final model not found at {final_path}")
        return None

def evaluate_model(args, checkpoint_path):
    """Evaluate a trained model on a fixed test set."""
    print(f"Evaluating model from {checkpoint_path}...")
    
    # Load model
    edge_feats = 1 if args.static_prior else 3
    if args.problem == 'bpp':
        model = net_extended.NetBPP(feats=1, edge_feats=edge_feats)
    elif args.problem == 'mkp':
        model = net_extended.NetMKP(m=args.m, feats=args.m + 1, edge_feats=edge_feats)
    elif args.problem == 'op':
        model = net_extended.NetOP(feats=2, edge_feats=edge_feats)
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Load test set (use utils_extended to get 100 instances)
    test_instances = []
    torch.manual_seed(9999) # Fixed seed for testing
    np.random.seed(9999)
    for _ in range(50): # 50 instances for stable evaluation
        test_instances.append(utils_extended.get_problem_data(
            args.problem, args.n_node, args.device, args.k_sparse
        ))
    
    results = []
    baseline_results = []
    
    dynamic_graph = train_extended.use_dynamic_edge_features(args)
    expected_edge_feats = train_extended.get_model_edge_feats(args)

    for instance_data in tqdm(test_instances, desc="Evaluating"):
        # 1. Evaluate with neural guidance
        # Extract raw data
        if args.problem == 'bpp':
            demand = instance_data['demand']
        elif args.problem == 'mkp':
            prize = instance_data['prize']
            weight = instance_data['weight']
        elif args.problem == 'op':
            distances = instance_data['distances']
            prizes = instance_data['prizes']

        # Setup ACO
        if args.problem == 'bpp':
            aco = faco_extended.MFACO_BPP(demand, capacity=args.capacity, n_ants=args.n_ants, device=args.device, decay=args.rho)
            pyg_args = (demand, args.device)
        elif args.problem == 'mkp':
            aco = faco_extended.MFACO_MKP(prize, weight, n_ants=args.n_ants, device=args.device, decay=args.rho)
            pyg_args = (prize, weight, args.device)
        elif args.problem == 'op':
            aco = faco_extended.MFACO_OP(distances, prizes, max_len=args.max_len, n_ants=args.n_ants, device=args.device, decay=args.rho)
            pyg_args = (distances, prizes, args.device)

        with torch.no_grad():
            if args.static_prior:
                # Static mode: compute prior once
                pyg_data = train_extended.build_pyg_data(aco, args.problem, *pyg_args, dynamic=False)
                # align_edge_attr_width will now handle 1 -> expected_edge_feats (1 or 3)
                pyg_data = train_extended.align_edge_attr_width(pyg_data, edge_feats)
                prior_output = model(pyg_data)
                prior = train_extended.reshape_prior_output(
                    prior_output, args.problem,
                    demand=demand if args.problem == 'bpp' else None,
                    prize=prize if args.problem == 'mkp' else None,
                    prizes=prizes if args.problem == 'op' else None,
                )
                # We update the heuristic here and keep it for all outer/inner iterations
                aco.heuristic = prior.to(device=args.device, dtype=torch.float32)
                aco._sync_cpp_inputs()

            # Run H outer iterations
            running_best = -float('inf')
            for outer in range(args.H):
                if not args.static_prior:
                    # Dynamic mode: recompute prior once per outer iteration
                    pyg_data = train_extended.build_pyg_data(aco, args.problem, *pyg_args, dynamic=dynamic_graph)
                    pyg_data = train_extended.align_edge_attr_width(pyg_data, edge_feats)
                    prior_output = model(pyg_data)
                    prior = train_extended.reshape_prior_output(
                        prior_output, args.problem,
                        demand=demand if args.problem == 'bpp' else None,
                        prize=prize if args.problem == 'mkp' else None,
                        prizes=prizes if args.problem == 'op' else None,
                    )
                
                for inner in range(args.mini_H):
                    current_prior = prior
                    if args.train_anneal:
                        factor = train_extended.compute_annealing_factor(inner, args.mini_H, args.gamma, args.min_gamma)
                        current_prior = prior * factor
                    
                    aco.heuristic = current_prior.to(device=args.device, dtype=torch.float32)
                    aco._sync_cpp_inputs()

                    # Sample
                    costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
                    
                    costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
                    objective_t = train_extended.raw_values_to_objective(costs_t, args.problem)
                    
                    running_best = max(running_best, float(objective_t.max().item()))
                    
                    # Update pheromone with the full batch
                    aco.update_pheromone(paths, costs)

            # Record final best from last sample
            results.append(running_best)
            
        # 2. Baseline ACO (no neural guidance)
        aco_base, _ = train_extended.setup_aco(args, instance_data, args.problem)
        running_best_base = -float('inf')
        # Run H * mini_H iterations of baseline ACO for fair comparison
        for _ in range(args.H * args.mini_H):
            costs_base, paths_base, _, _ = aco_base.sample(require_prob=False)
            
            # Update pheromone with the full batch
            aco_base.update_pheromone(paths_base, costs_base)
            
            costs_base_t = torch.as_tensor(costs_base, device=args.device, dtype=torch.float32)
            objective_base_t = train_extended.raw_values_to_objective(costs_base_t, args.problem)
            running_best_base = max(running_best_base, float(objective_base_t.max().item()))
            
        # Final objective from baseline
        baseline_results.append(running_best_base)



    avg_score = np.mean(results)
    avg_baseline = np.mean(baseline_results)
    improvement = (avg_score - avg_baseline) / abs(avg_baseline) * 100 if abs(avg_baseline) > 1e-10 else 0
    
    print(f"Average Score: {avg_score:.4f} (Baseline: {avg_baseline:.4f}, Improvement: {improvement:.2f}%)")
    return {
        'avg_score': avg_score,
        'avg_baseline': avg_baseline,
        'improvement': improvement,
        'std': np.std(results)
    }

def main():
    parser = argparse.ArgumentParser(description="Compare Static vs Dynamic Prior on Extended Problems")
    parser.add_argument("--problems", type=str, default="bpp,mkp,op", help="Comma-separated list of problems")
    parser.add_argument("--n_node", type=int, default=50, help="Problem size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=32, help="Steps per epoch")
    parser.add_argument("--device", type=str, default=None, help="Device")
    
    args_cli = parser.parse_args()
    problems = args_cli.problems.split(",")
    
    final_results = []
    
    for prob in problems:
        prob = prob.strip()
        print(f"\n{'='*60}\nPROBLEM: {prob.upper()}\n{'='*60}")
        
        # 1. Static Prior (DeepACO style)
        args_static = get_default_args(prob, args_cli.n_node)
        args_static.static_prior = True
        args_static.epochs = args_cli.epochs
        args_static.steps_per_epoch = args_cli.steps
        if args_cli.device: args_static.device = args_cli.device
        
        res_static = run_experiment(args_static)
        
        # 2. Dynamic Prior (NGFACO style)
        args_dynamic = get_default_args(prob, args_cli.n_node)
        args_dynamic.static_prior = False
        args_dynamic.epochs = args_cli.epochs
        args_dynamic.steps_per_epoch = args_cli.steps
        if args_cli.device: args_dynamic.device = args_cli.device
        
        res_dynamic = run_experiment(args_dynamic)
        
        final_results.append({
            'Problem': prob,
            'Static_Score': res_static['avg_score'],
            'Static_Imp%': res_static['improvement'],
            'Dynamic_Score': res_dynamic['avg_score'],
            'Dynamic_Imp%': res_dynamic['improvement'],
            'Gain_over_Static%': (res_dynamic['avg_score'] - res_static['avg_score']) / abs(res_static['avg_score']) * 100 if abs(res_static['avg_score']) > 1e-10 else 0
        })

    # Display summary table
    df = pd.DataFrame(final_results)
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save results
    with open("comparison_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Results saved to comparison_results.json")

if __name__ == "__main__":
    main()
