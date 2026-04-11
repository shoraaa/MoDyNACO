#!/usr/bin/env python3
"""
Testing script for Neural-Guided Fast ACO (NGFACO) - Extended for BPP, MKP, OP.

This module implements testing for trained neural priors that guide
ant colony optimization for BPP, MKP, and OP problems.
"""

import time
import torch
import argparse
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List, Any, Tuple

# Extended modules
import net_extended
import faco_extended
import utils_extended

# Specific class imports
from net_extended import NetBPP, NetMKP, NetOP
from faco_extended import MFACO_BPP, MFACO_MKP, MFACO_OP


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

EPS = 1e-10


def run_aco_rollout(aco: Any, problem_type: str, n_iterations: int) -> float:
    """Run an ACO solver and normalize the return value to a scalar cost/objective."""
    if problem_type == 'bpp':
        return float(aco.run(n_iterations))
    if problem_type in {'mkp', 'op'}:
        best_value, _ = aco.run(n_iterations)
        return float(best_value)
    raise ValueError(f"Unknown problem type: {problem_type}")


def is_maximization_problem(problem_type: str) -> bool:
    """Return whether larger objective values are better for the problem."""
    return problem_type in {'bpp', 'mkp', 'op'}


def raw_values_to_objective(values: torch.Tensor, problem_type: str) -> torch.Tensor:
    """Convert raw solver outputs into a higher-is-better objective tensor."""
    if problem_type == 'bpp':
        return -values
    if problem_type in {'mkp', 'op'}:
        return values
    raise ValueError(f"Unknown problem type: {problem_type}")


def select_best_value(values: torch.Tensor, problem_type: str) -> float:
    """Select the best value using the problem's optimization direction."""
    objective = raw_values_to_objective(values, problem_type)
    return float(objective.max().item())


def compute_relative_improvement(candidate: float, baseline: float, problem_type: str) -> Optional[float]:
    """Return percent improvement, positive when candidate is better than baseline."""
    if abs(baseline) <= EPS:
        return None
    if is_maximization_problem(problem_type):
        return (candidate - baseline) / abs(baseline) * 100.0
    return (baseline - candidate) / abs(baseline) * 100.0


def align_edge_attr_width(pyg_data: Any, expected_edge_feats: int) -> Any:
    """Slice or zero-pad edge attributes to match the checkpoint's expected width."""
    current_edge_feats = pyg_data.edge_attr.shape[1]
    if current_edge_feats == expected_edge_feats:
        return pyg_data
    if current_edge_feats > expected_edge_feats:
        pyg_data.edge_attr = pyg_data.edge_attr[:, :expected_edge_feats]
        return pyg_data

    pad = torch.zeros(
        (pyg_data.edge_attr.shape[0], expected_edge_feats - current_edge_feats),
        device=pyg_data.edge_attr.device,
        dtype=pyg_data.edge_attr.dtype,
    )
    pyg_data.edge_attr = torch.cat((pyg_data.edge_attr, pad), dim=1)
    return pyg_data


def reshape_prior_output(
    prior_output: torch.Tensor,
    problem_type: str,
    demand: Optional[torch.Tensor] = None,
    prize: Optional[torch.Tensor] = None,
    prizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reshape a flat model output into the dense prior matrix for the problem."""
    if problem_type == 'bpp':
        if demand is None:
            raise ValueError("demand is required for BPP prior reshaping")
        n = len(demand) - 1  # Exclude depot
    elif problem_type == 'mkp':
        if prize is None:
            raise ValueError("prize is required for MKP prior reshaping")
        n = len(prize)
    elif problem_type == 'op':
        if prizes is None:
            raise ValueError("prizes is required for OP prior reshaping")
        n = len(prizes)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return prior_output.view(n + 1, n + 1)


def setup_aco(
    args: argparse.Namespace,
    instance_data: Any,
    problem_type: str
) -> Tuple[Any, Tuple]:
    """
    Setup ACO solver for the given problem instance.

    Args:
        args: Testing arguments
        instance_data: Problem instance data (can be dict or raw data)
        problem_type: 'bpp', 'mkp', or 'op'

    Returns:
        Tuple of (aco_solver, pyg_args)
    """
    # Extract raw data if instance_data is a dictionary
    if isinstance(instance_data, dict):
        if problem_type == 'bpp':
            demand = instance_data['demand']
        elif problem_type == 'mkp':
            prize = instance_data['prize']
            weight = instance_data['weight']
        elif problem_type == 'op':
            distances = instance_data['distances']
            prizes = instance_data['prizes']
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    else:
        # Use raw data directly
        if problem_type == 'bpp':
            demand = instance_data
        elif problem_type == 'mkp':
            prize, weight = instance_data
        elif problem_type == 'op':
            distances, prizes = instance_data
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    if problem_type == 'bpp':
        kwargs = {
            'demand': demand,
            'capacity': args.capacity,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,  # Use False like DeepACO
            'heuristic': None,  # Will be set after generating prior
            'device': args.device,
        }
        pyg_args = (demand, args.device)
        aco = faco_extended.MFACO_BPP(**kwargs)

    elif problem_type == 'mkp':
        kwargs = {
            'prize': prize,
            'weight': weight,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,  # Use False like DeepACO
            'heuristic': None,  # Will be set after generating prior
            'device': args.device,
        }
        pyg_args = (prize, weight, args.device)
        aco = faco_extended.MFACO_MKP(**kwargs)

    elif problem_type == 'op':
        kwargs = {
            'distances': distances,
            'prizes': prizes,
            'max_len': args.max_len,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,  # Use False like DeepACO
            'heuristic': None,  # Will be set after generating prior
            'device': args.device,
        }
        pyg_args = (distances, prizes, args.device)
        aco = faco_extended.MFACO_OP(**kwargs)

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return aco, pyg_args


def build_pyg_data(
    aco: Any,
    *args,
    dynamic: bool = True
):
    """Build PyG data for the given problem."""
    problem_type = args[0]  # First arg is problem type

    if problem_type == 'bpp':
        demand = args[1]
        device = args[2]
        return utils_extended.build_pyg_data_bpp(
            demand,
            device,
            pheromone=aco.pheromone,
            dynamic=dynamic,
        ).to(device)

    elif problem_type == 'mkp':
        prize, weight = args[1], args[2]
        device = args[3]
        return utils_extended.build_pyg_data_mkp(
            prize,
            weight,
            device,
            pheromone=aco.pheromone,
            dynamic=dynamic,
        ).to(device)

    elif problem_type == 'op':
        distances, prizes = args[1], args[2]
        device = args[3]
        return utils_extended.build_pyg_data_op_dense(
            distances,
            prizes,
            device,
            pheromone=aco.pheromone,
            dynamic=dynamic,
        ).to(device)

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_instance(
    model: Any,
    instance_data: Any,
    args: argparse.Namespace,
    expected_edge_feats: int,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Test on a single instance.

    Args:
        model: Neural network model
        instance_data: Problem instance data
        args: Testing arguments

    Returns:
        Tuple of (avg_cost, best_cost, metrics_dict)
    """
    model.eval()

    # Extract raw data
    if isinstance(instance_data, dict):
        if args.problem == 'bpp':
            demand = instance_data['demand']
        elif args.problem == 'mkp':
            prize = instance_data['prize']
            weight = instance_data['weight']
        elif args.problem == 'op':
            distances = instance_data['distances']
            prizes = instance_data['prizes']
        else:
            raise ValueError(f"Unknown problem type: {args.problem}")
    else:
        # Use raw data directly
        if args.problem == 'bpp':
            demand = instance_data
        elif args.problem == 'mkp':
            prize, weight = instance_data
        elif args.problem == 'op':
            distances, prizes = instance_data
        else:
            raise ValueError(f"Unknown problem type: {args.problem}")

    # Build the initial PyG data. Dynamic mode will refresh this from the live ACO state.
    if args.problem == 'bpp':
        initial_pyg_data = utils_extended.build_pyg_data_bpp(demand, args.device)
    elif args.problem == 'mkp':
        initial_pyg_data = utils_extended.build_pyg_data_mkp(prize, weight, args.device)
    elif args.problem == 'op':
        initial_pyg_data = utils_extended.build_pyg_data_op_dense(distances, prizes, args.device)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")
    initial_pyg_data = align_edge_attr_width(initial_pyg_data, expected_edge_feats)

    # Create ACO solver without heuristic initially (will be updated each outer iteration)
    if args.problem == 'bpp':
        kwargs = {
            'demand': demand,
            'capacity': args.capacity,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,  # Use False like DeepACO
            'heuristic': None,  # Will be updated each outer iteration
            'device': args.device,
        }
        aco = faco_extended.MFACO_BPP(**kwargs)

    elif args.problem == 'mkp':
        kwargs = {
            'prize': prize,
            'weight': weight,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,  # Use False like DeepACO
            'heuristic': None,  # Will be updated each outer iteration
            'device': args.device,
        }
        aco = faco_extended.MFACO_MKP(**kwargs)

    elif args.problem == 'op':
        kwargs = {
            'distances': distances,
            'prizes': prizes,
            'max_len': args.max_len,
            'n_ants': args.n_ants,
            'decay': args.rho,
            'alpha': args.alpha,
            'beta': args.beta,
            'elitist': False,  # Use False like DeepACO
            'heuristic': None,  # Will be updated each outer iteration
            'device': args.device,
        }
        aco = faco_extended.MFACO_OP(**kwargs)

    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    if args.problem == 'bpp':
        pyg_args = (demand, args.device)
    elif args.problem == 'mkp':
        pyg_args = (prize, weight, args.device)
    elif args.problem == 'op':
        pyg_args = (distances, prizes, args.device)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    static_aco_cost = None
    static_improvement_pct = None
    if args.static_compare:
        static_aco, _ = setup_aco(args, instance_data, args.problem)
        with torch.no_grad():
            static_prior_output = model(initial_pyg_data)
            static_prior = reshape_prior_output(
                static_prior_output,
                args.problem,
                demand=demand if args.problem == 'bpp' else None,
                prize=prize if args.problem == 'mkp' else None,
                prizes=prizes if args.problem == 'op' else None,
            )
            static_aco.heuristic = static_prior.to(device=args.device, dtype=torch.float32)
            static_aco._sync_cpp_inputs()

            for outer in range(args.H):
                static_costs, _, _, _ = static_aco.sample(require_prob=False, prior=None, parallel_traced=True)
                static_aco.run(1)

        static_aco_cost = run_aco_rollout(static_aco, args.problem, args.H * args.mini_H)

    # Run H outer iterations, regenerating the prior from the current ACO state.
    with torch.no_grad():
        for outer in range(args.H):
            if args.no_dynamic_feats:
                pyg_data = initial_pyg_data
            elif args.problem == 'bpp':
                pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=True)
            elif args.problem == 'mkp':
                pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=True)
            elif args.problem == 'op':
                pyg_data = build_pyg_data(aco, args.problem, *pyg_args, dynamic=True)
            else:
                raise ValueError(f"Unknown problem type: {args.problem}")
            pyg_data = align_edge_attr_width(pyg_data, expected_edge_feats)

            # Generate neural prior for this outer iteration.
            prior_output = model(pyg_data)
            prior = reshape_prior_output(
                prior_output,
                args.problem,
                demand=demand if args.problem == 'bpp' else None,
                prize=prize if args.problem == 'mkp' else None,
                prizes=prizes if args.problem == 'op' else None,
            )

            # Update ACO solver's heuristic with new neural prior
            aco.heuristic = prior.to(device=args.device, dtype=torch.float32)
            aco._sync_cpp_inputs()

            # Sample with updated heuristic
            costs, paths, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)

            # Run ACO for pheromone update
            aco.run(1)

    # Get final sample cost
    costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
    objective_t = raw_values_to_objective(costs_t, args.problem)

    avg_cost = float(objective_t.mean().item())
    best_cost = select_best_value(costs_t, args.problem)

    # Run guided ACO rollout after the outer prior-refresh loop.
    best_aco_cost = run_aco_rollout(aco, args.problem, args.H * args.mini_H)

    pure_aco_cost = None
    aco_improvement_pct = None
    if not args.no_baselines:
        pure_aco, _ = setup_aco(args, instance_data, args.problem)
        pure_aco_cost = run_aco_rollout(pure_aco, args.problem, args.H * (args.mini_H + 1))
        aco_improvement_pct = compute_relative_improvement(best_aco_cost, pure_aco_cost, args.problem)
    if static_aco_cost is not None:
        static_improvement_pct = compute_relative_improvement(best_aco_cost, static_aco_cost, args.problem)

    metrics = {
        "avg_cost": avg_cost,
        "best_cost": best_cost,
        "best_aco_cost": best_aco_cost,
        "pure_aco_cost": pure_aco_cost,
        "aco_improvement_pct": aco_improvement_pct,
        "static_aco_cost": static_aco_cost,
        "static_improvement_pct": static_improvement_pct,
    }

    return avg_cost, best_cost, metrics


# =============================================================================
# MAIN TESTING LOOP
# =============================================================================

def test(args: argparse.Namespace):
    """Main testing loop."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    feats = state_dict["emb_net.v_lin0.weight"].shape[1]
    edge_feats = state_dict["emb_net.e_lin0.weight"].shape[1]

    # Create model
    if args.problem == 'bpp':
        model = NetBPP(feats=feats, edge_feats=edge_feats)
    elif args.problem == 'mkp':
        model = NetMKP(m=args.m, feats=feats, edge_feats=edge_feats)
    elif args.problem == 'op':
        model = NetOP(feats=feats, edge_feats=edge_feats)
    else:
        raise ValueError(f"Unknown problem: {args.problem}")

    model = model.to(args.device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")

    # Testing loop
    test_instances = utils_extended.load_test_dataset(
        args.problem, args.n_node, args.device, args.k_sparse
    )
    if args.test_size is not None:
        test_instances = list(test_instances[:args.test_size])

    all_avg_costs = []
    all_best_costs = []
    all_aco_costs = []
    all_pure_aco_costs = []
    all_aco_improvements = []
    all_static_aco_costs = []
    all_static_improvements = []

    for instance_data in tqdm(test_instances, desc="Testing"):
        # Test on instance
        avg_cost, best_cost, metrics = test_instance(model, instance_data, args, edge_feats)

        all_avg_costs.append(avg_cost)
        all_best_costs.append(best_cost)
        all_aco_costs.append(metrics["best_aco_cost"])
        if metrics["pure_aco_cost"] is not None:
            all_pure_aco_costs.append(metrics["pure_aco_cost"])
        if metrics["aco_improvement_pct"] is not None:
            all_aco_improvements.append(metrics["aco_improvement_pct"])
        if metrics["static_aco_cost"] is not None:
            all_static_aco_costs.append(metrics["static_aco_cost"])
        if metrics["static_improvement_pct"] is not None:
            all_static_improvements.append(metrics["static_improvement_pct"])

    # Compute statistics
    avg_cost_mean = np.mean(all_avg_costs)
    avg_cost_std = np.std(all_avg_costs)
    best_cost_mean = np.mean(all_best_costs)
    best_cost_std = np.std(all_best_costs)
    aco_cost_mean = np.mean(all_aco_costs)
    aco_cost_std = np.std(all_aco_costs)
    pure_aco_cost_mean = np.mean(all_pure_aco_costs) if all_pure_aco_costs else None
    pure_aco_cost_std = np.std(all_pure_aco_costs) if all_pure_aco_costs else None
    aco_improvement_mean = np.mean(all_aco_improvements) if all_aco_improvements else None
    aco_improvement_std = np.std(all_aco_improvements) if all_aco_improvements else None
    static_aco_cost_mean = np.mean(all_static_aco_costs) if all_static_aco_costs else None
    static_aco_cost_std = np.std(all_static_aco_costs) if all_static_aco_costs else None
    static_improvement_mean = np.mean(all_static_improvements) if all_static_improvements else None
    static_improvement_std = np.std(all_static_improvements) if all_static_improvements else None

    # Print results
    print(f"\nResults for {args.problem.upper()} (n={args.n_node}):")
    print(f"  Avg Cost: {avg_cost_mean:.4f} ± {avg_cost_std:.4f}")
    print(f"  Best Cost: {best_cost_mean:.4f} ± {best_cost_std:.4f}")
    print(f"  Guided ACO Cost: {aco_cost_mean:.4f} ± {aco_cost_std:.4f}")
    if pure_aco_cost_mean is not None:
        print(f"  Pure ACO Cost: {pure_aco_cost_mean:.4f} ± {pure_aco_cost_std:.4f}")
    if aco_improvement_mean is not None:
        print(f"  Guided vs Pure ACO Improvement: {aco_improvement_mean:.2f}% ± {aco_improvement_std:.2f}%")
    if static_aco_cost_mean is not None:
        print(f"  Static-Prior ACO Cost: {static_aco_cost_mean:.4f} ± {static_aco_cost_std:.4f}")
    if static_improvement_mean is not None:
        print(f"  Dynamic vs Static Prior Improvement: {static_improvement_mean:.2f}% ± {static_improvement_std:.2f}%")

    # Save results
    if args.save_results:
        results_path = Path(args.save_dir) / f"{args.problem}_n{args.n_node}_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "problem": args.problem,
            "n_node": args.n_node,
            "avg_cost_mean": float(avg_cost_mean),
            "avg_cost_std": float(avg_cost_std),
            "best_cost_mean": float(best_cost_mean),
            "best_cost_std": float(best_cost_std),
            "aco_cost_mean": float(aco_cost_mean),
            "aco_cost_std": float(aco_cost_std),
            "pure_aco_cost_mean": None if pure_aco_cost_mean is None else float(pure_aco_cost_mean),
            "pure_aco_cost_std": None if pure_aco_cost_std is None else float(pure_aco_cost_std),
            "aco_improvement_pct_mean": None if aco_improvement_mean is None else float(aco_improvement_mean),
            "aco_improvement_pct_std": None if aco_improvement_std is None else float(aco_improvement_std),
            "static_aco_cost_mean": None if static_aco_cost_mean is None else float(static_aco_cost_mean),
            "static_aco_cost_std": None if static_aco_cost_std is None else float(static_aco_cost_std),
            "dynamic_vs_static_improvement_pct_mean": None if static_improvement_mean is None else float(static_improvement_mean),
            "dynamic_vs_static_improvement_pct_std": None if static_improvement_std is None else float(static_improvement_std),
            "checkpoint": str(checkpoint_path),
            "config": config,
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {results_path}")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test NGFACO for BPP, MKP, OP")

    # Problem arguments
    parser.add_argument("--problem", type=str, choices=["bpp", "mkp", "op"], required=True,
                        help="Problem type")
    parser.add_argument("--n_node", type=int, default=50, help="Problem size")
    parser.add_argument("--m", type=int, default=5, help="Number of constraints for MKP")
    parser.add_argument("--capacity", type=float, default=150.0, help="Bin capacity for BPP")
    parser.add_argument("--max_len", type=float, default=4.0, help="Maximum route length for OP")

    # Testing arguments (matching DeepACO parameters)
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--test_size", type=int, default=16, help="Number of test instances")
    parser.add_argument("--H", type=int, default=5, help="Number of outer iterations (replaces T)")
    parser.add_argument("--mini_H", type=int, default=5, help="Number of inner iterations")

    # ACO arguments (matching DeepACO parameters)
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants")
    parser.add_argument("--k_sparse", type=int, default=32, help="K-NN size")
    parser.add_argument("--rho", type=float, default=0.9, help="Pheromone retention (0.9 = 90%% kept, 10%% decayed)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Heuristic weight")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="results_extended", help="Save directory")
    parser.add_argument("--save_results", action="store_true", help="Save results")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="Disable dynamic features")
    parser.add_argument("--no_baselines", "--no-baselines", dest="no_baselines", action="store_true",
                        help="Skip pure ACO baseline runs")
    parser.add_argument("--static_compare", action="store_true",
                        help="Also compare against a run that sets the neural prior only once at the beginning")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)
