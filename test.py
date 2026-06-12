#!/usr/bin/env python3
import torch
import argparse
import numpy as np
import random
import sys
import time
import os
import psutil
from pathlib import Path
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import json
import math
from datetime import datetime
import hashlib
import csv
from typing import Any, Dict, List, Optional, Tuple
import yaml


# Unified imports
import net
import faco
import utils
import baselines

from net import MultiDecoderNet, MultiHeadNet, Net
from baselines import get_baseline

# Import metric helpers from utils
from utils import (
    row_softmax, mean_row_kl, rel_l2_drift, top_set, top_turnover,
    top1_flip_rate, safe_corr, top_overlap_frac, row_top1_match_rate, EPS, infer_instance
)
from extended_common import (
    add_extended_problem_args,
    align_edge_attr_width as align_extended_edge_attr_width,
    build_pyg_data as build_extended_pyg_data,
    compute_relative_improvement as compute_extended_relative_improvement,
    create_model as create_extended_model,
    extract_problem_data as extract_extended_problem_data,
    prior_kwargs as get_extended_prior_kwargs,
    raw_values_to_objective as extended_raw_values_to_objective,
    reshape_prior_output as reshape_extended_prior_output,
    select_best_value as select_extended_best_value,
    setup_aco as setup_extended_aco,
)

BASE_PROBLEMS = {"tsp", "cvrp"}
EXTENDED_PROBLEMS = {"bpp", "mkp", "op"}

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = None
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()


def _clone_args(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    """Make a shallow copy of argparse.Namespace and apply overrides."""
    ns = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _base_cache_path(args: argparse.Namespace, val_list_len: int) -> Path:
    """Compute a deterministic cache path for baseline costs."""
    dataset_tag = "auto"
    dataset_stat = None
    if getattr(args, "generate_val", False):
        dataset_tag = f"generated:{getattr(args, 'save_generated', None) or 'memory'}"
    elif getattr(args, "dataset", None):
        dataset_tag = str(Path(args.dataset).expanduser().resolve())
        try:
            st = os.stat(dataset_tag)
            dataset_stat = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        except OSError:
            dataset_stat = {"size": None, "mtime_ns": None}

    # Baseline (heuristic-only) still depends on ACO hyperparams and randomness.
    key_payload = {
        "problem": args.problem,
        "alg": getattr(args, "alg", None),
        "n_node": int(args.n_node) if args.n_node is not None else None,
        "k_sparse": int(args.k_sparse),
        "n_ants": int(args.n_ants) if args.n_ants is not None else None,
        "H": int(args.H),
        "mini_H": int(args.mini_H),
        "rho": float(args.rho),
        "min_new_edges": int(args.min_new_edges),
        "no_local_search": bool(args.no_local_search),
        "no_smooth_mmas": bool(args.no_smooth_mmas),
        "no_extend_ls": bool(args.no_extend_ls),
        "no_normalized_heuristic": bool(args.no_normalized_heuristic),
        "disable_heuristic": bool(args.disable_heuristic),
        "L": int(getattr(args, "L", 0)),
        "seed": int(args.seed),
        "dataset": dataset_tag,
        "dataset_stat": dataset_stat,
        "n_instances": int(val_list_len),
    }
    key_str = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path("output") / "base_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"base_{args.problem}_{digest}.pt"


def _console_value(value, max_len: int = 96) -> str:
    """Format a value for human-readable console output."""
    if value is None:
        text = "-"
    elif isinstance(value, float):
        text = f"{value:.4f}"
    elif isinstance(value, (dict, list, tuple, set)):
        try:
            text = json.dumps(value, sort_keys=True, ensure_ascii=True)
        except TypeError:
            text = str(value)
    else:
        text = str(value)

    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _print_section(title: str, char: str = "=") -> None:
    line = char * max(12, len(title) + 4)
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def _print_table(rows, headers):
    # rows: list[list[str]]
    if not rows:
        return

    col_widths = [len(h) for h in headers]
    normalized_rows = []
    for row in rows:
        normalized = [str(cell) for cell in row]
        normalized_rows.append(normalized)
        for idx, cell in enumerate(normalized):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def _line(sep="-"):
        return "+" + "+".join(sep * (w + 2) for w in col_widths) + "+"

    print(_line("-"))
    print("| " + " | ".join(h.ljust(col_widths[idx]) for idx, h in enumerate(headers)) + " |")
    print(_line("="))
    for row in normalized_rows:
        print("| " + " | ".join(row[idx].ljust(col_widths[idx]) for idx in range(len(headers))) + " |")
    print(_line("-"))


def _print_kv_section(title: str, rows, *, key_header: str = "Field", value_header: str = "Value") -> None:
    if not rows:
        return
    _print_section(title)
    _print_table([(key, _console_value(value)) for key, value in rows], headers=[key_header, value_header])


def _resolve_method_plan(args: argparse.Namespace) -> list[str]:
    methods = []
    if not args.no_baseline:
        methods.append("base")

    if args.no_model or args.checkpoint == "none":
        return methods

    run_model_anneal = args.run_model_anneal
    run_model_no_anneal = args.run_model_no_anneal
    run_mix_anneal = args.run_mix_anneal and args.warmup
    run_mix_no_anneal = args.run_mix_no_anneal and args.warmup

    if not run_model_anneal and not run_model_no_anneal and not run_mix_anneal and not run_mix_no_anneal:
        if args.problem == "tsp":
            if args.H < 50:
                run_model_anneal = True
            else:
                run_mix_anneal = True
        else:
            if args.H < 50:
                run_model_no_anneal = True
            else:
                run_mix_no_anneal = True

    if run_model_anneal:
        methods.append("model(anneal)")
    if run_model_no_anneal:
        methods.append("model(no_anneal)")
    if run_mix_anneal:
        methods.append("mix(anneal)")
    if run_mix_no_anneal:
        methods.append("mix(no_anneal)")

    return methods


def _normalize_cli_argv(argv: Optional[List[str]] = None) -> List[str]:
    """Return a mutable CLI argument list."""
    if argv is None:
        return list(sys.argv[1:])
    return list(argv)


def _extract_cli_value(argv: List[str], flag: str) -> Optional[str]:
    """Extract a simple --flag value from argv, supporting --flag=value syntax."""
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _infer_problem_from_checkpoint(checkpoint_path: Optional[str]) -> Optional[str]:
    """Infer the training problem stored in a checkpoint config."""
    if not checkpoint_path or checkpoint_path == "none":
        return None
    try:
        path = Path(checkpoint_path)
        if not path.exists():
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        problem = config.get("problem")
        if isinstance(problem, str):
            return problem
    except Exception:
        return None
    return None


def _infer_test_problem(argv: Optional[List[str]] = None) -> Optional[str]:
    """Infer the requested evaluation problem from CLI or checkpoint."""
    args_list = _normalize_cli_argv(argv)
    cli_problem = _extract_cli_value(args_list, "--problem")
    if cli_problem:
        return cli_problem
    checkpoint_path = _extract_cli_value(args_list, "--checkpoint")
    return _infer_problem_from_checkpoint(checkpoint_path)


def _inject_problem_flag(argv: List[str], problem: Optional[str]) -> List[str]:
    """Inject an inferred --problem value for delegated extended parsing."""
    if not problem:
        return list(argv)
    if _extract_cli_value(argv, "--problem") is not None:
        return list(argv)
    return ["--problem", problem, *argv]


def _run_extended_aco_rollout(aco: Any, problem_type: str, n_iterations: int) -> float:
    """Run an extended-problem ACO solver and normalize the return value."""
    if problem_type == "bpp":
        return float(aco.run(n_iterations))
    if problem_type in {"mkp", "op"}:
        best_value, _ = aco.run(n_iterations)
        return float(best_value)
    raise ValueError(f"Unknown problem type: {problem_type}")


def _test_extended_instance(
    model: Any,
    instance_data: Any,
    args: argparse.Namespace,
    expected_edge_feats: int,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate a single extended-problem instance."""
    model.eval()
    data = extract_extended_problem_data(args.problem, instance_data)

    # Create a minimal mock ACO for static initial PyG data construction
    class MockACO:
        def __init__(self, n_nodes):
            self.pheromone = torch.ones((n_nodes, n_nodes))
            self.shortest_path = None

    if args.problem == "bpp":
        n_nodes = len(data["demand"])
        mock_aco = MockACO(n_nodes)
        initial_pyg_data = utils.build_pyg_data_bpp(
            demand=data["demand"],
            aco=mock_aco,
            device=args.device,
            dynamic=False,
        )
    elif args.problem == "mkp":
        n_nodes = len(data["prize"])
        mock_aco = MockACO(n_nodes + 1)  # MKP has dummy node
        initial_pyg_data = utils.build_pyg_data_mkp(
            prize=data["prize"],
            weight=data["weight"],
            aco=mock_aco,
            device=args.device,
            dynamic=False,
        )
    elif args.problem == "op":
        n_nodes = len(data["prizes"])
        mock_aco = MockACO(n_nodes + 1)  # OP has dummy node
        initial_pyg_data = utils.build_pyg_data_op(
            distances=data["distances"],
            prizes=data["prizes"],
            aco=mock_aco,
            device=args.device,
            dynamic=False,
        )
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")
    initial_pyg_data = align_extended_edge_attr_width(initial_pyg_data, expected_edge_feats)

    aco, pyg_args = setup_extended_aco(args, instance_data, args.problem, heuristic=None)

    static_aco_cost = None
    static_improvement_pct = None
    if args.static_compare:
        static_aco, _ = setup_extended_aco(args, instance_data, args.problem)
        with torch.no_grad():
            static_prior_output = model(initial_pyg_data)
            static_prior = reshape_extended_prior_output(
                static_prior_output,
                args.problem,
                **get_extended_prior_kwargs(args.problem, data),
            )
            static_aco.heuristic = static_prior.to(device=args.device, dtype=torch.float32)
            static_aco._sync_cpp_inputs()

            for _ in range(args.H):
                static_aco.sample(require_prob=False, prior=None, parallel_traced=True)
                static_aco.run(1)

        static_aco_cost = _run_extended_aco_rollout(static_aco, args.problem, args.H * args.mini_H)

    with torch.no_grad():
        for _ in range(args.H):
            if args.no_dynamic_feats:
                pyg_data = initial_pyg_data
            else:
                pyg_data = build_extended_pyg_data(aco, args.problem, *pyg_args, dynamic=True)
            pyg_data = align_extended_edge_attr_width(pyg_data, expected_edge_feats)

            prior_output = model(pyg_data)
            prior = reshape_extended_prior_output(
                prior_output,
                args.problem,
                **get_extended_prior_kwargs(args.problem, data),
            )
            aco.heuristic = prior.to(device=args.device, dtype=torch.float32)
            aco._sync_cpp_inputs()
            costs, _, _, _ = aco.sample(require_prob=False, prior=None, parallel_traced=True)
            aco.run(1)

    costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
    objective_t = extended_raw_values_to_objective(costs_t, args.problem)
    avg_cost = float(objective_t.mean().item())
    best_cost = select_extended_best_value(costs_t, args.problem)
    best_aco_cost = _run_extended_aco_rollout(aco, args.problem, args.H * args.mini_H)

    pure_aco_cost = None
    aco_improvement_pct = None
    if not args.no_baselines:
        pure_aco, _ = setup_extended_aco(args, instance_data, args.problem)
        pure_aco_cost = _run_extended_aco_rollout(pure_aco, args.problem, args.H * (args.mini_H + 1))
        aco_improvement_pct = compute_extended_relative_improvement(best_aco_cost, pure_aco_cost, args.problem)
    if static_aco_cost is not None:
        static_improvement_pct = compute_extended_relative_improvement(best_aco_cost, static_aco_cost, args.problem)

    return avg_cost, best_cost, {
        "avg_cost": avg_cost,
        "best_cost": best_cost,
        "best_aco_cost": best_aco_cost,
        "pure_aco_cost": pure_aco_cost,
        "aco_improvement_pct": aco_improvement_pct,
        "static_aco_cost": static_aco_cost,
        "static_improvement_pct": static_improvement_pct,
    }


def _test_extended_main(args: argparse.Namespace):
    """Main evaluation loop for BPP/MKP/OP."""
    # Load configuration from YAML if provided
    if args.config:
        print(f"Loading configuration from YAML: {args.config}")
        import yaml
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)

        # Override args with YAML values (CLI args take precedence if already set)
        for key, value in yaml_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    feats = state_dict["emb_net.v_lin0.weight"].shape[1]
    edge_feats = state_dict["emb_net.e_lin0.weight"].shape[1]
    model = create_extended_model(args.problem, m=args.m, feats=feats, edge_feats=edge_feats).to(args.device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")

    # Load test dataset based on problem type
    if args.problem == "bpp":
        test_instances = utils.load_bpp_test_dataset(args.n_node, args.device)
    elif args.problem == "mkp":
        test_instances = utils.load_mkp_test_dataset(args.n_node, args.m, args.device)
    elif args.problem == "op":
        test_instances = utils.load_op_test_dataset(args.n_node, args.device)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

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
        avg_cost, best_cost, metrics = _test_extended_instance(model, instance_data, args, edge_feats)
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


def _parse_extended_test_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse evaluation arguments for BPP/MKP/OP."""
    parser = argparse.ArgumentParser(description="Test NGFACO for BPP, MKP, OP", allow_abbrev=False)

    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file (for loading saved config)")

    add_extended_problem_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--test_size", type=int, default=16, help="Number of test instances")
    parser.add_argument("--H", type=int, default=5, help="Number of outer iterations (replaces T)")
    parser.add_argument("--mini_H", type=int, default=5, help="Number of inner iterations")
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants")
    parser.add_argument("--k_sparse", type=int, default=32, help="K-NN size")
    parser.add_argument("--rho", type=float, default=0.9, help="Pheromone retention (0.9 = 90%% kept, 10%% decayed)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Heuristic weight")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="results_extended", help="Save directory")
    parser.add_argument("--save_results", action="store_true", help="Save results")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="Disable dynamic features")
    # Add more arguments...
    parser.add_argument("--no_baselines", "--no-baselines", dest="no_baselines", action="store_true", help="Skip pure ACO baseline runs")
    parser.add_argument("--static_compare", action="store_true", help="Also compare against a run that sets the neural prior only once at the beginning")
    args = parser.parse_args(argv)

    # Load configuration from YAML if provided
    if args.config:
        print(f"Loading configuration from YAML: {args.config}")
        import yaml
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)

        # Override args with YAML values (CLI args take precedence if already set)
        for key, value in yaml_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    return args


def main(argv: Optional[List[str]] = None):
    args_list = _normalize_cli_argv(argv)
    inferred_problem = _infer_test_problem(args_list)
    if inferred_problem in EXTENDED_PROBLEMS:
        extended_argv = _inject_problem_flag(args_list, inferred_problem)
        args = _parse_extended_test_args(extended_argv)
        return _test_extended_main(args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file (for loading saved config)")
    parser.add_argument("--backend", type=str, default="standard", choices=["standard", "l2c"])
    parser.add_argument("--problem", type=str, default=None, choices=['tsp', 'cvrp'])
    parser.add_argument("--n_node", type=int, default=None)
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--alg", choices=["faco", "mmas"], default="faco", help="Algorithm type")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="none")
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone weight for MMAS")
    parser.add_argument("--beta", type=float, default=1.0, help="Heuristic weight for MMAS")
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_smooth_mmas", action="store_true")
    parser.add_argument("--no_extend_ls", action="store_true")
    parser.add_argument("--ls_scope", choices=["localized", "global"], default="localized")
    parser.add_argument("--ls_budget", choices=["truncated", "full"], default="truncated")
    parser.add_argument("--ls_max_opt", type=int, default=0,
                        help="Maximum accepted LS improving moves per call; 0 uses floor(N/4)")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--min_new_edges", type=int, default=12)
    parser.add_argument("--no_normalized_heuristic", action="store_true")
    parser.add_argument("--no_logit_net", action="store_true")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    parser.add_argument("--multi_head", action="store_true",
                        help="Use a Multi-Prediction style K-head edge-prior decoder")
    parser.add_argument("--num_heads", type=int, default=1,
                        help="Number of prediction heads; values >1 enable multi-head model loading")
    parser.add_argument("--head_zdim", type=int, default=16,
                        help="Latent code width for each multi-head decoder head")
    parser.add_argument("--head_decoder_type", choices=["film", "separate"], default="film",
                        help="Multi-head decoder implementation: FiLM-conditioned shared decoder or separate decoders")
    parser.add_argument("--head_deploy", choices=["ants", "head"], default="ants",
                        help="How to deploy K heads: mixed ant groups or one fixed head")
    parser.add_argument("--head_index", type=int, default=0,
                        help="Head index used when --head_deploy=head")
    parser.add_argument("--head_ant_weights", type=str, default=None,
                        help="Comma-separated ant allocation weights/counts for mixed-head deployment, e.g. 50,20,15,15")
    parser.add_argument("--head_router", choices=["static", "ema"], default="static",
                        help="Ant routing for mixed-head deployment; static preserves head_ant_weights")
    parser.add_argument("--head_router_alpha", type=float, default=0.25,
                        help="EMA update rate for adaptive head utility routing")
    parser.add_argument("--head_router_gamma", type=float, default=2.0,
                        help="Softmax strength for converting head utility to ant counts")
    parser.add_argument("--head_router_min_frac", type=float, default=0.0,
                        help="Minimum allocation fraction blended into each head by adaptive routing")
    parser.add_argument("--head_router_score_mode", choices=["mean", "topq", "best", "improvement"], default="topq",
                        help="Per-head score used by adaptive routing")
    parser.add_argument("--head_complement_coef", type=float, default=0.0,
                        help="Coefficient for high-priority edge complementarity reward across heads")
    parser.add_argument(
        "--head_input_transform",
        "--head-input-transform",
        choices=["none", "d4"],
        default="none",
        help="Apply a deterministic distance-preserving coordinate transform per head before encoding",
    )

    # Ablation studies
    parser.add_argument("--ablation_pheromone_features", action="store_true",
                        help="Remove pheromone-based features (tau_cv, log_tau_rel)")
    parser.add_argument("--ablation_incumbent_features", action="store_true", 
                        help="Remove incumbent-based features (source_succ, source_pred, new_edge)")
    parser.add_argument(
        "--edge_feature_set",
        "--edge-feature-set",
        choices=["full", "compact3"],
        default="full",
        help="Edge feature layout: full=6 channels, compact3=dist_norm/log_tau_rel/is_in_incumbent",
    )

    
    parser.add_argument("--baseline", type=str, default='default')
    parser.add_argument("--baseline_time_limit", type=float, default=2.0)
    parser.add_argument("--baseline_runs", type=int, default=1)
    parser.add_argument("--no_anneal", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_output", type=str, default="visualizations")
    parser.add_argument("--timed", action="store_true")
    parser.add_argument("--runtime_limit", type=float, default=None,
                        help="Wall-clock inference budget in seconds; stop when exceeded")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--L", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)

    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.5)
    parser.add_argument("--generate_val", action="store_true", help="Generate test set instead of loading from file")
    parser.add_argument("--save_generated", type=str, default=None, help="Path to save generated test dataset")
    parser.add_argument("--capacity_override", type=float, default=None,
                        help="Override CVRP capacity during generated evaluation dataset creation")
    parser.add_argument("--val_size", type=int, default=None, help="Limit validation set size")
    parser.add_argument("--log", action="store_true", help="Enable logging to file (auto-named)")
    parser.add_argument("--no_baseline", "--no-baseline", dest="no_baseline", action="store_true",
                        help="Skip pure MFACO baseline calculation")
    parser.add_argument("--rl_data", action="store_true", help="Load TSPLIB/CVRPLIB data instead of standard test set")
    
    # Selective method execution
    parser.add_argument("--run_model_anneal", action="store_true", help="Run Model(anneal) method")
    parser.add_argument("--run_mix_anneal", action="store_true", help="Run Mix(anneal) method")
    parser.add_argument("--run_model_no_anneal", action="store_true", help="Run Model(no_anneal) method")
    parser.add_argument("--run_mix_no_anneal", action="store_true", help="Run Mix(no_anneal) method")
    parser.add_argument("--no_model", action="store_true", help="Skip model loading and run only baseline methods")

    # Per-iteration logging (H * mini_H rows per run)
    parser.add_argument("--iter_log", action="store_true", help="Log mean/best at every mini-iteration to CSV")
    parser.add_argument("--iter_print", action="store_true", help="Print mean/best at every mini-iteration (very verbose)")
    parser.add_argument("--stage_metrics", action="store_true", help="Collect pre-LS and post-update stage metrics during evaluation")
    parser.add_argument("--collect_guidance_metrics", action="store_true",
                        help="Collect mean guidance and pheromone correlation metrics for executed neural methods")
    parser.add_argument("--summary_json", type=str, default=None, help="Optional path to write structured evaluation summary JSON")
    parser.add_argument("--verbose_config", "--verbose-config", dest="verbose_config", action="store_true",
                        help="Print the full flattened config table instead of the compact grouped console summary")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--encoder_layer_num", type=int, default=6)
    parser.add_argument("--head_num", type=int, default=8)
    parser.add_argument("--qkv_dim", type=int, default=16)
    parser.add_argument("--ff_hidden_dim", type=int, default=512)
    parser.add_argument("--k_nearest_num", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--prc", action="store_true", default=False,
                        help="Legacy L2C compatibility flag. The repo-native L2C backend does not use PRC.")

    args = parser.parse_args(args_list)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        def _cli_has_flag(key: str) -> bool:
            flag = "--" + key
            flag_hyphen = "--" + key.replace("_", "-")
            return any(
                token == flag
                or token == flag_hyphen
                or token.startswith(f"{flag}=")
                or token.startswith(f"{flag_hyphen}=")
                for token in args_list
            )

        for key, value in yaml_config.items():
            if hasattr(args, key) and not _cli_has_flag(key):
                setattr(args, key, value)



    # Args setup
    ckpt = None
    
    # Auto-load checkpoint if not provided and --no_model not set
    if args.checkpoint == "none" and not args.no_model:
        # Build expected checkpoint filename based on current args
        # Pattern: {problem}_n{n_node}_k{k_sparse}_ants{n_ants}_H{H}_miniH{mini_H}_rho{rho}_mne{min_new_edges}_ppo_lr{lr}_best.pt
        import glob
        pretrained_dir = f"pretrained/{args.problem}/n{args.n_node}"
        if os.path.exists(pretrained_dir):
            # Try to find a matching checkpoint

            pattern = (
                f"{args.problem}_n{args.n_node}_*l2c*_best.pt"
                if args.backend == "l2c"
                else f"{args.problem}_n{args.n_node}_*_rho{args.rho}_*_best.pt"
            )
            matches = glob.glob(os.path.join(pretrained_dir, pattern))
            if matches:
                # Prefer exact match, otherwise use first match
                args.checkpoint = matches[0]
                print(f"Auto-detected checkpoint: {args.checkpoint}")
            else:
                print(f"Warning: No checkpoints found in {pretrained_dir}, running without model")
                args.no_model = True
        else:
            print(f"Warning: Pretrained directory {pretrained_dir} not found, running without model")
            args.no_model = True
    
    if args.checkpoint != "none" and not args.no_model:
        print(f"Loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        config = ckpt.get("config", {})
        ckpt_rows = [("path", args.checkpoint)]
        for key in ["epoch", "val_cost", "val_gap"]:
            if key in ckpt:
                value = ckpt[key]
                if key == "val_gap" and value is not None:
                    value = f"{float(value):.4f}%"
                ckpt_rows.append((key, value))
        train_cfg_parts = []
        for key in ["problem", "n_node", "k_sparse", "n_ants", "H", "mini_H", "rho", "min_new_edges", "warmup", "edge_feature_set"]:
            if key in config:
                train_cfg_parts.append(f"{key}={config[key]}")
        if train_cfg_parts:
            ckpt_rows.append(("train_config", ", ".join(train_cfg_parts)))
        _print_kv_section("Checkpoint", ckpt_rows)

        # Override args from config if not present in sys.argv
        # Checkpoint/Device/Data args should NOT be overwritten usually
        ignore_args = {
            "checkpoint", "device", "dataset", "visualize", "visualize_output", 
            "timed", "verify", "baseline", "baseline_runs", "baseline_time_limit", 
            "threads", "seed", "save_dir", "wandb_project", "wandb_entity", "no_wandb", "warmup", "no_baseline",
            "val_size"
        }
        
        # If model was not trained with annealing, ignore its min_gamma (use ours)
        was_annealed = config.get("train_anneal", False) or config.get("anneal_prior", False)
        if not was_annealed:
             ignore_args.add("min_gamma")
        
        restored_overrides = []
        for k, v in config.items():
            if k in ignore_args: continue
            if not hasattr(args, k): continue

            # Simple check: if flag is in sys.argv, user overrode it
            flag_underscore = "--" + k
            flag_hyphen = "--" + k.replace("_", "-")

            if (flag_underscore not in sys.argv) and (flag_hyphen not in sys.argv):
                 current_val = getattr(args, k)
                 if current_val != v:
                     restored_overrides.append((k, current_val, v))
                     setattr(args, k, v)
        
        # Explicitly check for ablation flags in config if not in sys.argv
        if "ablation_pheromone_features" in config and "--ablation_pheromone_features" not in sys.argv:
            old_val = args.ablation_pheromone_features
            args.ablation_pheromone_features = config["ablation_pheromone_features"]
            if old_val != args.ablation_pheromone_features:
                restored_overrides.append(("ablation_pheromone_features", old_val, args.ablation_pheromone_features))
            
        if "ablation_incumbent_features" in config and "--ablation_incumbent_features" not in sys.argv:
            old_val = args.ablation_incumbent_features
            args.ablation_incumbent_features = config["ablation_incumbent_features"]
            if old_val != args.ablation_incumbent_features:
                restored_overrides.append(("ablation_incumbent_features", old_val, args.ablation_incumbent_features))
        if (
            "edge_feature_set" in config
            and "--edge_feature_set" not in sys.argv
            and "--edge-feature-set" not in sys.argv
        ):
            old_val = args.edge_feature_set
            args.edge_feature_set = config["edge_feature_set"]
            if old_val != args.edge_feature_set:
                restored_overrides.append(("edge_feature_set", old_val, args.edge_feature_set))
        if restored_overrides:
            _print_section("Restored config overrides")
            _print_table(
                [(k, _console_value(old), _console_value(new)) for k, old, new in restored_overrides],
                headers=["Parameter", "CLI/default", "Checkpoint"],
            )
        
    if args.problem is None:
         raise ValueError("Problem must be specified (in args or checkpoint).")
         
    if args.n_node is None:
         args.n_node = 100
    
    if args.baseline == 'default':
        args.baseline = 'lkh' if args.problem == 'tsp' else 'hgs'

    args.extend_ls = not args.no_extend_ls
    
    if args.threads is None:
        args.threads = psutil.cpu_count(logical=False)
    faco.set_faco_cpp_threads(args.threads)

    compact_rows = [
        ("backend", args.backend),
        ("problem", args.problem),
        ("n_node", args.n_node),
        ("device", args.device),
        ("seed", args.seed),
        ("dataset", args.dataset or "auto"),
        ("val_size", args.val_size if args.val_size is not None else "full"),
        ("checkpoint", args.checkpoint if args.checkpoint != "none" else "disabled"),
        ("methods", ", ".join(_resolve_method_plan(args)) or "none"),
        ("search", f"{args.alg}, k={args.k_sparse}, ants={args.n_ants}, H={args.H}, mini_H={args.mini_H}"),
        ("pheromone", f"rho={args.rho}, min_new_edges={args.min_new_edges}, alpha={args.alpha}, beta={args.beta}"),
        ("local_search", "off" if args.no_local_search else f"on ({args.ls_scope}, {args.ls_budget}, max_opt={args.ls_max_opt or 'auto'})"),
        ("warmup", "off" if not args.warmup else f"on (ratio={args.warmup_ratio})"),
        ("baseline", "disabled" if args.no_baseline else f"{args.baseline} (runs={args.baseline_runs}, limit={args.baseline_time_limit}s)"),
        ("outputs", f"log={args.log}, iter_log={args.iter_log}, summary_json={args.summary_json or '-'}"),
    ]
    if args.verbose_config:
        _print_kv_section(
            "Run config (full)",
            [(k, v) for k, v in sorted(vars(args).items())],
            key_header="Parameter",
            value_header="Value",
        )
    else:
        _print_kv_section("Run config", compact_rows)

    # Seeds
    # Seeds
    utils.set_seed(args.seed)

    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = "nockpt"
        if args.checkpoint != "none":
            ckpt_name = Path(args.checkpoint).stem
        
        data_name = "gen"
        if args.dataset:
            data_name = Path(args.dataset).stem
        
        log_dir = "logs"
        log_name = f"{log_dir}/test_{args.problem}_{args.n_node}_{ckpt_name}_{data_name}_{timestamp}_annealing{str(not args.no_anneal)}.txt"

        csv_dir = Path(log_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_name_base = f"test_{args.problem}_{args.n_node}_{ckpt_name}_{data_name}_{timestamp}_annealing{str(not args.no_anneal)}"
        csv_path_instances = csv_dir / f"{csv_name_base}_instances.csv"
        csv_path_summary = csv_dir / f"{csv_name_base}_summary.csv"
        csv_path_iters = csv_dir / f"{csv_name_base}_iters.csv"
        
        # Logger handles mkdir
        logger = Logger(log_name)
        sys.stdout = logger
        sys.stderr = logger
        print(f"Logging to {log_name}")
        print(f"CSV instances: {csv_path_instances}")
        print(f"CSV summary:   {csv_path_summary}")
        if args.iter_log:
            print(f"CSV iters:     {csv_path_iters}")
    else:
        csv_path_instances = None
        csv_path_summary = None
        csv_path_iters = None

        # If user wants iter logging without --log, still write a CSV.
        if args.iter_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = "nockpt" if args.checkpoint == "none" else Path(args.checkpoint).stem
            data_name = "gen" if not args.dataset else Path(args.dataset).stem
            out_dir = Path("output") / "iter_logs"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_name_base = f"test_{args.problem}_{args.n_node}_{ckpt_name}_{data_name}_{timestamp}_annealing{str(not args.no_anneal)}"
            csv_path_iters = out_dir / f"{csv_name_base}_iters.csv"
            print(f"CSV iters:     {csv_path_iters}")
    
    import functools
    # Setup modules
    if args.problem == 'tsp':
        build_fn = utils.build_pyg_data_tsp
        gen_fn = utils.generate_tsp_instance
        if args.alg == 'mmas':
            MFACO = faco.ACO_TSP
        else:
            MFACO = faco.MFACO_TSP
    else:
        build_fn = utils.build_pyg_data_cvrp
        gen_fn = utils.gen_cvrp_instance
        if args.alg == 'mmas':
            MFACO = faco.ACO_CVRP
        else:
            MFACO = faco.MFACO_CVRP

    # Dataset
    if args.generate_val:
        # Generate test dataset dynamically
        baseline_solver = 'none' # Don't force LKH during generation unless specified elsewhere
        val_list = utils.generate_and_save_dataset(
            problem=args.problem,
            n_node=args.n_node,
            n_instances=args.val_size if args.val_size is not None else 16,
            save_path=args.save_generated,
            baseline_solver=baseline_solver,
            baseline_runs=args.baseline_runs,
            time_limit=args.baseline_time_limit,
            device='cpu',
            capacity_override=args.capacity_override if args.problem == 'cvrp' else None,
        )
    elif args.dataset:
        print(f"Loading {args.dataset}...")
        if args.dataset.endswith(".txt") and args.problem == 'tsp':
             val_list = utils.load_tsp_txt_dataset(args.dataset)
        elif args.dataset.endswith(".txt") and args.problem == 'cvrp':
             val_list = utils.load_cvrp_txt_dataset(args.dataset)
        else:
            data = torch.load(args.dataset, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                if "coords" in data: val_list = data["coords"]
                else: val_list = data
            else:
                val_list = data
    else:
        print("Loading validation dataset...")
        val_list = utils.load_auto_dataset(
            args.n_node, 
            problem=args.problem, 
            rl_data=args.rl_data,
            device='cpu'
        )
        
        if val_list is None:
            print("Generating data...")
            print("Generating 16 instances on fly...")
            val_list = []
            for _ in range(16):
                if args.problem == 'tsp':
                    val_list.append(torch.from_numpy(gen_fn(args.n_node)))
                else:
                    c, d, cap = gen_fn(args.n_node, device='cpu', capacity=args.capacity_override)
                    val_list.append((c.cpu(), d.cpu(), cap))
            
            # Save for reuse
            utils.save_val_dataset(val_list, args.n_node, problem=args.problem)

    # Limit validation set size if requested
    if args.val_size is not None and val_list is not None:
        if isinstance(val_list, (list, tuple)) or torch.is_tensor(val_list):
            original_len = len(val_list)
            val_list = val_list[:args.val_size]
            print(f"Limited validation dataset from {original_len} to {len(val_list)} instances.")

    # Baseline
    baseline_values = None
    
    # Check if dataset has embedded baseline costs (e.g. from file)
    if isinstance(val_list, list) and len(val_list) > 0:
        if args.problem == 'tsp' and isinstance(val_list[0], tuple) and len(val_list[0]) >= 2:
            try:
                costs = [x[1] for x in val_list]
                # Allow python float/int and numpy scalars. Check strictly positive.
                if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) and c > 1e-6 for c in costs):
                    print("Using baseline costs from dataset.")
                    baseline_values = np.array(costs)
            except Exception: pass
            
        # CVRP Tuple: (coords, demand, capacity, cost, tour)
        elif args.problem == 'cvrp' and isinstance(val_list[0], tuple) and len(val_list[0]) == 5:
             try:
                 costs = [x[3] for x in val_list]
                 if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) and c > 1e-6 for c in costs):
                     print("Using baseline costs from dataset.")
                     baseline_values = np.array(costs)
             except Exception: pass
    
    # Hardcode for specific datasets if requested (fix for TSP10000)
    if args.dataset and "tsp10000" in args.dataset.lower() and "test" in args.dataset.lower():
         print("Hardcoding baseline cost to 71.778 for TSP10000 test set.")
         # Assuming val_list has correct length, we create an array of this cost
         if val_list:
             baseline_values = np.full(len(val_list), 71.778)

    if (not args.no_baseline) and args.baseline != 'none' and baseline_values is None:
        try:
            print("Computing baseline...")
            # Use TensorDataset wrapper for CVRP get_baseline compatibility
            if args.problem == 'cvrp' and isinstance(val_list, list) and len(val_list) > 0 and not isinstance(val_list[0], tuple):
                 # Only if not tuples (i.e. if already tensors)
                 pass
            
            # If tuple dataset (text), we need to extract coords/demands for baseline?

            
            if args.problem == 'cvrp' and isinstance(val_list, list) and not hasattr(val_list, 'tensors'):
                if len(val_list)>0 and isinstance(val_list[0], tuple) and len(val_list[0]) == 5:
                     # Text dataset tuple: (coords, demand, capacity, cost, tour)
                     cs = torch.stack([x[0] for x in val_list])
                     ds = torch.stack([x[1] for x in val_list])
                     caps = torch.stack([torch.tensor(x[2]) for x in val_list]) # Capacity is float

                     ds_wrapper = torch.utils.data.TensorDataset(cs, ds, caps)
                     baseline_values = get_baseline(ds_wrapper, problem='cvrp', n_node=args.n_node, time_limit=args.baseline_time_limit)
                elif len(val_list)>0 and isinstance(val_list[0], tuple) and len(val_list[0]) == 3:
                     # Generated: (c, d, cap)
                     cs = torch.stack([x[0] for x in val_list])
                     ds = torch.stack([x[1] for x in val_list])
                     caps = torch.stack([torch.tensor(x[2]) for x in val_list])
                     ds_wrapper = torch.utils.data.TensorDataset(cs, ds, caps)
                     baseline_values = get_baseline(ds_wrapper, problem='cvrp', n_node=args.n_node, time_limit=args.baseline_time_limit)
            else:
                # Handle potential tuple items in TSP (coords, cost, tour) for baselines
                # Just extract coords for baseline computation if needed
                if args.problem == 'tsp' and isinstance(val_list, list) and len(val_list) > 0 and isinstance(val_list[0], tuple):
                    val_list_coords = [x[0] if isinstance(x, tuple) else x for x in val_list]
                    baseline_values = get_baseline(val_list_coords, problem=args.problem, n_node=args.n_node, runs=args.baseline_runs, time_limit=args.baseline_time_limit)
                else:
                    baseline_values = get_baseline(val_list, problem=args.problem, n_node=args.n_node, runs=args.baseline_runs, time_limit=args.baseline_time_limit)
            
            if baseline_values is not None:
                 baseline_values = baseline_values.cpu().numpy()
                 print(f"Baseline mean: {baseline_values.mean()}")
        except Exception as e:
            print(f"Warning: Failed to compute external baseline ({e}). Continuing without it.")
            baseline_values = None

    # Model
    model = None
    if ckpt is not None and not args.no_model:
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        
        # Override args from config logic omitted for brevity, similar to original but using unified flags
        config = ckpt.get("config", {})
        # ... logic to override args if needed ...

        # Model
        # Check input feats from checkpoint if available
        feats = 2 if args.problem == 'tsp' else 4

        if "emb_net.v_lin0.weight" in state_dict:
            ckpt_feats = state_dict["emb_net.v_lin0.weight"].shape[1]
            if ckpt_feats != feats:
                print(f"Override feats={feats} with checkpoint feats={ckpt_feats}")
                feats = ckpt_feats

        # Always use 6 edge features, unless checkpoint differs
        edge_feats = 6
        if "emb_net.e_lin0.weight" in state_dict:
            ckpt_edge_feats = state_dict["emb_net.e_lin0.weight"].shape[1]
            if ckpt_edge_feats != edge_feats:
                print(f"Override edge_feats={edge_feats} with checkpoint edge_feats={ckpt_edge_feats}")
                edge_feats = ckpt_edge_feats
            if (
                ckpt_edge_feats == 3
                and getattr(args, "edge_feature_set", "full") == "full"
                and "--edge_feature_set" not in sys.argv
                and "--edge-feature-set" not in sys.argv
            ):
                print("Checkpoint has 3 edge features; using edge_feature_set=compact3")
                args.edge_feature_set = "compact3"

        multi_head = bool(getattr(args, "multi_head", False) or getattr(args, "num_heads", 1) > 1)
        if config.get("multi_head", False) or config.get("num_heads", 1) > 1:
            multi_head = True
            if "--num_heads" not in sys.argv and "--num-heads" not in sys.argv:
                args.num_heads = int(config.get("num_heads", args.num_heads))
            if "--head_zdim" not in sys.argv and "--head-zdim" not in sys.argv:
                args.head_zdim = int(config.get("head_zdim", args.head_zdim))
            if "--head_decoder_type" not in sys.argv and "--head-decoder-type" not in sys.argv:
                args.head_decoder_type = config.get("head_decoder_type", args.head_decoder_type)
            if "--head_deploy" not in sys.argv and "--head-deploy" not in sys.argv:
                args.head_deploy = config.get("head_deploy", args.head_deploy)
            if "--head_ant_weights" not in sys.argv and "--head-ant-weights" not in sys.argv:
                args.head_ant_weights = config.get("head_ant_weights", args.head_ant_weights)
            if "--head_input_transform" not in sys.argv and "--head-input-transform" not in sys.argv:
                args.head_input_transform = config.get("head_input_transform", args.head_input_transform)
        if multi_head:
            decoder_type = str(getattr(args, "head_decoder_type", "film") or "film").lower()
            model_cls = MultiDecoderNet if decoder_type == "separate" else MultiHeadNet
        else:
            model_cls = Net
        model_kwargs = {}
        if multi_head:
            model_kwargs.update(num_heads=args.num_heads, head_zdim=args.head_zdim)
        model = model_cls(
            feats=feats,
            edge_feats=edge_feats,
            logit_net=not args.no_logit_net,
            **model_kwargs,
        ).to(args.device)
        model.load_state_dict(state_dict)
        model.eval()

    # Eval
    results = {
        "base_cost": [],
        "base_time": [],
        "base_metrics": {},

        # Neural-guided costs (anneal on/off)
        "model_cost": [],
        "model_cost_no_anneal": [],
        "model_time": [],
        "model_time_no_anneal": [],
        "model_metrics": {},

        # Warmup/mix costs (anneal on/off)
        "mix_cost": [],
        "mix_cost_no_anneal": [],
        "mix_time": [],
        "mix_time_no_anneal": [],
        "mix_metrics": {},

        "opt_cost": [],
        "base_gap": [],
        "model_gap": [],
        "model_gap_no_anneal": [],
        "mix_gap": [],
        "mix_gap_no_anneal": [],
        "model_gap_no_anneal": [],
        "mix_gap": [],
        "mix_gap_no_anneal": [],
        "model_time_breakdown": {
            "neural": [], "sampling": [], "ls": [], "update": []
        }
    }

    if args.problem == 'cvrp':
        TARGET_ITERS = [1000, 2000, 5000, 10000]
    else:
        TARGET_ITERS = [1000, 2000, 5000, 10000]

    for itr in TARGET_ITERS:
        if itr % args.mini_H != 0:
            print(f"Warning: Target iteration {itr} is not a multiple of mini_H ({args.mini_H}). It will never be logged.")
    # Initialize keys for Target iterations
    for itr in TARGET_ITERS:
        results[f"base_cost_I{itr}"] = []
        results[f"base_time_I{itr}"] = []
        
        results[f"model_cost_I{itr}"] = []
        results[f"model_time_I{itr}"] = []
        results[f"model_neural_time_I{itr}"] = []
        
        results[f"model_no_anneal_cost_I{itr}"] = []
        results[f"model_no_anneal_time_I{itr}"] = []
        results[f"model_no_anneal_neural_time_I{itr}"] = []

        results[f"mix_cost_I{itr}"] = []
        results[f"mix_time_I{itr}"] = []
        results[f"mix_neural_time_I{itr}"] = []

        results[f"mix_no_anneal_cost_I{itr}"] = []
        results[f"mix_no_anneal_time_I{itr}"] = []
        results[f"mix_no_anneal_neural_time_I{itr}"] = []
        
    results["model_neural_time"] = []
    results["model_neural_time_no_anneal"] = []
    results["mix_neural_time"] = []
    results["mix_neural_time_no_anneal"] = []
    stage_metric_keys = [
        "avg_ant_before_ls",
        "avg_ant_after_ls",
        "avg_incumbent_after_aco",
        "final_ant_before_ls",
        "final_ant_after_ls",
        "final_incumbent_after_aco",
    ]
    for prefix in ["base", "model", "model_no_anneal", "mix", "mix_no_anneal"]:
        for metric_key in stage_metric_keys:
            results[f"{prefix}_{metric_key}"] = []
        results[f"{prefix}_mean_guidance"] = []
        results[f"{prefix}_pheromone_correlation"] = []
        results[f"{prefix}_metrics_payload"] = []

    iterable = val_list
    if args.problem == 'cvrp' and hasattr(val_list, 'tensors'):
         iterable = torch.utils.data.DataLoader(val_list, batch_size=1, shuffle=False)
    
    planned_methods = ", ".join(_resolve_method_plan(args)) or "none"
    print(f"Evaluating {len(val_list)} instances [{planned_methods}]...")

    # Baseline cache (heuristic-only MFACO). Reuse across runs for the same dataset+config.
    cached_base_costs = None
    if not args.no_baseline:
        cached_res = utils.load_pure_mfaco_cache(args, val_list)
        if cached_res is not None and "costs" in cached_res:
            cached_base_costs = cached_res["costs"]
            if len(cached_base_costs) == len(val_list):
                 print(f"Loaded cached base costs from shared cache ({len(cached_base_costs)} instances)")
            else:
                 print(f"Warning: cached costs length {len(cached_base_costs)} != dataset length {len(val_list)}")
                 cached_base_costs = None

    if args.iter_log and cached_base_costs is not None:
        # Can't reconstruct per-iteration traces from cached scalar costs.
        print("Iter logging enabled: ignoring cached base costs to record per-iteration trace.")
        cached_base_costs = None
    
    sample_snapshots = {}

    per_instance_rows = []
    opt_costs_summary = []

    iter_csv_f = None
    iter_csv_writer = None
    if args.iter_log and csv_path_iters is not None:
        try:
            iter_csv_f = open(csv_path_iters, "w", newline="")
            iter_csv_writer = csv.DictWriter(
                iter_csv_f,
                fieldnames=[
                    "idx",
                    "name",
                    "method",
                    "anneal",
                    "iter",
                    "t",
                    "mini_t",
                    "mean",
                    "best",
                    "mean_before_ls",
                    "mean_after_ls",
                    "incumbent_after_aco",
                ],
            )
            iter_csv_writer.writeheader()
        except Exception as e:
            print(f"Warning: failed to open iter CSV {csv_path_iters}: {e}")
            iter_csv_f = None
            iter_csv_writer = None

    def _record_history(prefix, history, res_dict):
        # Build a mapping from total iterations (H * mini_H) to values
        # history items: (H, time, cost) or (H, time, cost, neural_time)
        iter_map = {}
        for item in history:
            t = item[0]  # H value
            total_iters = t * args.mini_H
            val_tuple = (item[1], item[2])
            n_time = item[3] if len(item) > 3 else 0.0
            iter_map[total_iters] = (val_tuple[0], val_tuple[1], n_time)

        for itr in TARGET_ITERS:
            val = iter_map.get(itr)
            if val:
                res_dict[f"{prefix}_cost_I{itr}"].append(val[1])
                res_dict[f"{prefix}_time_I{itr}"].append(val[0])
                if f"{prefix}_neural_time_I{itr}" in res_dict:
                    res_dict[f"{prefix}_neural_time_I{itr}"].append(val[2])
            else:
                res_dict[f"{prefix}_cost_I{itr}"].append(None)
                res_dict[f"{prefix}_time_I{itr}"].append(None)
                if f"{prefix}_neural_time_I{itr}" in res_dict:
                    res_dict[f"{prefix}_neural_time_I{itr}"].append(None)

    def _append_stage_metrics(prefix, stage_metrics):
        stage_metrics = stage_metrics or {}
        for metric_key in stage_metric_keys:
            results[f"{prefix}_{metric_key}"].append(stage_metrics.get(metric_key))

    def _safe_metric_mean(values):
        if not values:
            return None
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr.mean())

    def _append_guidance_metrics(prefix, metric_payload):
        metric_payload = metric_payload or {}
        results[f"{prefix}_metrics_payload"].append(metric_payload)
        prior_mean = metric_payload.get("prior_mean", [])
        corr = metric_payload.get("corr", [])
        results[f"{prefix}_mean_guidance"].append(_safe_metric_mean(prior_mean))
        results[f"{prefix}_pheromone_correlation"].append(_safe_metric_mean(corr))

    total_instances = len(val_list)
    for i, item in enumerate(tqdm(iterable)):
        opt_cost = None
        
        # Unpack TSP tuple if present
        name = f"Instance {i}"
        if args.problem == 'tsp' and isinstance(item, tuple):
             # (coords, cost, tour) or (coords, cost, tour, name)
             coords = item[0]
             if len(item) > 1: opt_cost = item[1]
             if len(item) > 3: name = item[3]
             item = coords
        
        if args.problem == 'cvrp':
            if isinstance(item, list) and len(item)==1: item = item[0] # DataLoader batch=1
            
            # Unpack CVRP Text Tuple (coords, demand, capacity, cost, tour) or (..., name)
            if isinstance(item, tuple) and len(item) >= 5:
                # (coords, demand, capacity, cost, tour)
                coords, demand, capacity, cost, tour = item[:5]
                if len(item) > 5: name = item[5]
                
                if cost is not None and isinstance(cost, (float, int)) and cost > 0:
                    opt_cost = cost
                
                # Reduce to (coords, demand, capacity) for solver
                item = (coords, demand, capacity)

            # item is [coords, demand, cap] tensors if from DataLoader
            # or (coords, demand, cap) tuple if from list
            if isinstance(item, (list, tuple)):
                # If from DataLoader batch=1, we might have (1, N, 2). If from list, (N, 2).
                # Only unbatch if looks like batch dim 
                item = [x[0] if torch.is_tensor(x) and x.dim()==3 else x for x in item] # Unbatch if batched?
                # DataLoader adds batch dim? Yes batch_size=1 -> (1, n, 2).
                if torch.is_tensor(item[0]) and item[0].shape[0] == 1 and item[0].dim() == 3:
                     item[0] = item[0].squeeze(0).numpy()
                     item[1] = item[1].squeeze(0).numpy()
                     item[2] = float(item[2].item())
                elif torch.is_tensor(item[0]): # From list of tensors
                     item[0] = item[0].numpy()
                     item[1] = item[1].numpy()
                     item[2] = float(item[2])
                
                # Normalize demand if capacity > 1.0 (indicating raw data)
                capacity = float(item[2])
                if capacity > 1.0 + 1e-6:
                     item[1] = item[1] / capacity
                     item[2] = 1.0

            if opt_cost is not None and isinstance(opt_cost, (int, float)) and float(opt_cost) > 1e-12:
                opt_costs_summary.append(float(opt_cost))
            
        # Base
        base_m = None
        base_best = None
        base_iter_stats = None
        if not args.no_baseline:
            if cached_base_costs is not None:
                base_best = float(cached_base_costs[i])
                results["base_cost"].append(base_best)
                results["base_time"].append(0.0)
                _append_stage_metrics("base", None)
                # Fill intermediate history with None for cached results
                _record_history("base", [], results)
            else:
                tb0 = time.time()
                base_ret = infer_instance(
                    args.problem, MFACO, build_fn, None, item,
                    args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                    args,
                    use_heuristic_only=True,
                    collect_metrics=args.visualize,
                    metrics_every_step=args.visualize,
                    seed=args.seed + i
                )
                tb1 = time.time()
                _, base_best, base_timings, base_extra = base_ret
                base_m = base_extra.get("metrics")
                base_iter_stats = base_extra.get("iter_stats")
                results["base_cost"].append(base_best)
                results["base_time"].append(tb1 - tb0)
                _append_stage_metrics("base", base_extra.get("stage_metrics"))
                
                _record_history("base", base_extra.get("history", []), results)

                if args.iter_log and iter_csv_writer is not None and base_iter_stats is not None:
                    for st in base_iter_stats:
                        iter_csv_writer.writerow({
                            "idx": i,
                            "name": name,
                            "method": "Base",
                            "anneal": "-",
                            **st,
                        })

            
            if opt_cost is not None and opt_cost > 1e-6:
                 gap = (base_best - opt_cost) / opt_cost
                 results["base_gap"].append(gap)
                 results["opt_cost"].append(opt_cost)
            


        # Model
        model_best = None
        model_best_na = None
        mix_best = None
        mix_best_na = None
        model_m = None
        mix_m = None
        if model:
              # Determine which methods to run based on problem type and flags
              # Default: Mix(anneal) for TSP, Model(anneal) for CVRP
              run_model_anneal = args.run_model_anneal
              run_model_no_anneal = args.run_model_no_anneal
              run_mix_anneal = args.run_mix_anneal and args.warmup
              run_mix_no_anneal = args.run_mix_no_anneal and args.warmup

              if not run_model_anneal and not run_model_no_anneal and not run_mix_anneal and not run_mix_no_anneal:
                  if args.problem == 'tsp':
                      if args.H < 50:
                          run_model_anneal = True
                      else:
                          run_mix_anneal = True
                  else:
                      if args.H < 50:
                          run_model_no_anneal = True
                      else:
                          run_mix_no_anneal = True
              
              args_anneal = _clone_args(args, no_anneal=False)
              args_noanneal = _clone_args(args, no_anneal=True)
              collect_guidance_metrics = bool(args.visualize or args.collect_guidance_metrics)

              # Model (anneal ON)
              if run_model_anneal:
                  tm0 = time.time()
                  mod_ret = infer_instance(
                      args.problem, MFACO, build_fn, model, item,
                      args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                      args_anneal,
                      use_heuristic_only=False,
                      collect_metrics=collect_guidance_metrics,
                      metrics_every_step=collect_guidance_metrics,
                      seed=args.seed + i,
                      ablation_pheromone=args.ablation_pheromone_features,
                      ablation_incumbent=args.ablation_incumbent_features
                  )

                  tm1 = time.time()
                  _, model_best, mod_timings, mod_extra = mod_ret
                  model_m = mod_extra.get("metrics")
                  model_iter_stats = mod_extra.get("iter_stats")
                  results["model_cost"].append(model_best)
                  results["model_time"].append(tm1 - tm0)
                  results["model_neural_time"].append(mod_timings.get("time_neural", 0.0))
                  _append_stage_metrics("model", mod_extra.get("stage_metrics"))
                  _append_guidance_metrics("model", mod_extra.get("metrics"))

                  if args.iter_log and iter_csv_writer is not None and model_iter_stats is not None:
                      for st in model_iter_stats:
                          iter_csv_writer.writerow({
                              "idx": i,
                              "name": name,
                              "method": "Model",
                              "anneal": "on",
                              **st,
                          })

                  
                  if mod_timings:
                      if "time_neural" in mod_timings: results["model_time_breakdown"]["neural"].append(mod_timings["time_neural"])
                      if "time_sampling" in mod_timings: results["model_time_breakdown"]["sampling"].append(mod_timings["time_sampling"])
                      if "time_ls" in mod_timings: results["model_time_breakdown"]["ls"].append(mod_timings["time_ls"])
                      if "time_update" in mod_timings: results["model_time_breakdown"]["update"].append(mod_timings["time_update"])
                  
                  if opt_cost is not None and opt_cost > 1e-6:
                      gap = (model_best - opt_cost) / opt_cost
                      results["model_gap"].append(gap)
                  
                  _record_history("model", mod_extra.get("history", []), results)

              # Model (anneal OFF)
              if run_model_no_anneal:
                  tm0 = time.time()
                  mod_ret_na = infer_instance(
                      args.problem, MFACO, build_fn, model, item,
                      args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                      args_noanneal,
                      use_heuristic_only=False,
                      collect_metrics=collect_guidance_metrics,
                      metrics_every_step=collect_guidance_metrics,
                      seed=args.seed + i,
                      ablation_pheromone=args.ablation_pheromone_features,
                      ablation_incumbent=args.ablation_incumbent_features
                  )
                  tm1 = time.time()
                  _, model_best_na, mod_na_timings, mod_na_extra = mod_ret_na
                  model_na_iter_stats = mod_na_extra.get("iter_stats")
                  results["model_cost_no_anneal"].append(model_best_na)
                  results["model_time_no_anneal"].append(tm1 - tm0)
                  results["model_neural_time_no_anneal"].append(mod_na_timings.get("time_neural", 0.0))
                  _append_stage_metrics("model_no_anneal", mod_na_extra.get("stage_metrics"))
                  _append_guidance_metrics("model_no_anneal", mod_na_extra.get("metrics"))

                  if args.iter_log and iter_csv_writer is not None and model_na_iter_stats is not None:
                      for st in model_na_iter_stats:
                          iter_csv_writer.writerow({
                              "idx": i,
                              "name": name,
                              "method": "Model",
                              "anneal": "off",
                              **st,
                          })
                  # if args.iter_print and model_na_iter_stats is not None:
                  #     for st in model_na_iter_stats:
                  #         print(f"{name} Model(no_anneal) iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
                  
                  if opt_cost is not None and opt_cost > 1e-6:
                      gap_na = (model_best_na - opt_cost) / opt_cost
                      results["model_gap_no_anneal"].append(gap_na)
                  
                  # If anneal run didn't happen, use no_anneal metrics for visualization
                  if model_m is None:
                      model_m = mod_na_extra.get("metrics")
                  
                  _record_history("model_no_anneal", mod_na_extra.get("history", []), results)
                          # Mix methods (only if warmup is enabled)
              if args.warmup:
                  inject_step = int(args.H * args.warmup_ratio)
                  
                  # Mix (anneal ON)
                  if run_mix_anneal:
                      tmi0 = time.time()
                      mix_ret = infer_instance(
                          args.problem, MFACO, build_fn, model, item,
                          args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                          args_anneal,
                          use_heuristic_only=False,
                          collect_metrics=collect_guidance_metrics,
                          metrics_every_step=collect_guidance_metrics,
                          inject_step=inject_step,
                          seed=args.seed + i,
                          ablation_pheromone=args.ablation_pheromone_features,
                          ablation_incumbent=args.ablation_incumbent_features
                      )
                      tmi1 = time.time()
                      _, mix_best, mix_timings, mix_extra = mix_ret
                      mix_m = mix_extra.get("metrics")
                      mix_iter_stats = mix_extra.get("iter_stats")
                      results["mix_cost"].append(mix_best)
                      results["mix_time"].append(tmi1 - tmi0)
                      results["mix_neural_time"].append(mix_timings.get("time_neural", 0.0))
                      _append_stage_metrics("mix", mix_extra.get("stage_metrics"))
                      _append_guidance_metrics("mix", mix_extra.get("metrics"))

                      if args.iter_log and iter_csv_writer is not None and mix_iter_stats is not None:
                          for st in mix_iter_stats:
                              iter_csv_writer.writerow({
                                  "idx": i,
                                  "name": name,
                                  "method": "Mix",
                                  "anneal": "on",
                                  **st,
                              })
                      # if args.iter_print and mix_iter_stats is not None:
                      #     for st in mix_iter_stats:
                      #         print(f"{name} Mix(anneal) iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
                      
                      if opt_cost is not None and opt_cost > 1e-6:
                          gap = (mix_best - opt_cost) / opt_cost
                          results["mix_gap"].append(gap)
                      
                      _record_history("mix", mix_extra.get("history", []), results)

                  # Mix (anneal OFF)
                  if run_mix_no_anneal:
                      tmi0 = time.time()
                      mix_ret_na = infer_instance(
                          args.problem, MFACO, build_fn, model, item,
                          args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                          args_noanneal,
                          use_heuristic_only=False,
                          collect_metrics=False,
                          metrics_every_step=False,
                          inject_step=inject_step,
                          seed=args.seed + i,
                          ablation_pheromone=args.ablation_pheromone_features,
                          ablation_incumbent=args.ablation_incumbent_features
                      )
                      tmi1 = time.time()
                      _, mix_best_na, mix_na_timings, mix_na_extra = mix_ret_na
                      mix_na_iter_stats = mix_na_extra.get("iter_stats")
                      results["mix_cost_no_anneal"].append(mix_best_na)
                      results["mix_time_no_anneal"].append(tmi1 - tmi0)
                      results["mix_neural_time_no_anneal"].append(mix_na_timings.get("time_neural", 0.0))
                      _append_stage_metrics("mix_no_anneal", mix_na_extra.get("stage_metrics"))

                      if args.iter_log and iter_csv_writer is not None and mix_na_iter_stats is not None:
                          for st in mix_na_iter_stats:
                              iter_csv_writer.writerow({
                                  "idx": i,
                                  "name": name,
                                  "method": "Mix",
                                  "anneal": "off",
                                  **st,
                              })
                      # if args.iter_print and mix_na_iter_stats is not None:
                      #     for st in mix_na_iter_stats:
                      #         print(f"{name} Mix(no_anneal) iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
                      
                      if opt_cost is not None and opt_cost > 1e-6:
                          gap_na = (mix_best_na - opt_cost) / opt_cost
                          results["mix_gap_no_anneal"].append(gap_na)

                      _record_history("mix_no_anneal", mix_na_extra.get("history", []), results)
        
        if args.visualize:
            if i == 0:
                 # Initialize as lists to store per-instance metrics for mean/std
                 if base_m: 
                     results["base_metrics_list"] = {k: [] for k,v in base_m.items() if k != "snapshots"}
                 
                 if model and model_m: 
                     results["model_metrics_list"] = {k: [] for k,v in model_m.items() if k != "snapshots"}
                 
                 if model and args.warmup and mix_m: 
                     results["mix_metrics_list"] = {k: [] for k,v in mix_m.items() if k != "snapshots"}
            
            if base_m and "base_metrics_list" in results:
                 for k,v in base_m.items():
                    if k == "snapshots": continue
                    if k in results["base_metrics_list"]:
                         results["base_metrics_list"][k].append(np.array(v))
                              
            if model and model_m and "model_metrics_list" in results:
                 for k,v in model_m.items():
                    if k == "snapshots": continue
                    if k in results["model_metrics_list"]:
                        results["model_metrics_list"][k].append(np.array(v))

            if model and args.warmup and mix_m and "mix_metrics_list" in results:
                 for k,v in mix_m.items():
                    if k == "snapshots": continue
                    if k in results["mix_metrics_list"]:
                        results["mix_metrics_list"][k].append(np.array(v))

            if args.visualize and i == 0:
                 # Capture snapshots from i=0
                 if base_m and "snapshots" in base_m: sample_snapshots["base"] = base_m["snapshots"]
                 if model and model_m and "snapshots" in model_m: sample_snapshots["model"] = model_m["snapshots"]
                 if model and args.warmup and mix_m and "snapshots" in mix_m: sample_snapshots["mix"] = mix_m["snapshots"]

        # Output cost for each instance
        if True:  # Always print per-instance results
             def _metric_segment(label, value, elapsed_s=None):
                 if value is None:
                     return None
                 segment = f"{label} {value:.4f}"
                 if opt_cost is not None and opt_cost > 1e-6:
                     gap_pct = (value - opt_cost) / opt_cost * 100
                     segment += f" ({gap_pct:+.2f}%)"
                 if elapsed_s is not None:
                     segment += f" ({float(elapsed_s):.2f}s)"
                 return segment

             segments = [f"[{i + 1}/{total_instances}] {name}"]
             if opt_cost is not None:
                 segments.append(f"opt {opt_cost:.4f}")

             if baseline_values is not None and len(baseline_values) > i:
                 segments.append(f"bl {float(baseline_values[i]):.4f}")
             
             if results.get("base_cost") and len(results["base_cost"]) > i:
                 elapsed_s = results["base_time"][-1] if results.get("base_time") else None
                 seg = _metric_segment("base", results["base_cost"][-1], elapsed_s)
                 if seg:
                     segments.append(seg)

             if results.get("model_cost") and len(results["model_cost"]) > i:
                 elapsed_s = results["model_time"][-1] if results.get("model_time") else None
                 seg = _metric_segment("model", results["model_cost"][-1], elapsed_s)
                 if seg:
                     segments.append(seg)

             if results.get("model_cost_no_anneal") and len(results["model_cost_no_anneal"]) > i:
                 elapsed_s = results["model_time_no_anneal"][-1] if results.get("model_time_no_anneal") else None
                 seg = _metric_segment("model-na", results["model_cost_no_anneal"][-1], elapsed_s)
                 if seg:
                     segments.append(seg)

             if results.get("mix_cost") and len(results["mix_cost"]) > i:
                 elapsed_s = results["mix_time"][-1] if results.get("mix_time") else None
                 seg = _metric_segment("mix", results["mix_cost"][-1], elapsed_s)
                 if seg:
                     segments.append(seg)

             if results.get("mix_cost_no_anneal") and len(results["mix_cost_no_anneal"]) > i:
                 elapsed_s = results["mix_time_no_anneal"][-1] if results.get("mix_time_no_anneal") else None
                 seg = _metric_segment("mix-na", results["mix_cost_no_anneal"][-1], elapsed_s)
                 if seg:
                     segments.append(seg)

             tqdm.write(" | ".join(segments))

        # Per-instance CSV logging (always capture if --log)
        if args.log:
            bl_i = float(baseline_values[i]) if (baseline_values is not None and len(baseline_values) > i) else None
            row_dict = {
                "idx": i,
                "name": name,
                "opt": (float(opt_cost) if opt_cost is not None else None),
                "baseline": bl_i,
                "base": (float(base_best) if base_best is not None else None),
                "model_anneal": (float(model_best) if model_best is not None else None),
                "model_no_anneal": (float(model_best_na) if model_best_na is not None else None),
                "mix_anneal": (float(mix_best) if mix_best is not None else None),
                "mix_no_anneal": (float(mix_best_na) if mix_best_na is not None else None),
            }
            # Add iteration metrics
            for itr in TARGET_ITERS:
                 def _get_val(k, idx):
                     lst = results.get(k)
                     if lst and len(lst) > idx: return lst[idx]
                     return None
                 
                 row_dict[f"base_I{itr}"] = _get_val(f"base_cost_I{itr}", i)
                 row_dict[f"base_time_I{itr}"] = _get_val(f"base_time_I{itr}", i)
                 
                 row_dict[f"model_anneal_I{itr}"] = _get_val(f"model_cost_I{itr}", i)
                 row_dict[f"model_anneal_time_I{itr}"] = _get_val(f"model_time_I{itr}", i)
                 
                 row_dict[f"model_no_anneal_I{itr}"] = _get_val(f"model_no_anneal_cost_I{itr}", i)
                 row_dict[f"model_no_anneal_time_I{itr}"] = _get_val(f"model_no_anneal_time_I{itr}", i)

                 row_dict[f"mix_anneal_I{itr}"] = _get_val(f"mix_cost_I{itr}", i)
                 row_dict[f"mix_anneal_time_I{itr}"] = _get_val(f"mix_time_I{itr}", i)

                 row_dict[f"mix_no_anneal_I{itr}"] = _get_val(f"mix_no_anneal_cost_I{itr}", i)
                 row_dict[f"mix_no_anneal_time_I{itr}"] = _get_val(f"mix_no_anneal_time_I{itr}", i)

            per_instance_rows.append(row_dict)

    if iter_csv_f is not None:
        try:
            iter_csv_f.close()
        except Exception:
            pass

    def _mean_std(xs):
        if xs is None or len(xs) == 0:
            return None, None
        # Filter None
        valid = [x for x in xs if x is not None]
        if not valid: return None, None
        arr = np.array(valid, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    def _fmt(x, nd=4, empty="-"):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return empty
        return f"{x:.{nd}f}"

    def _stage_means(prefix):
        summary = {}
        for metric_key in stage_metric_keys:
            summary[metric_key] = _mean_std(results.get(f"{prefix}_{metric_key}", []))[0]
        return summary

    # Save base cache if we computed it this run
    # Save base cache if computed and not loaded
    if (not args.no_baseline) and (cached_base_costs is None):
        # We computed it fresh. Save it.
        try:
             cache_data = {
                 "costs": list(results["base_cost"]),
                 "avg_last": float(np.mean(results["base_cost"])), # Approximate for test
                 "avg_best": float(np.mean(results["base_cost"])),
                 "metrics": {} # Test doesn't collect detailed metrics usually?
             }
             utils.save_pure_mfaco_cache(args, val_list, cache_data)
             print("Saved base costs to shared cache.")
        except Exception as e:
            print(f"Warning: failed to save base cache: {e}")

    # Summary tables
    leaderboard_entries = []
    summary_rows = []

    def _mean_gap_to_baseline(cost_list, baseline_arr):
        if baseline_arr is None or cost_list is None:
            return None
        if len(cost_list) == 0 or len(baseline_arr) == 0:
            return None
        if len(cost_list) != len(baseline_arr):
            return None
        c = np.array(cost_list, dtype=float)
        b = np.array(baseline_arr, dtype=float)
        ok = np.isfinite(c) & np.isfinite(b) & (b > 1e-12)
        if not ok.any():
            return None
        gaps = (c[ok] - b[ok]) / b[ok]
        return gaps

    def _paired_delta_vs_base(cost_list):
        base_costs = results.get("base_cost", [])
        if not base_costs or not cost_list or len(cost_list) != len(base_costs):
            return None, None
        c = np.array(cost_list, dtype=float)
        b = np.array(base_costs, dtype=float)
        ok = np.isfinite(c) & np.isfinite(b) & (b > 1e-12)
        if not ok.any():
            return None, None
        delta_pct = ((c[ok] - b[ok]) / b[ok]) * 100.0
        beat_base_pct = float(np.mean(c[ok] < b[ok]) * 100.0)
        return float(np.mean(delta_pct)), beat_base_pct

    def _add_method_row(name, cost_list, time_list, gap_list=None, neural_time_list=None):
        m, s = _mean_std(cost_list)
        if m is None:
            return
        tm, _ = _mean_std(time_list)
        
        # Calculate A+B time format if neural_time_list is provided
        # Filter None from time_list for total sum
        valid_times = [t for t in time_list if t is not None]
        total_time_str = _fmt(np.sum(valid_times)) if valid_times else "-"
        
        time_str = _fmt(tm)
        if neural_time_list is not None:
             nm, _ = _mean_std(neural_time_list)
             valid_neural = [t for t in neural_time_list if t is not None]
             nt_sum = np.sum(valid_neural) if valid_neural else 0.0
             
             if tm is not None and nm is not None:
                 cpu_time = max(0.0, tm - nm)
                 # Total CPU time = Total Time - Total Neural Time
                 total_time_val = np.sum(valid_times) if valid_times else 0.0
                 cpu_total = max(0.0, total_time_val - nt_sum)
                 time_str = f"{cpu_time:.2f}+{nm:.2f}"
                 total_time_str = f"{cpu_total:.2f}+{nt_sum:.2f}"
        
        # Prefer gap to Opt (from txt dataset optimal values) when available.
        gap_ref = "-"
        gap_val = None
        gap_std = None
        
        if gap_list is not None and len(gap_list) > 0:
            gap_arr = np.array(gap_list) * 100.0
            gap_val = float(np.mean(gap_arr))
            gap_std = float(np.std(gap_arr))
            gap_ref = "opt"
        else:
            gaps = _mean_gap_to_baseline(cost_list, baseline_values)
            if gaps is not None:
                gap_arr = gaps * 100.0
                gap_val = float(np.mean(gap_arr))
                gap_std = float(np.std(gap_arr))
                gap_ref = "bl"

        delta_base_pct = None
        beat_base_pct = None
        if name != "Base":
            delta_base_pct, beat_base_pct = _paired_delta_vs_base(cost_list)

        leaderboard_entries.append({
            "name": name,
            "mean_cost": m,
            "std_cost": s,
            "gap_val": gap_val,
            "gap_std": gap_std,
            "gap_ref": gap_ref,
            "mean_time": time_str,
            "total_time": total_time_str,
            "delta_base_pct": delta_base_pct,
            "beat_base_pct": beat_base_pct,
        })
        summary_rows.append([
            name,
            _fmt(m),
            _fmt(s),
            _fmt(gap_val, nd=2),
            _fmt(gap_std, nd=2),
            gap_ref,
            time_str,
            total_time_str,
            "",
        ])

    if (not args.no_baseline) and results.get("base_cost"):
        _add_method_row("Base", results["base_cost"], results.get("base_time", []), results.get("base_gap", []))

    if model:
        # Only add rows for methods that were actually executed (non-empty results)
        if results.get("model_cost") and len(results["model_cost"]) > 0:
            _add_method_row("Model(anneal)", results["model_cost"], results.get("model_time", []), results.get("model_gap", []), neural_time_list=results.get("model_neural_time"))

        if results.get("model_cost_no_anneal") and len(results["model_cost_no_anneal"]) > 0:
            _add_method_row("Model(no_anneal)", results["model_cost_no_anneal"], results.get("model_time_no_anneal", []), results.get("model_gap_no_anneal", []), neural_time_list=results.get("model_neural_time_no_anneal"))

        if args.warmup:
            if results.get("mix_cost") and len(results["mix_cost"]) > 0:
                _add_method_row("Mix(anneal)", results["mix_cost"], results.get("mix_time", []), results.get("mix_gap", []), neural_time_list=results.get("mix_neural_time"))

            if results.get("mix_cost_no_anneal") and len(results["mix_cost_no_anneal"]) > 0:
                _add_method_row("Mix(no_anneal)", results["mix_cost_no_anneal"], results.get("mix_time_no_anneal", []), results.get("mix_gap_no_anneal", []), neural_time_list=results.get("mix_neural_time_no_anneal"))

    # Add iteration-checkpoint rows
    for itr in TARGET_ITERS:
         # Base
         if (not args.no_baseline) and results.get(f"base_cost_I{itr}"):
             _add_method_row(f"Base@I{itr}", results[f"base_cost_I{itr}"], results.get(f"base_time_I{itr}", []))
         
         # Model
         if results.get(f"model_cost_I{itr}"):
             _add_method_row(f"Model(anneal)@I{itr}", results[f"model_cost_I{itr}"], results.get(f"model_time_I{itr}", []), neural_time_list=results.get(f"model_neural_time_I{itr}"))
         if results.get(f"model_no_anneal_cost_I{itr}"):
             _add_method_row(f"Model(no_anneal)@I{itr}", results[f"model_no_anneal_cost_I{itr}"], results.get(f"model_no_anneal_time_I{itr}", []), neural_time_list=results.get(f"model_no_anneal_neural_time_I{itr}"))
         
         # Mix
         if args.warmup:
             if results.get(f"mix_cost_I{itr}"):
                 _add_method_row(f"Mix(anneal)@I{itr}", results[f"mix_cost_I{itr}"], results.get(f"mix_time_I{itr}", []), neural_time_list=results.get(f"mix_neural_time_I{itr}"))
             if results.get(f"mix_no_anneal_cost_I{itr}"):
                 _add_method_row(f"Mix(no_anneal)@I{itr}", results[f"mix_no_anneal_cost_I{itr}"], results.get(f"mix_no_anneal_time_I{itr}", []), neural_time_list=results.get(f"mix_no_anneal_neural_time_I{itr}"))

    reference_rows = []
    if opt_costs_summary:
        opt_mean, opt_std = _mean_std(opt_costs_summary)
        reference_rows.append(["Opt", _fmt(opt_mean), _fmt(opt_std), "Ground-truth / known target"])
    if baseline_values is not None:
        bl_mean, bl_std = _mean_std(baseline_values)
        reference_rows.append(["Baseline", _fmt(bl_mean), _fmt(bl_std), "External baseline solver"])

    main_entries = [entry for entry in leaderboard_entries if "@I" not in entry["name"]]
    checkpoint_entries = [entry for entry in leaderboard_entries if "@I" in entry["name"]]

    if main_entries:
        ranked_entries = sorted(main_entries, key=lambda entry: entry["mean_cost"])
        best_name = ranked_entries[0]["name"] if ranked_entries else None
        if best_name is not None:
            for row in summary_rows:
                if row[0] == best_name:
                    row[-1] = "BEST"
                    break

        leaderboard_rows = []
        for rank, entry in enumerate(ranked_entries, start=1):
            best_flag = "BEST" if entry["name"] == best_name else ""
            leaderboard_rows.append([
                str(rank),
                entry["name"],
                _fmt(entry["mean_cost"]),
                _fmt(entry["std_cost"]),
                _fmt(entry["gap_val"], nd=2),
                entry["gap_ref"],
                _fmt(entry["delta_base_pct"], nd=2),
                _fmt(entry["beat_base_pct"], nd=1),
                entry["mean_time"],
                best_flag,
            ])

        _print_section("Quality leaderboard")
        _print_table(
            leaderboard_rows,
            headers=["Rank", "Method", "MeanCost", "StdCost", "Gap%", "Ref", "ΔBase%", "BeatBase%", "MeanTime", "Best"],
        )

    if reference_rows:
        _print_section("Reference costs")
        _print_table(reference_rows, headers=["Reference", "MeanCost", "StdCost", "Notes"])

    if checkpoint_entries:
        checkpoint_map = {}
        for entry in checkpoint_entries:
            method_name, itr_label = entry["name"].split("@", 1)
            checkpoint_map.setdefault(method_name, {})[itr_label] = entry

        checkpoint_rows = []
        checkpoint_headers = ["Method"] + [f"I{itr}" for itr in TARGET_ITERS]
        for method_name in sorted(checkpoint_map.keys()):
            row = [method_name]
            has_any = False
            for itr in TARGET_ITERS:
                itr_key = f"I{itr}"
                entry = checkpoint_map[method_name].get(itr_key)
                cell = "-"
                if entry is not None and entry["mean_cost"] is not None:
                    cell = _fmt(entry["mean_cost"])
                    has_any = True
                row.append(cell)
            if has_any:
                checkpoint_rows.append(row)

        if checkpoint_rows:
            _print_section("Checkpoint mean cost")
            _print_table(checkpoint_rows, headers=checkpoint_headers)

    diagnostics_rows = []

    def _add_stage_diagnostics(label, prefix, timing_breakdown=None):
        stage_summary = _stage_means(prefix)
        before = stage_summary["avg_ant_before_ls"]
        after = stage_summary["avg_ant_after_ls"]
        final = stage_summary["avg_incumbent_after_aco"]
        if before is None and after is None and final is None and timing_breakdown is None:
            return

        stage_text = "-"
        if before is not None or after is not None or final is not None:
            stage_text = f"before={_fmt(before)} | after={_fmt(after)} | final={_fmt(final)}"

        timing_text = "-"
        if timing_breakdown and timing_breakdown["neural"]:
            t_nn = np.sum(timing_breakdown["neural"])
            t_samp = np.sum(timing_breakdown["sampling"])
            t_ls = np.sum(timing_breakdown["ls"])
            t_upd = np.sum(timing_breakdown["update"])
            t_total_calc = t_nn + t_samp + t_ls + t_upd
            if t_total_calc > 1e-9:
                timing_text = (
                    f"nn={t_nn / t_total_calc * 100:.1f}% | "
                    f"samp={t_samp / t_total_calc * 100:.1f}% | "
                    f"ls={t_ls / t_total_calc * 100:.1f}% | "
                    f"upd={t_upd / t_total_calc * 100:.1f}%"
                )

        diagnostics_rows.append([label, stage_text, timing_text])

    _add_stage_diagnostics("Base", "base")
    if model:
        _add_stage_diagnostics("Model(anneal)", "model", results["model_time_breakdown"] if results.get("model_cost") else None)
        _add_stage_diagnostics("Model(no_anneal)", "model_no_anneal")
        _add_stage_diagnostics("Mix(anneal)", "mix")
        _add_stage_diagnostics("Mix(no_anneal)", "mix_no_anneal")

    if diagnostics_rows:
        _print_section("Diagnostics")
        _print_table(diagnostics_rows, headers=["Method", "Stage means", "Timing split"])

    # CSV summary logging
    if summary_rows and args.log and csv_path_summary is not None:
        try:
            with open(csv_path_summary, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["problem", args.problem])
                writer.writerow(["n_node", args.n_node])
                writer.writerow(["checkpoint", args.checkpoint])
                writer.writerow(["dataset", args.dataset])
                writer.writerow(["seed", args.seed])
                writer.writerow([])
                writer.writerow(["Method", "MeanCost", "StdCost", "Gap%", "StdGap%", "GapRef", "MeanTime", "TotalTime", "Best"])
                for r in summary_rows:
                    writer.writerow(r)
        except Exception as e:
            print(f"Warning: failed to write CSV summary: {e}")

    def _sum_valid(values):
        return float(np.sum([v for v in values if v is not None])) if values else None

    def _flatten_metric_values(payloads, key):
        vals = []
        for payload in payloads:
            for value in (payload or {}).get(key, []) or []:
                if isinstance(value, (list, tuple)):
                    vals.extend(v for v in value if v is not None)
                elif value is not None:
                    vals.append(value)
        arr = np.array(vals, dtype=float) if vals else np.array([], dtype=float)
        arr = arr[np.isfinite(arr)]
        return arr

    def _head_count(payloads):
        for payload in payloads:
            counts_list = (payload or {}).get("head_counts", []) or []
            for counts in counts_list:
                if counts:
                    return len(counts)
        return 0

    def _head_fraction(payloads, key, n_heads):
        counts = np.zeros(n_heads, dtype=float)
        total = 0.0
        for payload in payloads:
            for head in (payload or {}).get(key, []) or []:
                try:
                    h = int(head)
                except Exception:
                    continue
                if 0 <= h < n_heads:
                    counts[h] += 1.0
                    total += 1.0
        if total <= 0:
            return None
        return [float(x / total) for x in counts]

    def _head_vector_mean(payloads, key, n_heads):
        rows = []
        for payload in payloads:
            for row in (payload or {}).get(key, []) or []:
                if not isinstance(row, (list, tuple)) or len(row) != n_heads:
                    continue
                vals = [np.nan if v is None else float(v) for v in row]
                rows.append(vals)
        if not rows:
            return None
        arr = np.array(rows, dtype=float)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(arr, axis=0)
        return [None if not np.isfinite(v) else float(v) for v in means]

    def _guidance_diagnostics(prefix):
        payloads = results.get(f"{prefix}_metrics_payload", [])
        out = {}
        for key in ["enhance", "rebellion", "suppression", "head_logit_corr", "head_topk_overlap"]:
            vals = _flatten_metric_values(payloads, key)
            out[key] = float(vals.mean()) if vals.size else None
        n_heads = _head_count(payloads)
        if n_heads:
            out["num_heads"] = int(n_heads)
            out["head_best_fraction"] = _head_fraction(payloads, "head_best", n_heads)
            out["head_improvement_fraction"] = _head_fraction(payloads, "head_improvement", n_heads)
            out["head_mean_after"] = _head_vector_mean(payloads, "head_mean_after", n_heads)
            out["head_best_after"] = _head_vector_mean(payloads, "head_best_after", n_heads)
            out["head_mean_before"] = _head_vector_mean(payloads, "head_mean_before", n_heads)
            out["head_best_before"] = _head_vector_mean(payloads, "head_best_before", n_heads)
            out["head_enhance"] = _head_vector_mean(payloads, "head_enhance", n_heads)
            out["head_rebellion"] = _head_vector_mean(payloads, "head_rebellion", n_heads)
            out["head_suppression"] = _head_vector_mean(payloads, "head_suppression", n_heads)
        return {k: v for k, v in out.items() if v is not None}

    def _build_method_summary(prefix, cost_key, time_key, gap_key=None, neural_time_key=None):
        costs = results.get(cost_key, [])
        mean_cost, std_cost = _mean_std(costs)
        if mean_cost is None:
            return None

        times = results.get(time_key, [])
        mean_time_s, std_time_s = _mean_std(times)
        payload = {
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "mean_time_s": mean_time_s,
            "std_time_s": std_time_s,
            "total_time_s": _sum_valid(times),
            "gap_pct": None,
        }

        gaps = results.get(gap_key, []) if gap_key else []
        if gaps:
            gap_mean, gap_std = _mean_std([g * 100.0 for g in gaps])
            payload["gap_pct"] = gap_mean
            payload["gap_std_pct"] = gap_std

        if neural_time_key:
            neural_times = results.get(neural_time_key, [])
            neural_mean, neural_std = _mean_std(neural_times)
            payload["mean_neural_time_s"] = neural_mean
            payload["std_neural_time_s"] = neural_std
            payload["total_neural_time_s"] = _sum_valid(neural_times)

        guidance_mean, _ = _mean_std(results.get(f"{prefix}_mean_guidance", []))
        pher_corr_mean, _ = _mean_std(results.get(f"{prefix}_pheromone_correlation", []))
        payload["mean_guidance"] = guidance_mean
        payload["pheromone_correlation"] = pher_corr_mean
        payload["guidance_diagnostics"] = _guidance_diagnostics(prefix)

        payload.update(_stage_means(prefix))
        return payload

    if args.summary_json:
        summary_payload = {
            "problem": args.problem,
            "n_node": int(args.n_node),
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "seed": int(args.seed),
            "methods": {},
        }
        method_specs = [
            ("base", "base", "base_cost", "base_time", "base_gap", None),
            ("model_anneal", "model", "model_cost", "model_time", "model_gap", "model_neural_time"),
            ("model_no_anneal", "model_no_anneal", "model_cost_no_anneal", "model_time_no_anneal", "model_gap_no_anneal", "model_neural_time_no_anneal"),
            ("mix_anneal", "mix", "mix_cost", "mix_time", "mix_gap", "mix_neural_time"),
            ("mix_no_anneal", "mix_no_anneal", "mix_cost_no_anneal", "mix_time_no_anneal", "mix_gap_no_anneal", "mix_neural_time_no_anneal"),
        ]
        for method_name, prefix, cost_key, time_key, gap_key, neural_time_key in method_specs:
            payload = _build_method_summary(prefix, cost_key, time_key, gap_key, neural_time_key)
            if payload is not None:
                summary_payload["methods"][method_name] = payload
        try:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Wrote summary JSON: {summary_path}")
        except Exception as e:
            print(f"Warning: failed to write summary JSON {args.summary_json}: {e}")

    # CSV instances logging (write once at end)
    if args.log and csv_path_instances is not None and per_instance_rows:
        try:
            fieldnames = [
                "idx",
                "name",
                "opt",
                "baseline",
                "base",
                "model_anneal",
                "model_no_anneal",
                "mix_anneal",
                "mix_no_anneal",
            ]
            for itr in TARGET_ITERS:
                fieldnames.extend([
                    f"base_I{itr}", f"base_time_I{itr}",
                    f"model_anneal_I{itr}", f"model_anneal_time_I{itr}",
                    f"model_no_anneal_I{itr}", f"model_no_anneal_time_I{itr}",
                    f"mix_anneal_I{itr}", f"mix_anneal_time_I{itr}",
                    f"mix_no_anneal_I{itr}", f"mix_no_anneal_time_I{itr}",
                ])
            with open(csv_path_instances, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_instance_rows)
        except Exception as e:
            print(f"Warning: failed to write CSV instances: {e}")

    if args.visualize:
        out = Path(args.visualize_output)
        out.mkdir(parents=True, exist_ok=True)
        # Increase font sizes for 2-column paper readability
        plt.rcParams.update({
            'font.family': 'serif', 'font.size': 28, 'axes.titlesize': 32,
            'axes.labelsize': 30, 'xtick.labelsize': 26, 'ytick.labelsize': 26,
            'legend.fontsize': 24, 'lines.linewidth': 4, 'lines.markersize': 0,
            'figure.titlesize': 34
        })

        N = len(val_list)
        
        # Helper to compute mean and std from list of arrays
        def compute_mean_std(arr_list):
            if not arr_list:
                return None, None
            # Pad to same length
            max_len = max(len(a) for a in arr_list)
            padded = np.array([np.pad(a, (0, max_len - len(a)), mode='edge') for a in arr_list])
            return padded.mean(axis=0), padded.std(axis=0)
        
        # Helper to plot with shaded std
        def plot_with_shade(ax, mean, std, label, color=None):
            x = np.arange(len(mean))
            line, = ax.plot(x, mean, label=label, color=color)
            if std is not None:
                ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())
            return line
        
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        if "base_metrics_list" in results and results["base_metrics_list"].get("cost"):
            mean, std = compute_mean_std(results["base_metrics_list"]["cost"])
            plot_with_shade(ax, mean, std, "Base", colors[color_idx])
            color_idx += 1
        
        if model and "model_metrics_list" in results and results["model_metrics_list"].get("cost"):
            mean, std = compute_mean_std(results["model_metrics_list"]["cost"])
            plot_with_shade(ax, mean, std, "Model", colors[color_idx])
            color_idx += 1
            
        if model and args.warmup and "mix_metrics_list" in results and results["mix_metrics_list"].get("cost"):
            mean, std = compute_mean_std(results["mix_metrics_list"]["cost"])
            plot_with_shade(ax, mean, std, "Mix", colors[color_idx])
            color_idx += 1
            
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "cost.pdf")
        plt.close()
        
        # Prior changes plot
        if model and "model_metrics_list" in results:
            mod_list = results["model_metrics_list"]
            if mod_list.get("l2"):
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                for k_idx, k in enumerate(["l2", "turnover", "flip"]):
                    if mod_list.get(k):
                        mean, std = compute_mean_std(mod_list[k])
                        plot_with_shade(ax, mean, std, k, colors[k_idx])
                plt.xlabel("Iteration")
                plt.ylabel("Value")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out / "prior_changes.pdf")
                plt.close()

            if mod_list.get("enhance"):
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                for k_idx, k in enumerate(["enhance", "suppression"]):
                    if mod_list.get(k):
                        mean, std = compute_mean_std(mod_list[k])
                        plot_with_shade(ax, mean, std, k, colors[k_idx])
                plt.title("Model-Pheromone Interaction")
                plt.xlabel("Iteration")
                plt.ylabel("Rate")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out / "interaction.pdf")
                plt.close()

        if sample_snapshots:
            print("Plotting matrix snapshots...")
            for mode, snaps in sample_snapshots.items():
                for snap in snaps:
                    t = snap["t"]
                    pher = snap.get("pheromone")
                    neural_prior = snap.get("neural_prior")
                    
                    # Columns: Pheromone, [Neural Prior]
                    ncols = 1
                    if neural_prior is not None: ncols += 1
                    
                    width = 10 * ncols
                    height = 10
                    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
                    if ncols == 1: axes = [axes]
                    
                    MAX_ROWS = 32
                    
                    # Helper for safe plotting
                    def safe_plot_heatmap(ax, tensor, title, cmap):
                        # Truncate to MAX_ROWS
                        if tensor.shape[0] > MAX_ROWS:
                            tensor = tensor[:MAX_ROWS]
                            title += f" (first {MAX_ROWS} rows)"
                            
                        arr = tensor.numpy()
                        # Handle NaNs/Infs
                        if not np.isfinite(arr).all():
                            arr = np.nan_to_num(arr, nan=0.0, posinf=np.nanmax(arr[np.isfinite(arr)]), neginf=np.nanmin(arr[np.isfinite(arr)]))
                        
                        # Handle constant values to avoid norm errors
                        vmin, vmax = arr.min(), arr.max()
                        if math.isclose(vmin, vmax):
                            vmax = vmin + 1e-6

                        im = ax.imshow(arr, aspect='equal', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
                        ax.set_title(title)
                        ax.set_xlabel("Neighbor Rank")
                        ax.set_ylabel("Node Index")
                        fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.046, pad=0.04)

                    # 1. Pheromone
                    safe_plot_heatmap(axes[0], pher, "Pheromone", 'viridis')
                    
                    # 2. Neural Prior
                    if neural_prior is not None:
                        safe_plot_heatmap(axes[1], neural_prior, "Guidance", 'inferno')
                    
                    plt.tight_layout()
                    plt.savefig(out / f"matrix_{mode}_t{t}.pdf", dpi=300)
                    plt.close()

if __name__ == "__main__":
    main()
