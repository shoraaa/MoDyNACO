#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


DEFAULT_SAVE_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("output") / "train_and_test_cvrp"
DEFAULT_EVAL_JSON_NAME = "evaluation_summary.json"
DEFAULT_RUN_META_NAME = "run_metadata.json"
DEFAULT_SWEEP_CSV_NAME = "capacity_sweep.csv"


def parse_capacities(value: str) -> list[float]:
    capacities: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        cap = float(part)
        if cap <= 0:
            raise argparse.ArgumentTypeError("Capacities must be positive")
        capacities.append(cap)
    if not capacities:
        raise argparse.ArgumentTypeError("At least one capacity is required")
    return capacities


def default_run_name(args: argparse.Namespace) -> str:
    if getattr(args, "capacity_override", None) is not None:
        return f"cvrp_n{args.n_node}_cap{args.capacity_override:g}_seed{args.seed}"
    return f"cvrp_n{args.n_node}_seed{args.seed}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an in-scale CVRP model and immediately evaluate it with test.py."
    )
    parser.add_argument("--n_node", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--capacities", type=parse_capacities, default=None,
                        help="Comma-separated capacities. Trains and tests one model per capacity.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=32)
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--min_new_edges", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ppo_lr", type=float, default=5e-6)
    parser.add_argument("--reinforce_lr", type=float, default=1e-4)
    parser.add_argument("--algo", choices=["ppo", "reinforce"], default="ppo")
    parser.add_argument("--alg", choices=["faco", "mmas"], default="faco")

    parser.add_argument("--no_smooth_mmas", action="store_true", default=True)
    parser.add_argument("--smooth_mmas", dest="no_smooth_mmas", action="store_false")
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_extend_ls", action="store_true")
    parser.add_argument("--ls_scope", choices=["localized", "global"], default="localized")
    parser.add_argument("--ls_budget", choices=["truncated", "full"], default="truncated")
    parser.add_argument("--ls_max_opt", type=int, default=0)
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--no_normalized_heuristic", action="store_true")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    parser.add_argument("--no_logit_net", action="store_true")
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument("--baseline_time_limit", type=float, default=0.5)
    parser.add_argument("--val_size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--train_warmup", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.5)
    parser.add_argument("--train_anneal", action="store_true")
    parser.add_argument("--no_anneal", action="store_true", default=True)
    parser.add_argument("--anneal", dest="no_anneal", action="store_false")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.0)
    parser.add_argument("--L", type=int, default=0)

    parser.add_argument("--run_name", "--wandb_name", dest="run_name", type=str, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--metadata_json", type=Path, default=None)
    parser.add_argument("--sweep_csv", type=Path, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Explicit checkpoint to test. Required with --skip_train.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the train/test commands without running them.")
    parser.add_argument("--train_extra", nargs=argparse.REMAINDER, default=None,
                        help="Extra args forwarded to train.py. Put this last.")
    return parser


def default_output_dir(args: argparse.Namespace) -> Path:
    run_tag = args.run_name if args.run_name else default_run_name(args)
    return args.output_dir or (DEFAULT_OUTPUT_DIR / run_tag)


def maybe_append_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def build_train_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "train.py",
        "--problem",
        "cvrp",
        "--n_node",
        str(args.n_node),
        "--device",
        args.device,
        "--save_dir",
        str(args.save_dir),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--steps_per_epoch",
        str(args.steps_per_epoch),
        "--k_sparse",
        str(args.k_sparse),
        "--n_ants",
        str(args.n_ants),
        "--H",
        str(args.H),
        "--mini_H",
        str(args.mini_H),
        "--rho",
        str(args.rho),
        "--algo",
        args.algo,
        "--alg",
        args.alg,
        "--baseline",
        args.baseline,
        "--baseline_time_limit",
        str(args.baseline_time_limit),
        "--val_size",
        str(args.val_size),
        "--generate_val",
    ]

    if args.min_new_edges is not None:
        cmd.extend(["--min_new_edges", str(args.min_new_edges)])
    if getattr(args, "capacity_override", None) is not None:
        cmd.extend(["--capacity_override", str(args.capacity_override)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    else:
        cmd.extend(["--ppo_lr", str(args.ppo_lr), "--reinforce_lr", str(args.reinforce_lr)])
    if args.run_name:
        cmd.extend(["--run_name", args.run_name])
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.train_warmup:
        cmd.append("--train_warmup")
    if args.warmup_ratio != 0.5:
        cmd.extend(["--warmup_ratio", str(args.warmup_ratio)])
    if args.train_anneal:
        cmd.append("--train_anneal")
    if args.gamma != 1.0:
        cmd.extend(["--gamma", str(args.gamma)])
    if args.min_gamma != 0.0:
        cmd.extend(["--min_gamma", str(args.min_gamma)])
    if args.L != 0:
        cmd.extend(["--L", str(args.L)])

    maybe_append_flag(cmd, args.no_smooth_mmas, "--no_smooth_mmas")
    maybe_append_flag(cmd, args.no_local_search, "--no_local_search")
    maybe_append_flag(cmd, args.no_extend_ls, "--no_extend_ls")
    maybe_append_flag(cmd, args.disable_heuristic, "--disable_heuristic")
    maybe_append_flag(cmd, args.no_normalized_heuristic, "--no_normalized_heuristic")
    maybe_append_flag(cmd, args.no_dynamic_feats, "--no_dynamic_feats")
    maybe_append_flag(cmd, args.no_logit_net, "--no_logit_net")

    if args.ls_scope != "localized":
        cmd.extend(["--ls_scope", args.ls_scope])
    if args.ls_budget != "truncated":
        cmd.extend(["--ls_budget", args.ls_budget])
    if args.ls_max_opt != 0:
        cmd.extend(["--ls_max_opt", str(args.ls_max_opt)])
    if args.train_extra:
        cmd.extend(args.train_extra)
    return cmd


def build_test_command(args: argparse.Namespace, checkpoint: Path, summary_json: Path) -> list[str]:
    cmd = [
        args.python,
        "test.py",
        "--problem",
        "cvrp",
        "--n_node",
        str(args.n_node),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
        "--baseline",
        args.baseline,
        "--k_sparse",
        str(args.k_sparse),
        "--n_ants",
        str(args.n_ants),
        "--H",
        str(args.H),
        "--mini_H",
        str(args.mini_H),
        "--rho",
        str(args.rho),
        "--summary_json",
        str(summary_json),
        "--collect_guidance_metrics",
        "--generate_val",
    ]
    if args.min_new_edges is not None:
        cmd.extend(["--min_new_edges", str(args.min_new_edges)])
    if getattr(args, "capacity_override", None) is not None:
        cmd.extend(["--capacity_override", str(args.capacity_override)])
    if args.val_size is not None:
        cmd.extend(["--val_size", str(args.val_size)])
    maybe_append_flag(cmd, args.no_smooth_mmas, "--no_smooth_mmas")
    maybe_append_flag(cmd, args.no_local_search, "--no_local_search")
    maybe_append_flag(cmd, args.no_extend_ls, "--no_extend_ls")
    maybe_append_flag(cmd, args.disable_heuristic, "--disable_heuristic")
    maybe_append_flag(cmd, args.no_normalized_heuristic, "--no_normalized_heuristic")
    maybe_append_flag(cmd, args.no_dynamic_feats, "--no_dynamic_feats")
    maybe_append_flag(cmd, args.no_logit_net, "--no_logit_net")
    maybe_append_flag(cmd, args.no_anneal, "--no_anneal")
    if args.ls_scope != "localized":
        cmd.extend(["--ls_scope", args.ls_scope])
    if args.ls_budget != "truncated":
        cmd.extend(["--ls_budget", args.ls_budget])
    if args.ls_max_opt != 0:
        cmd.extend(["--ls_max_opt", str(args.ls_max_opt)])
    return cmd


def run_command(cmd: Sequence[str], cwd: Path) -> int:
    process = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return process.wait()


def find_latest_best_checkpoint(save_dir: Path, n_node: int, started_at: float | None = None) -> Path:
    checkpoint_dir = save_dir / "cvrp" / f"n{n_node}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    candidates = sorted(checkpoint_dir.glob("*_best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No *_best.pt checkpoint found in {checkpoint_dir}")

    if started_at is not None:
        fresh = [path for path in candidates if path.stat().st_mtime >= started_at - 1.0]
        if fresh:
            candidates = fresh

    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_single_capacity(args: argparse.Namespace, repo_root: Path, out_dir: Path) -> dict:
    summary_json = args.summary_json or (out_dir / DEFAULT_EVAL_JSON_NAME)
    metadata_json = args.metadata_json or (out_dir / DEFAULT_RUN_META_NAME)

    train_cmd = build_train_command(args)
    checkpoint: Path | None = args.checkpoint
    started_at = time.time()
    phase_total = int(not args.skip_train) + int(not args.skip_test)
    if phase_total == 0:
        phase_total = 1
    phase_bar = tqdm(total=phase_total, desc="Pipeline", unit="phase")

    if args.dry_run:
        print("Train command:")
        print(" ".join(train_cmd))
    elif not args.skip_train:
        phase_bar.set_description("Pipeline train")
        phase_bar.set_postfix_str("running train.py")
        train_code = run_command(train_cmd, repo_root)
        if train_code != 0:
            phase_bar.close()
            raise SystemExit(train_code)
        checkpoint = find_latest_best_checkpoint(args.save_dir, args.n_node, started_at=started_at)
        phase_bar.update(1)
        phase_bar.set_postfix_str(f"trained {checkpoint.name}")

    if args.dry_run and checkpoint is None:
        checkpoint = args.save_dir / "cvrp" / f"n{args.n_node}" / "<trained_best_checkpoint>.pt"
    elif checkpoint is None:
        checkpoint = find_latest_best_checkpoint(args.save_dir, args.n_node)
    if not args.dry_run and not checkpoint.exists():
        phase_bar.close()
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    test_cmd = build_test_command(args, checkpoint, summary_json)
    if args.dry_run:
        print("\nTest command:")
        print(" ".join(test_cmd))
    elif not args.skip_test:
        phase_bar.set_description("Pipeline test")
        phase_bar.set_postfix_str("running test.py")
        test_code = run_command(test_cmd, repo_root)
        if test_code != 0:
            phase_bar.close()
            raise SystemExit(test_code)
        phase_bar.update(1)
        phase_bar.set_postfix_str(f"tested {checkpoint.name}")

    if args.dry_run:
        phase_bar.set_description("Pipeline dry-run")
        phase_bar.set_postfix_str("commands prepared")
        phase_bar.update(phase_total)

    metadata = {
        "capacity_override": getattr(args, "capacity_override", None),
        "checkpoint": str(checkpoint),
        "dry_run": args.dry_run,
        "output_dir": str(out_dir),
        "summary_json": str(summary_json),
        "train_command": train_cmd,
        "test_command": test_cmd,
        "skip_train": args.skip_train,
        "skip_test": args.skip_test,
    }
    if summary_json.exists():
        metadata["test_summary"] = json.loads(summary_json.read_text(encoding="utf-8"))
    write_metadata(metadata_json, metadata)
    phase_bar.close()

    print(f"Resolved checkpoint: {checkpoint}")
    print(f"Wrote metadata: {metadata_json}")
    if not args.skip_test:
        print(f"Expected test summary: {summary_json}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(cli_argv)
    if args.run_name is None:
        args.run_name = default_run_name(args)

    repo_root = Path(__file__).resolve().parent
    if args.skip_train and args.checkpoint is None:
        parser.error("--checkpoint is required when --skip_train is set")
    if args.capacities:
        root_out_dir = default_output_dir(args)
        root_out_dir.mkdir(parents=True, exist_ok=True)
        sweep_csv = args.sweep_csv or (root_out_dir / DEFAULT_SWEEP_CSV_NAME)
        sweep_rows = []
        sweep_bar = tqdm(args.capacities, desc="Capacity Sweep", unit="cap")
        for capacity in sweep_bar:
            sweep_args = argparse.Namespace(**vars(args))
            sweep_args.capacity_override = float(capacity)
            if args.run_name == default_run_name(args):
                sweep_args.run_name = default_run_name(sweep_args)
            else:
                sweep_args.run_name = f"{args.run_name}_cap{capacity:g}"
            sweep_args.summary_json = None
            sweep_args.metadata_json = None
            cap_out_dir = root_out_dir / f"cap{capacity:g}"
            cap_out_dir.mkdir(parents=True, exist_ok=True)
            sweep_bar.set_postfix_str(f"cap={capacity:g}")
            metadata = run_single_capacity(sweep_args, repo_root, cap_out_dir)
            summary = metadata.get("test_summary", {})
            methods = summary.get("methods", {})
            model_summary = methods.get("model_no_anneal") or methods.get("model_anneal") or {}
            sweep_rows.append({
                "capacity": capacity,
                "checkpoint": metadata.get("checkpoint"),
                "mean_cost": model_summary.get("mean_cost"),
                "gap_pct": model_summary.get("gap_pct"),
                "mean_time_s": model_summary.get("mean_time_s"),
                "mean_guidance": model_summary.get("mean_guidance"),
                "pheromone_correlation": model_summary.get("pheromone_correlation"),
                "summary_json": metadata.get("summary_json"),
            })
        with sweep_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "capacity",
                    "checkpoint",
                    "mean_cost",
                    "gap_pct",
                    "mean_time_s",
                    "mean_guidance",
                    "pheromone_correlation",
                    "summary_json",
                ],
            )
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"Wrote sweep CSV: {sweep_csv}")
    else:
        out_dir = default_output_dir(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_single_capacity(args, repo_root, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
