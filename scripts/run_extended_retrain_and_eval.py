#!/usr/bin/env python3
"""Retrain and evaluate reconstructed extended-problem runs sequentially.

The paper table requires three distinct methods:
pure ACO, a separately trained static-prior model, and a dynamic model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PAIRS = [
    (
        ROOT / "configs/extended/bpp_n120_retrain20.yaml",
        ROOT / "configs/extended/bpp_n120_static_retrain20.yaml",
    ),
    (
        ROOT / "configs/extended/mkp_n300_retrain20.yaml",
        ROOT / "configs/extended/mkp_n300_static_retrain20.yaml",
    ),
    (
        ROOT / "configs/extended/op_n100_retrain20.yaml",
        ROOT / "configs/extended/op_n100_static_retrain20.yaml",
    ),
]

DYNAMIC_CONFIGS = [
    ROOT / "configs/extended/bpp_n120_retrain20.yaml",
    ROOT / "configs/extended/mkp_n300_retrain20.yaml",
    ROOT / "configs/extended/op_n100_retrain20.yaml",
]


def model_name(cfg: dict[str, Any]) -> str:
    name = (
        f"{cfg['problem']}_n{cfg['n_node']}_k{cfg['k_sparse']}_ants{cfg['n_ants']}"
        f"_H{cfg['H']}_miniH{cfg['mini_H']}_rho{cfg['rho']}"
        f"_lr{cfg['lr']}"
    )
    if cfg.get("train_anneal", False):
        name += f"_anneal_g{cfg.get('gamma', 1.0)}_mg{cfg.get('min_gamma', 0.0)}"
    if cfg.get("warmup", 0) > 0:
        name += f"_warmup{cfg['warmup']}"
    if cfg.get("no_dynamic_feats", False):
        name += "_static"
    if cfg.get("static_prior", False):
        name += "_static_prior"
    if cfg.get("multi_head", False):
        name += f"_mh{cfg.get('num_heads', 4)}_{cfg.get('head_decoder_type', 'deep_lora')}"
    return name


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_command(config_path: Path, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.no_wandb:
        cmd += ["--no_wandb"]
    return cmd


def test_command(
    config_path: Path,
    cfg: dict[str, Any],
    checkpoint: Path,
    static_checkpoint: Path,
    args: argparse.Namespace,
) -> list[str]:
    save_dir = Path(args.results_dir) / config_path.stem
    cmd = [
        sys.executable,
        "test.py",
        "--problem",
        str(cfg["problem"]),
        "--n_node",
        str(cfg["n_node"]),
        "--checkpoint",
        str(checkpoint),
        "--static_checkpoint",
        str(static_checkpoint),
        "--test_size",
        str(args.test_size),
        "--H",
        str(cfg["H"]),
        "--mini_H",
        str(cfg["mini_H"]),
        "--n_ants",
        str(cfg["n_ants"]),
        "--k_sparse",
        str(cfg["k_sparse"]),
        "--rho",
        str(cfg["rho"]),
        "--alpha",
        str(cfg.get("alpha", 1.0)),
        "--beta",
        str(cfg.get("beta", 1.0)),
        "--static_compare",
        "--save_results",
        "--save_dir",
        str(save_dir),
    ]
    if args.device is not None:
        cmd += ["--device", args.device]
    else:
        cmd += ["--device", str(cfg.get("device", "cuda:0"))]
    if cfg["problem"] == "mkp":
        cmd += ["--m", str(cfg.get("m", 5))]
    elif cfg["problem"] == "bpp":
        cmd += ["--capacity", str(cfg.get("capacity", 150.0))]
    elif cfg["problem"] == "op":
        cmd += ["--max_len", str(cfg.get("max_len", 4.0))]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially retrain BPP/MKP/OP and evaluate ACO vs separate static-prior vs dynamic guidance."
    )
    parser.add_argument("--skip-train", action="store_true", help="Only run evaluation from existing checkpoints.")
    parser.add_argument("--skip-dynamic-train", action="store_true", help="Do not train dynamic checkpoints.")
    parser.add_argument("--skip-static-train", action="store_true", help="Do not train static-prior checkpoints.")
    parser.add_argument("--skip-test", action="store_true", help="Only run training.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda:0 or cpu.")
    parser.add_argument("--test-size", type=int, default=100, help="Number of test instances per problem.")
    parser.add_argument("--results-dir", default="results_extended/retrain20_eval", help="Evaluation output root.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B during training.")
    args = parser.parse_args()

    for dynamic_config_path, static_config_path in CONFIG_PAIRS:
        cfg = load_config(dynamic_config_path)
        static_cfg = load_config(static_config_path)
        checkpoint = ROOT / cfg.get("save_dir", "checkpoints_extended") / f"{model_name(cfg)}_best.pt"
        static_checkpoint = ROOT / static_cfg.get("save_dir", "checkpoints_extended") / f"{model_name(static_cfg)}_best.pt"

        print(f"\n=== {cfg['problem'].upper()} | {dynamic_config_path.relative_to(ROOT)} ===", flush=True)
        if not args.skip_train and not args.skip_dynamic_train:
            run_command(train_command(dynamic_config_path, args), dry_run=args.dry_run)
        if not args.skip_train and not args.skip_static_train:
            run_command(train_command(static_config_path, args), dry_run=args.dry_run)

        if not args.skip_test:
            if not args.dry_run and not checkpoint.exists():
                raise FileNotFoundError(f"Expected dynamic checkpoint not found after training: {checkpoint}")
            if not args.dry_run and not static_checkpoint.exists():
                raise FileNotFoundError(f"Expected static checkpoint not found after training: {static_checkpoint}")
            run_command(
                test_command(dynamic_config_path, cfg, checkpoint, static_checkpoint, args),
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
