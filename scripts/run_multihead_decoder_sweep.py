#!/usr/bin/env python3
"""Small decoder-architecture screen for DyNACO multi-head priors.

The default is a dry-run over the <1K TSP test file with deliberately small
training/evaluation settings. Use --run to execute.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "data/TSP/data/test_set/STAR_TSP_lt1000.txt"


@dataclass(frozen=True)
class Variant:
    name: str
    decoder: str
    rank: int = 4
    loss_js: float = 0.0
    init: str = "anchored"


VARIANTS = [
    Variant("control_lora_r1", "lora", rank=1, loss_js=0.0),
    Variant("control_lora_r8", "lora", rank=8, loss_js=0.0),
    Variant("deep_lora_r4", "deep_lora", rank=4, loss_js=0.0),
    Variant("deep_lora_r4_js", "deep_lora", rank=4, loss_js=0.01),
    Variant("film_js", "film", rank=4, loss_js=0.01),
    Variant("per_head_mlp_js", "per_head_mlp", rank=4, loss_js=0.01),
]


def _run(cmd: list[str], cwd: Path, log_path: Path, execute: bool) -> int:
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        if not execute:
            return 0
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return int(proc.returncode)


def _latest_best_checkpoint(save_dir: Path, problem: str, n_node: int) -> Path | None:
    ckpt_dir = save_dir / problem / f"n{n_node}"
    matches = sorted(ckpt_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _method_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = payload.get("methods", {})
    method = methods.get("model_anneal") or methods.get("model_no_anneal") or {}
    diag = method.get("guidance_diagnostics", {}) or {}
    return {
        "gap_pct": method.get("gap_pct"),
        "mean_cost": method.get("mean_cost"),
        "mean_time_s": method.get("mean_time_s"),
        "head_logit_corr": diag.get("head_logit_corr"),
        "head_topk_overlap": diag.get("head_topk_overlap"),
        "head_selected_edge_overlap": diag.get("head_selected_edge_overlap"),
        "head_best_fraction": json.dumps(diag.get("head_best_fraction")),
        "head_improvement_fraction": json.dumps(diag.get("head_improvement_fraction")),
    }


def build_train_cmd(args: argparse.Namespace, variant: Variant, save_dir: Path) -> list[str]:
    cmd = [
        "uv", "run", "train.py",
        "--problem", "tsp",
        "--n_node", str(args.train_n_node),
        "--k_sparse", str(args.k_sparse),
        "--algo", "ppo",
        "--alg", "faco",
        "--n_ants", str(args.n_ants),
        "--H", str(args.train_H),
        "--mini_H", str(args.train_mini_H),
        "--epochs", str(args.epochs),
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--ppo_epochs", str(args.ppo_epochs),
        "--ppo_lr", str(args.lr),
        "--rho", str(args.rho),
        "--min_new_edges", str(args.min_new_edges),
        "--multi_head",
        "--num_heads", str(args.num_heads),
        "--head_decoder_type", variant.decoder,
        "--lora_rank", str(variant.rank),
        "--head_adapter_init", variant.init,
        "--loss_js", str(variant.loss_js),
        "--val_dataset", args.dataset,
        "--val_size", str(args.val_size),
        "--baseline", "none",
        "--device", args.device,
        "--threads", str(args.threads),
        "--seed", str(args.seed),
        "--save_dir", str(save_dir),
        "--run_tag", variant.name,
        "--no_wandb",
    ]
    return cmd


def build_eval_cmd(args: argparse.Namespace, variant: Variant, ckpt: Path, summary_path: Path) -> list[str]:
    return [
        "uv", "run", "test.py",
        "--problem", "tsp",
        "--n_node", str(args.train_n_node),
        "--k_sparse", str(args.k_sparse),
        "--dataset", args.dataset,
        "--checkpoint", str(ckpt),
        "--n_ants", str(args.eval_ants),
        "--H", str(args.eval_H),
        "--mini_H", str(args.eval_mini_H),
        "--multi_head",
        "--num_heads", str(args.num_heads),
        "--head_decoder_type", variant.decoder,
        "--lora_rank", str(variant.rank),
        "--head_adapter_init", variant.init,
        "--head_ant_weights", args.head_ant_weights,
        "--val_size", str(args.val_size),
        "--baseline", "none",
        "--no_baseline",
        "--collect_guidance_metrics",
        "--stage_metrics",
        "--summary_json", str(summary_path),
        "--device", args.device,
        "--threads", str(args.threads),
        "--seed", str(args.seed),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute commands; default only writes commands")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-n-node", type=int, default=100)
    parser.add_argument("--k-sparse", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-ants", type=int, default=8)
    parser.add_argument("--train-H", type=int, default=2)
    parser.add_argument("--train-mini-H", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--min-new-edges", type=int, default=4)
    parser.add_argument("--eval-ants", type=int, default=8)
    parser.add_argument("--eval-H", type=int, default=2)
    parser.add_argument("--eval-mini-H", type=int, default=2)
    parser.add_argument("--val-size", type=int, default=1)
    parser.add_argument("--head-ant-weights", default="25,25,25,25")
    parser.add_argument("--only", nargs="*", default=None, help="Optional variant-name filter")
    args = parser.parse_args(argv)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or ROOT / "results" / "multihead_decoder_sweep" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    variants = [v for v in VARIANTS if not args.only or v.name in set(args.only)]
    manifest_path = out_dir / "manifest.jsonl"
    summary_csv = out_dir / "decoder_sweep.csv"
    commands_log = out_dir / "commands.log"

    rows = []
    with commands_log.open("w", encoding="utf-8") as commands, manifest_path.open("w", encoding="utf-8") as manifest:
        for variant in variants:
            before = set((checkpoint_dir / "tsp" / f"n{args.train_n_node}").glob("*_best.pt"))
            train_cmd = build_train_cmd(args, variant, checkpoint_dir)
            commands.write(" ".join(train_cmd) + "\n")
            train_rc = _run(train_cmd, ROOT, log_dir / f"{variant.name}_train.log", args.run)

            ckpt = None
            eval_rc = None
            summary_path = out_dir / f"{variant.name}_summary.json"
            if train_rc == 0 and args.run:
                after = set((checkpoint_dir / "tsp" / f"n{args.train_n_node}").glob("*_best.pt"))
                new_ckpts = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt = new_ckpts[0] if new_ckpts else _latest_best_checkpoint(checkpoint_dir, "tsp", args.train_n_node)
                if ckpt is not None:
                    eval_cmd = build_eval_cmd(args, variant, ckpt, summary_path)
                    commands.write(" ".join(eval_cmd) + "\n")
                    eval_rc = _run(eval_cmd, ROOT, log_dir / f"{variant.name}_eval.log", args.run)
            elif not args.run:
                eval_cmd = build_eval_cmd(args, variant, Path("<checkpoint-from-train>"), summary_path)
                commands.write(" ".join(eval_cmd) + "\n")

            record = {
                "variant": variant.name,
                "decoder": variant.decoder,
                "rank": variant.rank,
                "loss_js": variant.loss_js,
                "train_rc": train_rc,
                "eval_rc": eval_rc,
                "checkpoint": str(ckpt) if ckpt else "",
                "summary_json": str(summary_path),
            }
            record.update(_method_summary(summary_path))
            rows.append(record)
            manifest.write(json.dumps(record) + "\n")

    fieldnames = [
        "variant", "decoder", "rank", "loss_js", "train_rc", "eval_rc",
        "gap_pct", "mean_cost", "mean_time_s", "head_logit_corr",
        "head_topk_overlap", "head_selected_edge_overlap",
        "head_best_fraction", "head_improvement_fraction",
        "checkpoint", "summary_json",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    readme = out_dir / "SUMMARY.md"
    mode = "executed" if args.run else "dry-run"
    readme.write_text(
        f"# Multi-Head Decoder Sweep ({mode})\n\n"
        f"- Dataset: `{args.dataset}`\n"
        f"- Output: `{out_dir}`\n"
        f"- Commands: `commands.log`\n"
        f"- Summary CSV: `decoder_sweep.csv`\n"
        f"- Variants: {', '.join(v.name for v in variants)}\n",
        encoding="utf-8",
    )

    if shutil.which("uv") is None:
        print("Warning: uv not found on PATH; commands were still written.")
    print(f"Wrote {mode} sweep artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
