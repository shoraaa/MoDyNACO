#!/usr/bin/env python3
"""Independent-policy ensemble ablation for DyNACO.

The default mode is a dry-run that writes the exact train/eval commands. Add
--run to execute. The ablation trains R independent single-head policies, then
evaluates them together as policy heads inside one ACO rollout.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "data/TSP/data/test_set/STAR_TSP_lt1000.txt"
METHOD_COLUMNS = {
    "model_anneal": ("model_anneal", "model_anneal"),
    "model_no_anneal": ("model_no_anneal", "model_no_anneal"),
}


def _run(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    execute: bool,
    *,
    label: str,
    stream_output: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[start] {label}", flush=True)
    print(f"        log: {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        if not execute:
            print(f"[dry-run] {label}", flush=True)
            return 0
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if stream_output:
                print(line, end="", flush=True)
        proc.wait()
        rc = int(proc.returncode)
    elapsed = time.time() - t0
    status = "done" if rc == 0 else f"failed rc={rc}"
    print(f"[{status}] {label} ({elapsed:.1f}s)", flush=True)
    return rc


def _latest_best_checkpoint(save_dir: Path, problem: str, n_node: int) -> Path | None:
    ckpt_dir = save_dir / problem / f"n{n_node}"
    matches = sorted(ckpt_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _new_best_checkpoint(before: set[Path], save_dir: Path, problem: str, n_node: int) -> Path | None:
    ckpt_dir = save_dir / problem / f"n{n_node}"
    after = set(ckpt_dir.glob("*_best.pt"))
    new_ckpts = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return new_ckpts[0] if new_ckpts else _latest_best_checkpoint(save_dir, problem, n_node)


def _new_instance_csv(before: set[Path]) -> Path | None:
    after = set((ROOT / "logs" / "csv").glob("*_instances.csv"))
    new_csvs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return new_csvs[0] if new_csvs else None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _load_member_rows(csv_path: Path, method_column: str, time_column: str | None) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            rows[idx] = {
                "idx": idx,
                "name": row.get("name") or str(idx),
                "size": _safe_float(row.get("size")),
                "size_group": row.get("size_group"),
                "opt": _safe_float(row.get("opt")),
                "baseline": _safe_float(row.get("baseline")),
                "cost": _safe_float(row.get(method_column)),
                "time": _safe_float(row.get(time_column)) if time_column and time_column in row else None,
            }
    return rows


def _split_counts(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total < parts:
        raise ValueError(f"total={total} must be >= parts={parts}")
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _aggregate_ind_r(
    member_csvs: list[Path],
    method: str,
    out_dir: Path,
    *,
    total_eval_ants: int,
    ant_counts: list[int],
) -> dict[str, Any]:
    method_column, _time_prefix = METHOD_COLUMNS[method]
    time_column = None
    members = [_load_member_rows(path, method_column, time_column) for path in member_csvs]
    all_indices = sorted(set.intersection(*(set(m.keys()) for m in members)))
    rows = []
    costs = []
    gaps = []
    times = []
    wins = [0 for _ in members]

    for idx in all_indices:
        candidates = []
        total_time = 0.0
        saw_time = False
        for member_idx, member_rows in enumerate(members):
            row = member_rows[idx]
            cost = row["cost"]
            if cost is not None:
                candidates.append((float(cost), member_idx))
            if row["time"] is not None:
                saw_time = True
                total_time += float(row["time"])
        if not candidates:
            continue
        best_cost, winner = min(candidates, key=lambda x: x[0])
        wins[winner] += 1
        ref = members[0][idx]["opt"] or members[0][idx]["baseline"]
        gap_pct = None if ref is None or ref <= 0 else (best_cost - float(ref)) / float(ref) * 100.0
        costs.append(best_cost)
        if saw_time:
            times.append(total_time)
        if gap_pct is not None and math.isfinite(gap_pct):
            gaps.append(float(gap_pct))
        rows.append({
            "idx": idx,
            "name": members[0][idx]["name"],
            "size": members[0][idx]["size"],
            "size_group": members[0][idx]["size_group"],
            "reference": ref,
            "ind_r_cost": best_cost,
            "ind_r_gap_pct": gap_pct,
            "ind_r_time_s": total_time if saw_time else None,
            "winner_member": winner,
            **{f"member_{m}_cost": members[m][idx]["cost"] for m in range(len(members))},
        })

    per_instance_path = out_dir / "ind_r_per_instance.csv"
    fieldnames = [
        "idx", "name", "size", "size_group", "reference", "ind_r_cost",
        "ind_r_gap_pct", "ind_r_time_s", "winner_member",
    ] + [f"member_{m}_cost" for m in range(len(members))]
    with per_instance_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "method": f"Ind-{len(members)}",
        "member_count": len(members),
        "total_eval_ants": int(total_eval_ants),
        "member_ant_counts": [int(x) for x in ant_counts],
        "aggregation": "best cost per instance across independently trained single-head policies",
        "instances": len(rows),
        "mean_cost": _summary(costs)["mean"],
        "std_cost": _summary(costs)["std"],
        "gap_pct": _summary(gaps)["mean"],
        "gap_std_pct": _summary(gaps)["std"],
        "mean_time_s": _summary(times)["mean"],
        "total_time_s": float(sum(times)) if times else None,
        "member_win_fraction": [w / len(rows) if rows else 0.0 for w in wins],
        "per_instance_csv": str(per_instance_path),
    }
    (out_dir / "ind_r_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _load_config_defaults(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults: dict[str, Any] = {}
    if "problem" in raw:
        defaults["problem"] = raw["problem"]
    if "n_node" in raw:
        defaults["train_n_node"] = raw["n_node"]
    if "k_sparse" in raw:
        defaults["k_sparse"] = raw["k_sparse"]
    if "algo" in raw:
        defaults["algo"] = raw["algo"]
    if "alg" in raw:
        defaults["alg"] = raw["alg"]
    if "n_ants" in raw:
        defaults["train_ants"] = raw["n_ants"]
        defaults["eval_ants"] = raw["n_ants"]
    if "H" in raw:
        defaults["train_H"] = raw["H"]
        defaults["eval_H"] = raw["H"]
    if "mini_H" in raw:
        defaults["train_mini_H"] = raw["mini_H"]
        defaults["eval_mini_H"] = raw["mini_H"]
    if "epochs" in raw:
        defaults["epochs"] = raw["epochs"]
    if "steps_per_epoch" in raw:
        defaults["steps_per_epoch"] = raw["steps_per_epoch"]
    if "ppo_epochs" in raw:
        defaults["ppo_epochs"] = raw["ppo_epochs"]
    if "lr" in raw:
        defaults["lr"] = raw["lr"]
    if "ppo_lr" in raw:
        defaults["lr"] = raw["ppo_lr"]
    if "rho" in raw:
        defaults["rho"] = raw["rho"]
    if "min_new_edges" in raw:
        defaults["min_new_edges"] = raw["min_new_edges"]
    if "edge_feature_set" in raw:
        defaults["edge_feature_set"] = raw["edge_feature_set"]
    if "val_dataset" in raw:
        defaults["train_dataset"] = raw["val_dataset"]
    if "train_dataset" in raw:
        defaults["train_dataset"] = raw["train_dataset"]
    if "test_dataset" in raw:
        defaults["test_dataset"] = raw["test_dataset"]
    if "val_size" in raw:
        defaults["val_size"] = raw["val_size"]
    if "device" in raw:
        defaults["device"] = raw["device"]
    if "seed" in raw:
        defaults["seed"] = raw["seed"]
    if "num_heads" in raw:
        defaults["members"] = raw["num_heads"]
        defaults["num_heads"] = raw["num_heads"]
    if "head_decoder_type" in raw:
        defaults["head_decoder_type"] = raw["head_decoder_type"]
    if "lora_rank" in raw:
        defaults["lora_rank"] = raw["lora_rank"]
    if "head_adapter_init" in raw:
        defaults["head_adapter_init"] = raw["head_adapter_init"]
    if "head_ant_weights" in raw and raw["head_ant_weights"] is not None:
        defaults["head_ant_weights"] = raw["head_ant_weights"]
    if "use_wandb" in raw:
        defaults["wandb"] = bool(raw["use_wandb"])
    if "wandb_project" in raw and raw["wandb_project"] is not None:
        defaults["wandb_project"] = raw["wandb_project"]
    if "wandb_entity" in raw and raw["wandb_entity"] is not None:
        defaults["wandb_entity"] = raw["wandb_entity"]
    if "wandb_group" in raw and raw["wandb_group"] is not None:
        defaults["wandb_group"] = raw["wandb_group"]

    passthrough = {
        "members", "checkpoints", "ants_per_policy", "eval_method",
        "collect_guidance_metrics", "multihead_checkpoint", "threads", "out_dir",
    }
    for key in passthrough:
        if key in raw:
            defaults[key] = raw[key]
    return defaults


def build_train_cmd(args: argparse.Namespace, member: int, seed: int, save_dir: Path) -> list[str]:
    cmd = [
        "uv", "run", "train.py",
        "--problem", args.problem,
        "--n_node", str(args.train_n_node),
        "--k_sparse", str(args.k_sparse),
        "--algo", "ppo",
        "--alg", args.alg,
        "--n_ants", str(args.train_ants),
        "--H", str(args.train_H),
        "--mini_H", str(args.train_mini_H),
        "--epochs", str(args.epochs),
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--ppo_epochs", str(args.ppo_epochs),
        "--ppo_lr", str(args.lr),
        "--rho", str(args.rho),
        "--min_new_edges", str(args.min_new_edges),
        "--edge_feature_set", args.edge_feature_set,
        "--val_dataset", args.train_dataset,
        "--val_size", str(args.val_size),
        "--baseline", "none",
        "--device", args.device,
        "--threads", str(args.threads),
        "--seed", str(seed),
        "--save_dir", str(save_dir),
        "--run_tag", f"ind_r_m{member}_seed{seed}",
    ]
    if args.wandb:
        cmd.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", args.wandb_entity])
        group = args.wandb_group or f"{args.problem}_ind_r_{args.members}"
        cmd.extend(["--wandb_group", group])
    else:
        cmd.append("--no_wandb")
    return cmd


def build_eval_cmd(
    args: argparse.Namespace,
    ckpt: Path,
    summary_path: Path,
    *,
    ants: int,
    seed: int,
    multi_head: bool = False,
) -> list[str]:
    cmd = [
        "uv", "run", "test.py",
        "--problem", args.problem,
        "--n_node", str(args.train_n_node),
        "--k_sparse", str(args.k_sparse),
        "--alg", args.alg,
        "--dataset", args.test_dataset,
        "--checkpoint", str(ckpt),
        "--n_ants", str(ants),
        "--H", str(args.eval_H),
        "--mini_H", str(args.eval_mini_H),
        "--val_size", str(args.val_size),
        "--baseline", "none",
        "--no_baseline",
        "--log",
        "--summary_json", str(summary_path),
        "--device", args.device,
        "--threads", str(args.threads),
        "--seed", str(seed),
    ]
    if args.eval_method == "model_anneal":
        cmd.append("--run_model_anneal")
    else:
        cmd.append("--run_model_no_anneal")
    if args.collect_guidance_metrics:
        cmd.extend(["--collect_guidance_metrics", "--stage_metrics"])
    if multi_head:
        cmd.extend([
            "--multi_head",
            "--num_heads", str(args.num_heads),
            "--head_decoder_type", args.head_decoder_type,
            "--lora_rank", str(args.lora_rank),
            "--head_adapter_init", args.head_adapter_init,
            "--head_ant_weights", args.head_ant_weights,
        ])
    return cmd


def build_ensemble_eval_cmd(
    args: argparse.Namespace,
    checkpoints: list[Path],
    summary_path: Path,
    *,
    ant_counts: list[int],
) -> list[str]:
    cmd = [
        "uv", "run", "test.py",
        "--problem", args.problem,
        "--n_node", str(args.train_n_node),
        "--k_sparse", str(args.k_sparse),
        "--alg", args.alg,
        "--dataset", args.test_dataset,
        "--ensemble_checkpoints", *[str(p) for p in checkpoints],
        "--n_ants", str(args.eval_ants),
        "--H", str(args.eval_H),
        "--mini_H", str(args.eval_mini_H),
        "--val_size", str(args.val_size),
        "--baseline", "none",
        "--no_baseline",
        "--log",
        "--summary_json", str(summary_path),
        "--device", args.device,
        "--threads", str(args.threads),
        "--seed", str(args.seed),
        "--head_ant_weights", ",".join(str(x) for x in ant_counts),
    ]
    if args.eval_method == "model_anneal":
        cmd.append("--run_model_anneal")
    else:
        cmd.append("--run_model_no_anneal")
    if args.collect_guidance_metrics:
        cmd.extend(["--collect_guidance_metrics", "--stage_metrics"])
    return cmd


def main(argv: list[str] | None = None) -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML config; CLI flags override YAML values")
    parser.add_argument("--run", action="store_true", help="Execute commands; default only writes artifacts")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset",
        default=DEFAULT_DATASET,
        help="Dataset used by train.py as --val_dataset for checkpoint selection.",
    )
    parser.add_argument(
        "--dataset",
        dest="train_dataset",
        help="Deprecated alias for --train-dataset; final testing uses --test-dataset.",
    )
    parser.add_argument(
        "--test-dataset",
        dest="test_dataset",
        default=None,
        help="Held-out dataset used by the final test.py --dataset run. Required.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Existing single-head member checkpoints")
    parser.add_argument("--train-n-node", type=int, default=100)
    parser.add_argument("--k-sparse", type=int, default=8)
    parser.add_argument("--alg", choices=["faco", "mmas"], default="faco")
    parser.add_argument("--train-ants", type=int, default=8)
    parser.add_argument("--train-H", type=int, default=2)
    parser.add_argument("--train-mini-H", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--min-new-edges", type=int, default=4)
    parser.add_argument("--edge-feature-set", default="compact3", choices=["full", "compact3"])
    parser.add_argument("--eval-ants", type=int, default=8, help="Total Ind-R inference ants")
    parser.add_argument("--ants-per-policy", type=int, default=None)
    parser.add_argument("--eval-H", type=int, default=2)
    parser.add_argument("--eval-mini-H", type=int, default=2)
    parser.add_argument("--eval-method", choices=sorted(METHOD_COLUMNS), default="model_anneal")
    parser.add_argument("--val-size", type=int, default=1)
    parser.add_argument("--collect-guidance-metrics", action="store_true")
    parser.add_argument("--multihead-checkpoint", type=Path, default=None,
                        help="Optional DeepLoRA/multi-head checkpoint to evaluate with the full total ant budget")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-decoder-type", default="deep_lora",
                        choices=["lora", "deep_lora", "film", "multi_decoder", "per_head_mlp", "lowrank"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--head-adapter-init", default="random",
                        choices=["anchored", "zero", "small", "random"])
    parser.add_argument("--head-ant-weights", default="25,25,25,25")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging for each independent member training run")
    parser.add_argument("--wandb-project", default="dynaco")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--no-stream-output",
        dest="stream_output",
        action="store_false",
        default=True,
        help="Do not mirror child train/test output to the terminal; logs are still written.",
    )
    if config_args.config is not None:
        parser.set_defaults(**_load_config_defaults(config_args.config))
    args = parser.parse_args(argv)

    if not args.test_dataset:
        raise ValueError("--test-dataset is required; Ind-R must not evaluate on the training validation dataset")
    if Path(args.train_dataset) == Path(args.test_dataset):
        raise ValueError("--test-dataset must differ from --train-dataset/val_dataset")

    if args.checkpoints and len(args.checkpoints) != args.members:
        raise ValueError("--checkpoints length must equal --members")
    if args.ants_per_policy is None:
        ant_counts = _split_counts(args.eval_ants, args.members)
    else:
        ant_counts = [int(args.ants_per_policy) for _ in range(args.members)]
    total_assigned = sum(ant_counts)
    if total_assigned != args.eval_ants:
        print(
            f"Warning: assigned member ants = {total_assigned}, "
            f"not eval_ants={args.eval_ants}; summary records both values."
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or ROOT / "results" / "independent_policy_ensemble" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    commands_log = out_dir / "commands.log"
    manifest_path = out_dir / "manifest.jsonl"

    provided_ckpts = [Path(p).expanduser().resolve() for p in args.checkpoints] if args.checkpoints else []
    ckpts = list(provided_ckpts)
    records = []

    mode = "execute" if args.run else "dry-run"
    print(
        "[ind-r] "
        f"mode={mode} problem={args.problem} members={args.members} "
        f"train_dataset={args.train_dataset} test_dataset={args.test_dataset}",
        flush=True,
    )
    print(
        "[ind-r] "
        f"train ants={args.train_ants}, H={args.train_H}, mini_H={args.train_mini_H}, "
        f"epochs={args.epochs}, steps_per_epoch={args.steps_per_epoch}, ppo_epochs={args.ppo_epochs}",
        flush=True,
    )
    print(
        "[ind-r] "
        f"eval ants={args.eval_ants}, member_ant_counts={','.join(str(x) for x in ant_counts)}, "
        f"H={args.eval_H}, mini_H={args.eval_mini_H}, method={args.eval_method}",
        flush=True,
    )
    print(f"[ind-r] output_dir={out_dir}", flush=True)

    with commands_log.open("w", encoding="utf-8") as commands, manifest_path.open("w", encoding="utf-8") as manifest:
        for member in range(args.members):
            seed = args.seed + member
            ckpt = provided_ckpts[member] if provided_ckpts else None
            train_rc = None
            if ckpt is None:
                print(f"[member {member + 1}/{args.members}] training seed={seed}", flush=True)
                before = set((checkpoint_dir / args.problem / f"n{args.train_n_node}").glob("*_best.pt"))
                train_cmd = build_train_cmd(args, member, seed, checkpoint_dir)
                commands.write(" ".join(train_cmd) + "\n")
                train_rc = _run(
                    train_cmd,
                    ROOT,
                    log_dir / f"member_{member}_train.log",
                    args.run,
                    label=f"member {member + 1}/{args.members} train",
                    stream_output=args.stream_output,
                )
                if args.run and train_rc == 0:
                    ckpt = _new_best_checkpoint(before, checkpoint_dir, args.problem, args.train_n_node)
                    print(f"[member {member + 1}/{args.members}] checkpoint={ckpt}", flush=True)
                elif args.run:
                    print(f"[member {member + 1}/{args.members}] training failed rc={train_rc}", flush=True)
            else:
                print(f"[member {member + 1}/{args.members}] using checkpoint={ckpt}", flush=True)
            if ckpt is not None and not provided_ckpts:
                ckpts.append(ckpt)

            record = {
                "member": member,
                "seed": seed,
                "train_rc": train_rc,
                "checkpoint": str(ckpt) if ckpt else "",
            }
            records.append(record)
            manifest.write(json.dumps(record, sort_keys=True) + "\n")

        eval_ckpts = ckpts
        if not args.run and not eval_ckpts:
            eval_ckpts = [Path(f"<member-{member}-checkpoint-from-train>") for member in range(args.members)]
        if args.run and len(eval_ckpts) != args.members:
            raise RuntimeError(f"Expected {args.members} trained checkpoints, found {len(eval_ckpts)}")

        ensemble_summary_path = out_dir / "ind_r_summary.json"
        ensemble_eval_cmd = build_ensemble_eval_cmd(
            args,
            eval_ckpts,
            ensemble_summary_path,
            ant_counts=ant_counts,
        )
        commands.write(" ".join(ensemble_eval_cmd) + "\n")
        before_csvs = set((ROOT / "logs" / "csv").glob("*_instances.csv"))
        print(f"[ensemble] evaluating {len(eval_ckpts)} policies as heads", flush=True)
        ensemble_eval_rc = _run(
            ensemble_eval_cmd,
            ROOT,
            log_dir / "ind_r_ensemble_eval.log",
            args.run,
            label="ensemble eval",
            stream_output=args.stream_output,
        )
        if args.run and ensemble_eval_rc != 0:
            raise RuntimeError(f"Ind-R ensemble evaluation failed with exit code {ensemble_eval_rc}")
        ensemble_instance_csv = _new_instance_csv(before_csvs) if args.run and ensemble_eval_rc == 0 else None
        if ensemble_instance_csv is not None:
            shutil.copy2(ensemble_instance_csv, out_dir / "ind_r_instances.csv")
            print(f"[ensemble] copied instances CSV to {out_dir / 'ind_r_instances.csv'}", flush=True)
        if args.run and ensemble_summary_path.exists():
            print(f"[ensemble] summary JSON: {ensemble_summary_path}", flush=True)
        manifest.write(json.dumps({
            "stage": "ind_r_ensemble_eval",
            "eval_rc": ensemble_eval_rc,
            "summary_json": str(ensemble_summary_path),
            "instances_csv": str(ensemble_instance_csv) if ensemble_instance_csv else "",
            "member_ant_counts": ant_counts,
            "checkpoints": [str(p) for p in eval_ckpts],
        }, sort_keys=True) + "\n")

        if args.multihead_checkpoint is not None:
            summary_path = out_dir / "multihead_summary.json"
            eval_cmd = build_eval_cmd(
                args,
                args.multihead_checkpoint,
                summary_path,
                ants=args.eval_ants,
                seed=args.seed,
                multi_head=True,
            )
            commands.write(" ".join(eval_cmd) + "\n")
            _run(
                eval_cmd,
                ROOT,
                log_dir / "multihead_eval.log",
                args.run,
                label="multihead comparison eval",
                stream_output=args.stream_output,
            )

    readme = out_dir / "SUMMARY.md"
    mode = "executed" if args.run else "dry-run"
    summary_lines = [
        f"# Independent-Policy Ensemble Ablation ({mode})",
        "",
        f"- Train validation dataset: `{args.train_dataset}`",
        f"- Test dataset: `{args.test_dataset}`",
        f"- Members: `{args.members}`",
        f"- Total eval ants: `{args.eval_ants}`",
        f"- Member ant counts: `{','.join(str(x) for x in ant_counts)}`",
        f"- Commands: `commands.log`",
        f"- Manifest: `manifest.jsonl`",
        f"- Ensemble summary JSON: `ind_r_summary.json`",
        f"- Ensemble instance CSV: `ind_r_instances.csv`",
    ]
    readme.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if shutil.which("uv") is None:
        print("Warning: uv not found on PATH; commands were still written.")
    print(f"[ind-r] wrote {mode} artifacts to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
