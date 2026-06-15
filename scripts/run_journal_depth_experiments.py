#!/usr/bin/env python3
"""Run the journal-depth DyNACO experiment plan.

The script is intentionally conservative:
  * dry-run is the default;
  * each run writes a separate summary JSON and stdout/stderr log;
  * existing summaries are skipped unless --force is passed;
  * failures are recorded and the remaining runs continue.

The completed experiments directly support part of the paper's deeper analysis:
  1. allocation sweep under a fixed ant budget;
  2. per-instance win/loss evidence via --log CSVs;
  3. guidance/stage diagnostics via existing evaluator metrics;
  4. train-size sensitivity for the currently available n=500 CVRP runs.

They use evaluator instrumentation for per-head contribution and heatmap
similarity. They still do not export sampled edge sets per head, so selected
edge overlap remains approximated by head-prior top-k overlap.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


TSP_SINGLE_CKPT = (
    "pretrained/tsp/n1000/"
    "tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt"
)
TSP_MH_CKPT = (
    "pretrained/tsp/n1000/"
    "tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_"
    "mh4_hg0_hdants_ha25-25-25-25_anchor1_balanced_anchor_best.pt"
)
CVRP_SINGLE_CKPT = (
    "pretrained/cvrp/n1000/"
    "cvrp_n1000_k32_ants100_H10_miniH100_rho0.5_mne12_ppo_lr5e-06_best.pt"
)
CVRP_MH_CKPT = (
    "pretrained/cvrp/n1000/"
    "cvrp_n1000_k32_ants100_H10_miniH100_rho0.5_mne12_ppo_lr5e-06_"
    "mh4_hg2_hdants_ha85-5-5-5_anchor1_anchored_best.pt"
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    stage: str
    config: str
    checkpoint: str
    summary_name: str
    head_weights: str | None = None
    notes: str = ""
    extra_args: tuple[str, ...] = ()

    def command(self, out_dir: Path, args: argparse.Namespace) -> list[str]:
        cmd = [
            "uv",
            "run",
            "test.py",
            "--config",
            self.config,
            "--checkpoint",
            self.checkpoint,
            "--summary_json",
            str(out_dir / self.summary_name),
        ]
        if self.head_weights:
            cmd += ["--head_ant_weights", self.head_weights]
        if args.log:
            cmd.append("--log")
        if args.iter_log:
            cmd.append("--iter_log")
        if args.stage_metrics:
            cmd.append("--stage_metrics")
        if args.guidance_metrics:
            cmd.append("--collect_guidance_metrics")
        if args.timed:
            cmd.append("--timed")
        if args.val_size is not None:
            cmd += ["--val_size", str(args.val_size)]
        if args.device:
            cmd += ["--device", args.device]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        cmd.extend(self.extra_args)
        return cmd

    def summary_path(self, out_dir: Path) -> Path:
        return out_dir / self.summary_name


def slug_weights(weights: str) -> str:
    return weights.replace(",", "-")


def allocation_runs() -> list[RunSpec]:
    weights = ["100,0,0,0", "97,1,1,1", "85,5,5,5", "70,10,10,10", "50,20,15,15", "25,25,25,25"]
    runs: list[RunSpec] = [
        RunSpec(
            name="tsp_full_single",
            stage="allocation",
            config="configs/eval/tsp_n1000_original_full_eval.yaml",
            checkpoint=TSP_SINGLE_CKPT,
            summary_name="tsp_full_single_summary.json",
            notes="Matched single-head baseline for full TSPLIB.",
        ),
        RunSpec(
            name="cvrp_full_single",
            stage="allocation",
            config="configs/eval/cvrp_n1000_original_cvrlib_eval.yaml",
            checkpoint=CVRP_SINGLE_CKPT,
            summary_name="cvrp_full_single_summary.json",
            notes="Matched single-head baseline for full CVRPLIB.",
        ),
    ]
    for w in weights:
        s = slug_weights(w)
        runs.append(
            RunSpec(
                name=f"tsp_full_mh_{s}",
                stage="allocation",
                config="configs/eval/tsp_n1000_multihead_anchor_full_eval.yaml",
                checkpoint=TSP_MH_CKPT,
                head_weights=w,
                summary_name=f"tsp_full_mh_ha{s}_summary.json",
                notes="Full TSPLIB allocation sweep using the balanced-anchor TSP multi-head checkpoint.",
            )
        )
        runs.append(
            RunSpec(
                name=f"cvrp_full_mh_{s}",
                stage="allocation",
                config="configs/eval/cvrp_n1000_multihead_anchor_cvrlib_eval.yaml",
                checkpoint=CVRP_MH_CKPT,
                head_weights=w,
                summary_name=f"cvrp_full_mh_ha{s}_summary.json",
                notes="Full CVRPLIB allocation sweep using the anchored CVRP multi-head checkpoint.",
            )
        )
    return runs


def train_size_runs() -> list[RunSpec]:
    return [
        RunSpec(
            name="cvrp_lt1000_n500_single",
            stage="train_size",
            config="configs/eval/cvrp_n500_original_cvrlib_lt1000_eval.yaml",
            checkpoint="pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_best.pt",
            summary_name="cvrp_lt1000_n500_single_summary.json",
            notes="CVRP n=500 train-size split, <1K.",
        ),
        RunSpec(
            name="cvrp_lt1000_n500_mh",
            stage="train_size",
            config="configs/eval/cvrp_n500_multihead_cvrlib_lt1000_eval.yaml",
            checkpoint="pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_mh4_hg2_hdants_best.pt",
            summary_name="cvrp_lt1000_n500_mh_summary.json",
            notes="CVRP n=500 multi-head train-size split, <1K.",
        ),
        RunSpec(
            name="cvrp_ge1000_n500_single",
            stage="train_size",
            config="configs/eval/cvrp_n500_original_cvrlib_ge1000_eval.yaml",
            checkpoint="pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_best.pt",
            summary_name="cvrp_ge1000_n500_single_summary.json",
            notes="CVRP n=500 train-size split, >=1K.",
        ),
        RunSpec(
            name="cvrp_ge1000_n500_mh",
            stage="train_size",
            config="configs/eval/cvrp_n500_multihead_cvrlib_ge1000_eval.yaml",
            checkpoint="pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_mh4_hg2_hdants_best.pt",
            summary_name="cvrp_ge1000_n500_mh_summary.json",
            notes="CVRP n=500 multi-head train-size split, >=1K.",
        ),
    ]


def all_runs() -> list[RunSpec]:
    return allocation_runs() + train_size_runs()


def selected_runs(args: argparse.Namespace) -> list[RunSpec]:
    runs = all_runs()
    if args.stage != "all":
        runs = [r for r in runs if r.stage == args.stage]
    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r.name in wanted]
    if args.exclude:
        blocked = set(args.exclude)
        runs = [r for r in runs if r.name not in blocked]
    return runs


def check_inputs(runs: Iterable[RunSpec]) -> list[str]:
    missing: list[str] = []
    for run in runs:
        for rel in [run.config, run.checkpoint]:
            path = ROOT / rel
            if not path.exists():
                missing.append(rel)
    return sorted(set(missing))


def metric_from_summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    method = data.get("methods", {}).get("model_anneal", {})
    return {
        "gap_pct": method.get("gap_pct"),
        "mean_cost": method.get("mean_cost"),
        "mean_time_s": method.get("mean_time_s"),
        "reliability": method.get("reliability") or method.get("solved"),
    }


def write_summary_table(out_dir: Path) -> None:
    rows = []
    for path in sorted(out_dir.glob("*_summary.json")):
        try:
            metrics = metric_from_summary(path)
        except Exception as exc:  # noqa: BLE001
            rows.append((path.name, f"ERROR: {exc}", "", ""))
            continue
        rows.append(
            (
                path.name,
                metrics.get("gap_pct"),
                metrics.get("mean_cost"),
                metrics.get("mean_time_s"),
            )
        )
    md = out_dir / "summary_table.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("| Summary | Gap % | Mean Cost | Mean Time s |\n")
        f.write("|---|---:|---:|---:|\n")
        for name, gap, cost, time_s in rows:
            f.write(f"| `{name}` | {gap} | {cost} | {time_s} |\n")


def _extract_csv_path(log_path: Path, label: str) -> Path | None:
    if not log_path.exists():
        return None
    pattern = re.compile(rf"CSV {re.escape(label)}:\s+(.+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            p = Path(m.group(1).strip())
            return p if p.is_absolute() else ROOT / p
    return None


def copy_run_csvs(run_name: str, log_path: Path, out_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    csv_out = out_dir / "per_instance"
    csv_out.mkdir(parents=True, exist_ok=True)
    for label, suffix in [("instances", "instances"), ("summary", "summary"), ("iters", "iters")]:
        src = _extract_csv_path(log_path, label)
        if src is None or not src.exists():
            continue
        dst = csv_out / f"{run_name}_{suffix}.csv"
        shutil.copy2(src, dst)
        copied[f"csv_{suffix}"] = str(dst)
    return copied


def write_flat_per_instance_table(out_dir: Path) -> None:
    csv_dir = out_dir / "per_instance"
    rows = []
    if csv_dir.exists():
        for path in sorted(csv_dir.glob("*_instances.csv")):
            run_name = path.name[: -len("_instances.csv")]
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    opt = row.get("opt")
                    cost = row.get("model_anneal")
                    try:
                        opt_f = float(opt) if opt not in (None, "", "None") else None
                        cost_f = float(cost) if cost not in (None, "", "None") else None
                        gap = 100.0 * (cost_f - opt_f) / opt_f if opt_f and cost_f is not None else None
                    except Exception:
                        gap = None
                    rows.append({
                        "run": run_name,
                        "idx": row.get("idx"),
                        "name": row.get("name"),
                        "opt": opt,
                        "model_anneal": cost,
                        "gap_pct": "" if gap is None else f"{gap:.8f}",
                    })
    out = out_dir / "per_instance_runs.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run", "idx", "name", "opt", "model_anneal", "gap_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_report(out_dir: Path) -> None:
    report = out_dir / "experiment_coverage.md"
    report.write_text(
        """# Journal-Depth Experiment Coverage

This file states what the runner covers versus what still needs evaluator instrumentation.

| Planned experiment | Status in this runner | Evidence produced |
|---|---|---|
| Allocation sweep under matched budget | Covered | Summary JSONs for single-head and multi-head allocations `100,0,0,0`, `97,1,1,1`, `85,5,5,5`, `70,10,10,10`, `50,20,15,15`, `25,25,25,25` on full TSPLIB and full CVRPLIB. |
| Per-instance win/loss analysis | Covered as raw evidence | Use `--log`; the runner copies evaluator per-instance CSVs into `per_instance/` and writes `per_instance_runs.csv` for downstream win/loss pivots. |
| Runtime decomposition | Covered by existing timings | Use `--timed --stage-metrics`; evaluator summaries expose neural/sampling/local-search/update timing where available plus stage costs. |
| Stagnation / anti-stagnation diagnostic | Covered for top-k candidate edges | Use `--guidance-metrics`; summaries include enhance, rebellion, and suppression metrics against pheromone, plus per-head versions for multi-head runs. |
| Train-size sensitivity | Partially covered | CVRP n=500 split runs are included. TSP n=500 train-size comparison is not included because a matched n=500 TSP full-library checkpoint/config is not encoded here. |
| Head contribution by ant group | Covered for mixed-head runs | Summaries include `head_best_fraction`, `head_improvement_fraction`, and per-head mean/best ant costs before and after local projection. |
| Head similarity / collapse analysis | Covered for heatmaps | Summaries include mean pairwise head-logit correlation and head-prior top-k overlap. Sampled selected-edge overlap still needs backend trace export. |

Recommended next code work:

1. Add backend trace export for sampled edge sets per ant if selected-edge overlap must be measured directly.
2. Add a higher-level parser that pivots `per_instance_runs.csv` into paper-ready win/loss/tie tables by problem, size bucket, and allocation.
""",
        encoding="utf-8",
    )


def run_one(run: RunSpec, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    summary = run.summary_path(out_dir)
    log_path = out_dir / "logs" / f"{run.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = run.command(out_dir, args)
    record: dict[str, object] = {
        "name": run.name,
        "stage": run.stage,
        "summary": str(summary),
        "log": str(log_path),
        "command": cmd,
        "notes": run.notes,
    }
    if summary.exists() and not args.force:
        record["status"] = "skipped_existing"
        return record
    if args.dry_run:
        record["status"] = "dry_run"
        return record

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    record["returncode"] = proc.returncode
    record["status"] = "ok" if proc.returncode == 0 and summary.exists() else "failed"
    record.update(copy_run_csvs(run.name, log_path, out_dir))
    return record


def print_plan(runs: list[RunSpec], out_dir: Path, args: argparse.Namespace) -> None:
    print(f"Output directory: {out_dir}")
    print(f"Runs selected: {len(runs)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Coverage report: {out_dir / 'experiment_coverage.md'}")
    print()
    for i, run in enumerate(runs, 1):
        cmd = run.command(out_dir, args)
        print(f"[{i:02d}] {run.stage}: {run.name}")
        if run.notes:
            print(f"     {run.notes}")
        print("     " + " ".join(cmd))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run journal-depth DyNACO experiments: allocation sweep, per-instance logs, guidance metrics, and train-size sensitivity."
    )
    parser.add_argument("--stage", choices=["all", "allocation", "train_size"], default="allocation")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: results/journal_depth/<timestamp>")
    parser.add_argument("--run", dest="dry_run", action="store_false", help="Execute commands. Default is dry-run.")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--force", action="store_true", help="Re-run even when a summary JSON already exists.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these run names.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Skip these run names.")
    parser.add_argument("--val-size", type=int, default=None, help="Limit validation instances for smoke tests.")
    parser.add_argument("--device", default=None, help="Override evaluator device, e.g. cuda:0.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log", action="store_true", help="Enable evaluator per-instance CSV and console log capture.")
    parser.add_argument("--iter-log", action="store_true", help="Enable per-iteration CSV logging. Implies long outputs.")
    parser.add_argument("--stage-metrics", action="store_true", help="Collect evaluator stage metrics for runtime/quality diagnostics.")
    parser.add_argument("--guidance-metrics", action="store_true", help="Collect guidance/pheromone correlation metrics.")
    parser.add_argument("--timed", action="store_true", help="Enable evaluator backend timing breakdown.")
    parser.add_argument("--no-input-check", action="store_true", help="Do not fail early on missing configs/checkpoints.")
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild summary_table.md from existing summary JSONs.")
    parser.add_argument("--coverage-only", action="store_true", help="Only write experiment_coverage.md and exit.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "results" / "journal_depth" / timestamp)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_coverage_report(out_dir)

    if args.coverage_only:
        print(f"Wrote {out_dir / 'experiment_coverage.md'}")
        return 0

    if args.summarize_only:
        write_summary_table(out_dir)
        write_flat_per_instance_table(out_dir)
        print(f"Wrote {out_dir / 'summary_table.md'}")
        print(f"Wrote {out_dir / 'per_instance_runs.csv'}")
        return 0

    runs = selected_runs(args)
    if not runs:
        print("No runs selected.", file=sys.stderr)
        return 2

    missing = check_inputs(runs)
    if missing and not args.no_input_check:
        print("Missing required inputs:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2

    print_plan(runs, out_dir, args)
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for run in runs:
            record = run_one(run, out_dir, args)
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
            manifest.flush()
            print(f"{record['status']}: {run.name}")

    write_summary_table(out_dir)
    write_flat_per_instance_table(out_dir)
    write_coverage_report(out_dir)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary table: {out_dir / 'summary_table.md'}")
    print(f"Wrote per-instance table: {out_dir / 'per_instance_runs.csv'}")
    print(f"Wrote coverage report: {out_dir / 'experiment_coverage.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
