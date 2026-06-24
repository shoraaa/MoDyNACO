#!/usr/bin/env python3
"""Visualize head diversity from state_recurrence_experiment.py outputs.

Reads one or more classifier_metrics.json files (each produced by a
--behavior-classification run on a single domain) and produces one PDF
per heatmap — no cell text, colour scale only.

Output files (in --output-dir):
  cosine_{label}.pdf      – residual cosine matrix
  winstep_{label}.pdf     – row-normalised win-step distribution

Usage:
  python scripts/visualize_head_diversity.py \\
      results/state_recurrence/uniform1k/*/classifier_metrics.json \\
      results/state_recurrence/clustered16k/*/classifier_metrics.json \\
      -o figures/head_diversity
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize


def load_run(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    bm = data.get("behavior_classifier", data)
    config_path = path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    domains = config.get("domains", ["unknown"])
    label = ", ".join(domains) if isinstance(domains, list) else str(domains)
    return {
        "label": label,
        "cosine": np.array(bm["cosine_similarity_matrix"]),
        "win_matrix": np.array(bm["win_matrix"]),
        "mean_offdiag_cosine": bm.get("mean_offdiag_cosine"),
        "cramers_v": bm.get("cramers_v"),
        "test_accuracy": bm.get("test_accuracy"),
        "majority_chance": bm.get("majority_chance"),
    }


def cross_domain_corr(runs: List[Dict[str, Any]]) -> Optional[float]:
    if len(runs) < 2:
        return None
    mask = ~np.eye(runs[0]["cosine"].shape[0], dtype=bool)
    a = runs[0]["cosine"][mask]
    b = runs[1]["cosine"][mask]
    return float(np.corrcoef(a, b)[0, 1])


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def save_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    cmap: str,
    norm,
    path: Path,
    dpi: int,
    cbar_label: str = "",
):
    n_rows, n_cols = mat.shape
    cell = 0.52
    pad_left = 0.4
    pad_top = 0.4
    cbar_w = 0.18
    cbar_gap = 0.15
    fig_w = pad_left + n_cols * cell + cbar_gap + cbar_w + 0.15
    fig_h = pad_top + n_rows * cell + 0.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=pad_left / fig_w,
        right=(pad_left + n_cols * cell) / fig_w,
        bottom=0.15 / fig_h,
        top=(fig_h - pad_top) / fig_h,
    )

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()

    cbar_left = (pad_left + n_cols * cell + cbar_gap) / fig_w
    cbar_bottom = 0.15 / fig_h
    cbar_height = (n_rows * cell) / fig_h
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_w / fig_w, cbar_height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize head diversity across domains")
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="classifier_metrics.json files from --behavior-classification runs")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("figures/head_diversity"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    runs = [load_run(p) for p in args.inputs]
    n_heads = runs[0]["cosine"].shape[0]
    H = runs[0]["win_matrix"].shape[0]
    head_labels = [f"h{i}" for i in range(n_heads)]
    step_labels = [f"t{i}" for i in range(H)]

    cos_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    win_max = 0.0
    for r in runs:
        row_sums = r["win_matrix"].sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0
        r["win_norm"] = r["win_matrix"] / row_sums
        win_max = max(win_max, float(r["win_norm"].max()))
    win_max = max(win_max, 0.01)
    win_norm = Normalize(vmin=0, vmax=win_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for r in runs:
        slug = _slug(r["label"])

        cos_path = args.output_dir / f"cosine_{slug}.pdf"
        save_heatmap(r["cosine"], head_labels, head_labels,
                     "RdBu_r", cos_norm, cos_path, args.dpi,
                     cbar_label="residual cosine")
        print(f"  {cos_path}")

        win_path = args.output_dir / f"winstep_{slug}.pdf"
        save_heatmap(r["win_norm"], step_labels, head_labels,
                     "Blues", win_norm, win_path, args.dpi,
                     cbar_label="win fraction")
        print(f"  {win_path}")

    corr = cross_domain_corr(runs)
    print()
    for r in runs:
        print(f"  {r['label']:30s}  behavior={r['test_accuracy']:.2%}  "
              f"off-diag cos={r['mean_offdiag_cosine']:.3f}  "
              f"Cramér V={r['cramers_v']:.3f}")
    if corr is not None:
        print(f"  cross-domain matrix corr = {corr:.4f}")


if __name__ == "__main__":
    main()
