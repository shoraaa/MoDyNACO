# Multi-Head Decoder Config Sweep

These configs compare decoder capacity while keeping the TSP `<1K` dataset and
ACO budget fixed. All variants use `head_adapter_init: random`, so every head
starts with a normally randomized head-specific adapter/code instead of
preserving an anchored base head or forcing heads to stay close to the base.

## Train One Variant

```bash
uv run python train.py --config configs/train/decoder_sweep/tsp_lt1k_deep_lora_r4.yaml
```

## Evaluate The Same Variant

```bash
uv run python test.py --config configs/eval/decoder_sweep/tsp_lt1k_deep_lora_r4.yaml
```

## Train All Variants

```bash
for cfg in configs/train/decoder_sweep/tsp_lt1k_*.yaml; do
  uv run python train.py --config "$cfg"
done
```

## Evaluate All Variants

```bash
for cfg in configs/eval/decoder_sweep/tsp_lt1k_*.yaml; do
  uv run python test.py --config "$cfg"
done
```

## Analyze Results

```bash
uv run python - <<'PY'
import json
from pathlib import Path

rows = []
for path in sorted(Path("results/multihead_decoder_configs/eval").glob("*_summary.json")):
    payload = json.loads(path.read_text())
    method = payload.get("methods", {}).get("model_anneal", {})
    diag = method.get("guidance_diagnostics", {})
    rows.append({
        "variant": path.stem.replace("_summary", ""),
        "gap_pct": method.get("gap_pct"),
        "mean_cost": method.get("mean_cost"),
        "mean_time_s": method.get("mean_time_s"),
        "head_logit_corr": diag.get("head_logit_corr"),
        "head_topk_overlap": diag.get("head_topk_overlap"),
        "head_selected_edge_overlap": diag.get("head_selected_edge_overlap"),
        "head_best_fraction": diag.get("head_best_fraction"),
    })

rows.sort(key=lambda r: (float("inf") if r["gap_pct"] is None else r["gap_pct"],
                         float("inf") if r["mean_cost"] is None else r["mean_cost"]))
for r in rows:
    print(r)
PY
```

Read quality first (`gap_pct`, `mean_cost`), then diversity:
`head_logit_corr`, `head_topk_overlap`, and `head_selected_edge_overlap`.
If quality improves but `head_best_fraction` is concentrated on one head, the
variant is behaving more like a stronger single-head decoder than useful
multi-head diversity.
