# Journal-Depth Experiment Coverage

This file states what the runner covers versus what still needs evaluator instrumentation.

| Planned experiment | Status in this runner | Evidence produced |
|---|---|---|
| Allocation sweep under matched budget | Covered | Summary JSONs for single-head and multi-head allocations `100,0,0,0`, `97,1,1,1`, `85,5,5,5`, `70,10,10,10`, `50,20,15,15`, `25,25,25,25` on full TSPLIB and full CVRPLIB. |
| Adaptive prior allocation | Covered as an opt-in stage | Use `--stage adaptive_allocation`; runs start from `25,25,25,25` and enable `--head_router ema --head_router_alpha 0.25 --head_router_min_frac 0.05`. Summary payloads include router counts and utilities when guidance metrics are collected. |
| Per-instance win/loss analysis | Covered as raw evidence | Use `--log`; the runner copies evaluator per-instance CSVs into `per_instance/` and writes `per_instance_runs.csv` for downstream win/loss pivots. |
| Runtime decomposition | Covered by existing timings | Use `--timed --stage-metrics`; evaluator summaries expose neural/sampling/local-search/update timing where available plus stage costs. |
| Stagnation / anti-stagnation diagnostic | Covered for top-k candidate edges | Use `--guidance-metrics`; summaries include enhance, rebellion, and suppression metrics against pheromone, plus per-head versions for multi-head runs. |
| Train-size sensitivity | Partially covered | CVRP n=500 split runs are included. TSP n=500 train-size comparison is not included because a matched n=500 TSP full-library checkpoint/config is not encoded here. |
| Head contribution by ant group | Covered for mixed-head runs | Summaries include `head_best_fraction`, `head_improvement_fraction`, and per-head mean/best ant costs before and after local projection. |
| Head similarity / collapse analysis | Covered for heatmaps | Summaries include mean pairwise head-logit correlation and head-prior top-k overlap. Sampled selected-edge overlap still needs backend trace export. |

Recommended next code work:

1. Add backend trace export for sampled edge sets per ant if selected-edge overlap must be measured directly.
2. Add a higher-level parser that pivots `per_instance_runs.csv` into paper-ready win/loss/tie tables by problem, size bucket, and allocation.
