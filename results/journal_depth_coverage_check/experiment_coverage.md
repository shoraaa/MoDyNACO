# Journal-Depth Experiment Coverage

This file states what the runner covers versus what still needs evaluator instrumentation.

| Planned experiment | Status in this runner | Evidence produced |
|---|---|---|
| Allocation sweep under matched budget | Covered | Summary JSONs for single-head and multi-head allocations `100,0,0,0`, `97,1,1,1`, `85,5,5,5`, `70,10,10,10`, `50,20,15,15`, `25,25,25,25` on full TSPLIB and full CVRPLIB. |
| Per-instance win/loss analysis | Partially covered | Use `--log`; evaluator writes per-instance CSVs and the run log records their paths. A separate parser can aggregate win/loss/tie from those CSVs. |
| Runtime decomposition | Partially covered | Use `--stage-metrics`; evaluator summary/logs expose existing neural/sampling/local-search/update timing where available. It is not a full profiler. |
| Stagnation / anti-stagnation diagnostic | Partially covered | Use `--guidance-metrics`; current evaluator records aggregate guidance and pheromone-correlation metrics, not the full Enhance/Suppress top-q diagnostic. |
| Train-size sensitivity | Partially covered | CVRP n=500 split runs are included. TSP n=500 train-size comparison is not included because a matched n=500 TSP full-library checkpoint/config is not encoded here. |
| Head contribution by ant group | Not covered yet | Needs evaluator/backend instrumentation to return per-head ant costs, best-ant ownership, and improvement ownership at each guidance step. |
| Head similarity / collapse analysis | Not covered yet | Needs exported per-head heatmaps or logits to compute cosine/rank similarity and selected-edge overlap. |

Recommended next code work:

1. Add per-head ant-cost and best-ant ownership logging inside the mixed-head inference path.
2. Export per-head sparse logits for sampled guidance steps.
3. Add a parser that joins per-instance CSVs with summary JSONs and produces win/loss/tie tables by problem, size bucket, and allocation.
