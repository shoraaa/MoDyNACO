# Config Layout

Configs are grouped by how they are used:

- `train/`: training runs for TSP, CVRP, STAR-style subsets, and multi-head variants.
- `eval/`: checkpoint evaluation configs used for TSPLIB, CVRPLIB, STAR subsets, and journal-depth runs.
- `smoke/`: small or short pipeline checks.
- `extended/`: non-routing extended problems (`bpp`, `mkp`, `op`).
- `archive/`: older exploratory variants kept for reference, not recommended as entry points.

Recommended starting points:

- TSP training: `configs/train/tsp_n1000_ppo.yaml`
- CVRP training: `configs/train/cvrp_n1000_ppo.yaml`
- TSP full evaluation: `configs/eval/tsp_n1000_original_full_eval.yaml`
- CVRP full evaluation: `configs/eval/cvrp_n1000_original_cvrlib_eval.yaml`
- CVRP n=500 PolyNet multi-head training:
  - `configs/train/cvrp_n500_multihead_polynet_train.yaml`
- Quick smoke test: `configs/smoke/test_tsp_n100_ppo.yaml`
