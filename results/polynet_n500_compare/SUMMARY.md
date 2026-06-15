# CVRP n=500 Weighted Heads vs PolyNet

## Setup

- Training budget: `H=2`, `mini_H=20`, `epochs=3`, `steps_per_epoch=8`
- Evaluation budget: `H=2`, `mini_H=20`
- Evaluation set: first 10 instances from `data/CVRP/data/test_set/STAR_CVRPlib_lt1000.txt`
- Seed: `1234`
- Backend: `faco`, `k_sparse=32`, `n_ants=100`

## Training

| Mode | Best epoch | Best validation cost | Best validation gap | Train time |
|---|---:|---:|---:|---:|
| Weighted heads | 2 | 33.9438 | 3.9259% | 25.89s |
| PolyNet best-head | 1 | 33.8209 | 3.5490% | 23.17s |

## Test

| Mode | Mean cost | Gap | Gap std | Mean time |
|---|---:|---:|---:|---:|
| Weighted heads | 22.9532 | 3.0149% | 1.4292% | 0.0962s |
| PolyNet best-head | 22.9789 | 2.9129% | 1.5195% | 0.0929s |

## Artifacts

- Weighted checkpoint: `pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_mh4_hg2_hdants_weighted_compare_best.pt`
- PolyNet checkpoint: `pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_mh4_hg2_hdants_htrainpolynet_polynet_best.pt`
- Weighted summary: `results/polynet_n500_compare/weighted_summary.json`
- PolyNet summary: `results/polynet_n500_compare/polynet_summary.json`
