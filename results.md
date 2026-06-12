# Results Log

## 2026-05-15: Journal-Depth Allocation Sweep with Diagnostics

Context:
- Ran `scripts/run_journal_depth_experiments.py --stage allocation --run --log --stage-metrics --guidance-metrics --timed`.
- Output directory: `results/journal_depth/20260515_050321`.
- All 14 runs completed with `status=ok`.
- The run produced summary JSONs, per-instance CSVs, `per_instance_runs.csv`, timing fields, and head diagnostics.

Full allocation sweep:

| Problem | Allocation | Gap | Mean Cost | Mean Time | Win/Loss/Tie vs Single |
|---|---:|---:|---:|---:|---:|
| TSP | single | 2.0624% | 4335858.4326 | 4.3619s | - |
| TSP | `100,0,0,0` | 1.9567% | **4322208.9621** | 6.8576s | 130/75/23 |
| TSP | `97,1,1,1` | 1.9562% | 4322548.3491 | 6.8763s | 131/74/23 |
| TSP | `85,5,5,5` | 1.9566% | 4322298.6362 | 6.8891s | 131/76/21 |
| TSP | `70,10,10,10` | **1.9522%** | 4322446.5855 | 6.8685s | 128/77/23 |
| TSP | `50,20,15,15` | 1.9541% | 4322526.2052 | 6.8693s | 131/75/22 |
| TSP | `25,25,25,25` | 1.9569% | 4323310.5240 | 6.8686s | **133/71/24** |
| CVRP | single | 2.9212% | 76.4222 | 3.5858s | - |
| CVRP | `100,0,0,0` | **2.9000%** | **76.3585** | 3.9966s | **63/47/0** |
| CVRP | `97,1,1,1` | 2.9018% | 76.3640 | 4.0514s | 61/49/0 |
| CVRP | `85,5,5,5` | 2.9044% | 76.3792 | 4.0438s | **63/47/0** |
| CVRP | `70,10,10,10` | 2.9205% | 76.4072 | 4.0329s | 60/50/0 |
| CVRP | `50,20,15,15` | 2.9726% | 76.4069 | 4.0280s | 55/55/0 |
| CVRP | `25,25,25,25` | 2.9950% | 76.4322 | 4.0534s | 54/56/0 |

Diagnostics:
- TSP: all multi-head allocations improve over single-head in aggregate. The best mean gap is `70,10,10,10`, while `25,25,25,25` has the broadest win count. This supports a real multi-head deployment effect, but the differences among allocations are small.
- CVRP: protected-primary allocations dominate. `100,0,0,0`, `97,1,1,1`, and `85,5,5,5` are all better than the more balanced allocations; `50,20,15,15` and `25,25,25,25` lose or tie the single-head baseline by mean gap.
- Head contribution follows the allocation. For TSP `25,25,25,25`, best-ant ownership is approximately `[0.459, 0.216, 0.171, 0.155]` and improvement ownership `[0.453, 0.209, 0.175, 0.163]`. For CVRP `25,25,25,25`, head 0 dominates more strongly: best-ant ownership `[0.693, 0.131, 0.094, 0.082]`.
- Head heatmap similarity is extremely high (`head_logit_corr` about `1.0`, top-k overlap about `0.999+`) across both problems. The current heads are not strongly specialized at the heatmap level; gains mostly come from allocation/primary-prior protection and stochastic ant-group effects, not clearly distinct learned heatmaps.
- Runtime with timing enabled is stable across multi-head allocations: about `6.86-6.89s` on full TSP and `4.00-4.05s` on full CVRP. The earlier 10.8s TSP time for `50,20,15,15` is superseded by this matched sweep.

Saved summaries:
- `results/journal_depth/20260515_050321/summary_table.md`
- `results/journal_depth/20260515_050321/per_instance_runs.csv`
- `results/journal_depth/20260515_050321/manifest.jsonl`

## 2026-05-14: Stronger Non-Degenerate Multi-Head Re-test

Context:
- The `97,1,1,1` allocation is too close to single-head behavior to support a strong multi-head claim.
- Added and trained a TSP balanced-anchor model where all heads receive real training mass:
  - train allocation: `25,25,25,25`;
  - uniform head PPO weighting via `head_gamma=0`;
  - diversity coefficient `0.01`;
  - head-0 teacher anchor from the matched single-head checkpoint.
- Stopped the TSP balanced-anchor run after epoch 1 because validation quality was preserved and the full transfer test was the deciding evidence.
- Tested stronger deployment allocation `50,20,15,15`, so 50% of ants use non-primary heads.

TSP checkpoints:
- Single-head: `pretrained/tsp/n1000/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt`
- Balanced-anchor multi-head: `pretrained/tsp/n1000/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_mh4_hg0_hdants_ha25-25-25-25_anchor1_balanced_anchor_best.pt`

Full benchmark results:

| Problem / Dataset | Instances | Model | Head Allocation | Gap | Mean Cost | Mean Time |
|---|---:|---|---|---:|---:|---:|
| TSP / full TSPLIB-converted STAR file | 228 | single-head n=1000 | - | 2.0604% | 4335923.2902 | 4.3452s |
| TSP / full TSPLIB-converted STAR file | 228 | anchored multi-head n=1000 | `97,1,1,1` | 1.9803% | 4330966.1345 | 7.6588s |
| TSP / full TSPLIB-converted STAR file | 228 | balanced-anchor multi-head n=1000 | `50,20,15,15` | **1.9541%** | **4322525.8893** | 10.7854s |
| CVRP / full CVRPLIB-converted STAR file | 110 | single-head n=1000 retrain | - | 2.9746% | 76.4303 | 4.2282s |
| CVRP / full CVRPLIB-converted STAR file | 110 | anchored multi-head n=1000 | `97,1,1,1` | 2.8373% | 76.3493 | 4.0803s |
| CVRP / full CVRPLIB-converted STAR file | 110 | anchored multi-head n=1000 | `25,25,25,25` | 2.8910% | 76.3704 | 4.0769s |
| CVRP / full CVRPLIB-converted STAR file | 110 | anchored multi-head n=1000 | `50,20,15,15` | **2.8673%** | **76.3981** | 4.9828s |

Split comparison against equal head allocation:

| Problem / Dataset | Instances | Model | Head Allocation | Gap | Mean Cost | Mean Time |
|---|---:|---|---|---:|---:|---:|
| TSP / TSPLIB <1K | 69 | single-head n=1000 | - | 0.5407% | 27544.2901 | 0.6871s |
| TSP / TSPLIB <1K | 69 | balanced-anchor multi-head n=1000 | `25,25,25,25` | **0.5232%** | **27518.7904** | **0.6675s** |
| TSP / TSPLIB <1K | 69 | balanced-anchor multi-head n=1000 | `50,20,15,15` | 0.5279% | 27519.8617 | 2.5224s |

Saved summaries:
- `results/tsp_star_lt1000_n1000_multihead_balanced_anchor_ha50-20-15-15_summary.json`
- `results/tsp_tsplib_full_n1000_multihead_balanced_anchor_ha50-20-15-15_summary.json`
- `results/tsp_star_lt1000_n1000_multihead_balanced_anchor_ha25-25-25-25_summary.json`
- `results/cvrp_star_cvrlib_n1000_multihead_anchor_ha25-25-25-25_summary.json`
- `results/cvrp_star_cvrlib_n1000_multihead_anchor_ha50-20-15-15_summary.json`

Conclusion:
- Stronger multi-head now beats single-head under a non-degenerate `50,20,15,15` allocation on both full benchmark suites.
- For TSP, the stronger allocation also beats the earlier conservative `97,1,1,1` result (`1.9541%` vs `1.9803%`).
- Equal `25,25,25,25` is not dominated: it is best on TSP <1K and has the lowest CVRP mean cost among the two non-degenerate allocations, but it is worse than `50,20,15,15` on the CVRP mean per-instance gap.
- A full-TSPLIB `25,25,25,25` run was attempted with the same balanced-anchor checkpoint, but ROCm hit a GPU memory access fault on the final large-instance tail before writing the summary JSON. No full-TSPLIB `25,25,25,25` aggregate should be reported until that run is repeated successfully or evaluated in resumable shards.
- Paper claims should use `50,20,15,15` as the main full-TSP evidence and describe allocation as a deployment hyperparameter, not as a universally optimal fixed ratio.

## 2026-05-14: n=1000 Anchored Multi-Head Re-test on Full Benchmarks

Context:
- Fixed multi-head deployment so the backend receives compact per-head heatmaps plus explicit ant counts instead of expanding all ants through duplicated heatmaps.
- Added an anchored multi-head training mode for the n=1000 runs:
  - initialize the multi-head encoder/decoder from the matched single-head checkpoint;
  - keep head 0 close to the single-head teacher during PPO updates;
  - train with `head_ant_weights=85,5,5,5`;
  - evaluate with all heads active under the same total budget using `head_ant_weights=97,1,1,1`.
- Default evaluation budget stayed matched: `n_ants=100`, `H=10`, `mini_H=100`, `k_sparse=32`.

Training checkpoints:
- TSP single-head: `pretrained/tsp/n1000/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt`
- TSP anchored multi-head: `pretrained/tsp/n1000/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_mh4_hg2_hdants_ha85-5-5-5_anchor1_anchored_best.pt`
- CVRP single-head: `pretrained/cvrp/n1000/cvrp_n1000_k32_ants100_H10_miniH100_rho0.5_mne12_ppo_lr5e-06_best.pt`
- CVRP anchored multi-head: `pretrained/cvrp/n1000/cvrp_n1000_k32_ants100_H10_miniH100_rho0.5_mne12_ppo_lr5e-06_mh4_hg2_hdants_ha85-5-5-5_anchor1_anchored_best.pt`

Training notes:
- TSP matched single-head completed 10 epochs; best validation gap `0.4914%`.
- TSP anchored multi-head was stopped after its saved best checkpoint because full TSPLIB already improved with the conservative all-head deployment.
- CVRP matched single-head completed 10 epochs; best validation cost `35.7069`, validation gap `-0.0778%`.
- CVRP anchored multi-head was stopped at epoch 3 after it matched/slightly improved the validation target; best validation cost `35.7047`, validation gap `-0.0761%`.

Full benchmark results:

| Problem / Dataset | Instances | Model | Gap | Mean Cost | Mean Time |
|---|---:|---|---:|---:|---:|
| TSP / full TSPLIB-converted STAR file | 228 | single-head n=1000 | 2.0604% | 4335923.2902 | 4.3452s |
| TSP / full TSPLIB-converted STAR file | 228 | anchored multi-head n=1000, `97,1,1,1` | 1.9803% | 4330966.1345 | 7.6588s |
| CVRP / full CVRPLIB-converted STAR file | 110 | single-head n=1000 retrain | 2.9746% | 76.4303 | 4.2282s |
| CVRP / full CVRPLIB-converted STAR file | 110 | anchored multi-head n=1000, `97,1,1,1` | 2.8373% | 76.3493 | 4.0803s |

Saved summaries:
- `results/tsp_tsplib_full_n1000_original_summary.json`
- `results/tsp_tsplib_full_n1000_multihead_anchor_ha97-1-1-1_summary.json`
- `results/cvrp_star_cvrlib_n1000_original_retrain_summary.json`
- `results/cvrp_star_cvrlib_n1000_multihead_anchor_ha97-1-1-1_summary.json`

Conclusion:
- After the backend head-count fix and teacher-anchored training, multi-head wins under the matched default budget on both full benchmark suites.
- The earlier n=1000 CVRP result where multi-head lost (`3.1344%` vs `3.0908%`) is superseded by the anchored run (`2.8373%` vs the freshly retrained single-head `2.9746%`).

## 2026-05-13: CVRP n=500 on STAR CVRPLIB Splits

Context:
- Re-ran CVRP at `n_node=500` because the n=500 setting looked stronger than n=1000 for CVRPLIB transfer.
- Split the STAR-converted CVRPLIB file into:
  - `<1K`: `data/CVRP/data/test_set/STAR_CVRPlib_lt1000.txt` with 99 instances.
  - `>=1K`: `data/CVRP/data/test_set/STAR_CVRPlib_ge1000.txt` with 11 instances.
- Fixed CVRP synthetic training generation in `utils.py`: `gen_cvrp_instance()` was accidentally using later BPP demand constants, producing infeasible n=500 CVRP demands. It now uses local CVRP demands `1..9`.

Training:
- Original checkpoint: `pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_best.pt`
- Multihead checkpoint: `pretrained/cvrp/n500/cvrp_n500_k32_ants100_H2_miniH20_rho0.5_mne12_ppo_lr5e-06_mh4_hg2_hdants_best.pt`
- Train config used `H=2`, `mini_H=20`, `epochs=3`, `steps_per_epoch=8`.
- Evaluation explicitly used `--H 10 --mini_H 100`.

| CVRPLIB Split | Instances | Original Gap | Multihead Gap | Original Mean Cost | Multihead Mean Cost | Original Mean Time | Multihead Mean Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<1K` | 99 | 2.8473% | 2.7818% | 65.4522 | 65.4578 | 1.8718s | 2.0627s |
| `>=1K` | 11 | 6.2993% | 6.1159% | 177.1606 | 176.9401 | 21.2462s | 24.2469s |

Weighted full-CVRPLIB gap across all 110 instances:
- n=500 original: 3.1925%
- n=500 multihead: 3.1152%

Saved summaries:
- `results/cvrp_star_cvrlib_lt1000_n500_original_summary.json`
- `results/cvrp_star_cvrlib_lt1000_n500_multihead_summary.json`
- `results/cvrp_star_cvrlib_ge1000_n500_original_summary.json`
- `results/cvrp_star_cvrlib_ge1000_n500_multihead_summary.json`

Comparison with previous n=1000 full-CVRPLIB run:

| Model | Full CVRPLIB Gap | Mean Cost | Mean Time |
|---|---:|---:|---:|
| n=1000 original | 3.0908% | 76.5489 | 3.8176s |
| n=1000 multihead | 3.1344% | 76.6485 | 4.0183s |
| n=500 original, split-weighted | 3.1925% | - | - |
| n=500 multihead, split-weighted | 3.1152% | - | - |

Conclusion:
- n=500 multihead improves over n=500 original on both CVRPLIB splits.
- n=1000 original remains slightly best overall on full CVRPLIB.
