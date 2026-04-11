# VRAM Usage: Dynamic (6 Edge Features) vs Static (1 Edge Feature)

Source: [output/vram_profile/vram_profile.csv](/home/shora/Research/old/NGFACO/output/vram_profile/vram_profile.csv)

## TSP Comparison Against Other Methods

This section compares our TSP runs against the provided POMO, SIGD, and LEHD baselines, plus measured original DeepACO TSP results. Our memory numbers use peak reserved VRAM from the source CSV. Baseline memory numbers were provided in MB and are converted here to GB for consistency.

| Method | 1K Memory (GB) | 5K Memory (GB) | 10K Memory (GB) | 50K Memory (GB) | 100K Memory (GB) |
|---|---:|---:|---:|---:|---:|
| Ours (Dynamic) | 0.08 | 0.18 | 0.35 | 1.57 | 3.11 |
| Ours (Static) | 0.08 | 0.18 | 0.31 | 1.54 | 3.04 |
| DeepACO (Original TSP) | 0.14 | 2.11 | 8.31 | OOM | OOM |
| POMO | 0.10 | 2.54 | 10.11 | OOM | OOM |
| SIGD | 0.04 | 0.95 | 3.76 | OOM | OOM |
| LEHD | 0.10 | 2.27 | 9.01 | OOM | OOM |

`DeepACO (Original TSP)` is measured from the fast peak-VRAM profiling run using peak reserved VRAM. It OOMs at `50K` and `100K` on a 16 GB GPU.

## Peak Reserved VRAM (GB)

| Problem | Scale | Dynamic | Static |
|---|---:|---:|---:|
| TSP | 1K | 0.076 | 0.076 |
| TSP | 5K | 0.178 | 0.178 |
| TSP | 10K | 0.346 | 0.307 |
| TSP | 50K | 1.572 | 1.535 |
| TSP | 100K | 3.111 | 3.041 |
| CVRP | 1K | 0.076 | 0.076 |
| CVRP | 5K | 0.178 | 0.178 |
| CVRP | 10K | 0.348 | 0.309 |
| CVRP | 50K | 1.566 | 1.529 |
| CVRP | 100K | 3.113 | 3.041 |

## Peak Allocated VRAM (GB)

| Problem | Scale | Dynamic | Static |
|---|---:|---:|---:|
| TSP | 1K | 0.058 | 0.057 |
| TSP | 5K | 0.162 | 0.159 |
| TSP | 10K | 0.293 | 0.286 |
| TSP | 50K | 1.312 | 1.282 |
| TSP | 100K | 2.584 | 2.524 |
| CVRP | 1K | 0.057 | 0.057 |
| CVRP | 5K | 0.162 | 0.158 |
| CVRP | 10K | 0.292 | 0.286 |
| CVRP | 50K | 1.306 | 1.276 |
| CVRP | 100K | 2.573 | 2.513 |
