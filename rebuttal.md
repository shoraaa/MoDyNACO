We thanks the reviewer.
# For Q1:
We present the additional result regarding the impact of perturbation size M and the candidate size K in the table below.

| K   | TSP-1K Obj. | Time (s) | TSP-5K Obj. | Time (s) | CVRP-1K Obj. | Time (s) | CVRP-5K Obj. | Time (s) |
| :-- | ----------: | -------: | ----------: | -------: | -----------: | -------: | -----------: | -------: |
| 16  | 23.2374     | 0.2587   | 51.6764     | 0.6005   | 36.9135      | 0.7927   | 99.0747      | 1.9070   |
| 24  | 23.2315     | 0.2826   | 51.6415     | 0.6941   | 36.7246      | 1.0260   | 96.9905      | 2.5810   |
| 32  | 23.2322     | 0.3155   | 51.6321     | 0.7595   | 36.2780      | 1.2693   | 96.8090      | 2.7276   |
| 48  | 23.2090     | 0.3550   | 51.4889     | 0.9827   | 36.0597      | 1.6790   | 95.8279      | 3.5052   |
| 64  | **23.1814** | 0.4579   | **51.4719** | 1.1724   | **35.9841**  | 2.1999   | **95.0787**  | 4.4208   |

| M   | TSP-1K Obj. | Time (s) | TSP-5K Obj. | Time (s) | CVRP-1K Obj. | Time (s) | CVRP-5K Obj. | Time (s) |
| --- | ----------: | -------: | ----------: | -------: | -----------: | -------: | -----------: | -------: |
| 4   | 23.2579     | 0.2678   | 51.6470     | 0.6867   | 36.2187      | 1.1129   | 96.2789      | 2.5090   |
| 8   | **23.2277** | 0.2880   | **51.6239** | 0.7346   | 36.4345      | **1.0466** | 97.1280    | **2.3748** |
| 12  | 23.2322     | 0.3155   | 51.6321     | 0.7595   | 36.2780      | 1.2693   | 96.8090      | 2.7276   |
| 16  | 23.2563     | 0.3339   | 51.7718     | 0.8226   | **36.2283**  | 1.4180   | **96.4833**  | 3.2432   |
| 20  | 23.3619     | 0.3693   | 52.0491     | 0.8552   | 36.2712      | 1.5147   | 98.2721      | 2.9165   |

# For Q2
The challenges that would arise in multi-scale training for DyNACO mainly come from the problem-specific difference in optimal/good solution distribution. While our methods already showcase significant robustness across scales (using 1K node trained model to test on TSPLIB/CVRPLIB real world dataset), we observed that TSP generalization seems better than CVRP. In fact, model trained on 1K instances even get better results when tested on 10K instances than their in-scale counterpart, while CVRP generalization is alway worse. One possible explanation is through pheromone correlation of each scale, observed in Figure A1 and A3: At difference scales, TSP model consistently show the same pattern or even values across steps, while CVRP landscape look vastly different.  This make sense: different TSP scales have the same local "shape" on how it could be good, but CVRP is more complex in that regard (more sub-route, etc.). So while we think multi-scale is very possible for DyNACO (especially in COPs that have intuitively scale-invariant local structure like TSP), COPs with vastly difference structures across scales would need to be investigate more on how to anchor them effectively. 

# For Q3
The main contribution come from the difference is optimal/good solution distribution across difference scales of CVRP, more specifically capacity constraint input which lead to difference structures (e.g. number of sub-routes). So to answer your questions, it essentially both. This polarity reversal reveal to us how the solver need difference guidance across scale: Larger scale need *exploitation* (emphasize existing pheromone, less noisy ant decision), while smaller scale require *exploration* (counteract local optimal status of pheromones). 


# For W1

|            | ACO w/o pheromone stablization | ACO             | DyNACO w/o pheromone stablization | ours              |
| ---------- | ------------------------------ | --------------- | --------------------------------- | ----------------- |
| CVRP-1K    | 36.63 (4.09%)                  | 36.22 (2.94%)   | 36.42 (3.51%)                     | **34.83 (2.18%)** |
| TSPLIB(16) | 1323473 (0.99%)                | 1325466 (1.31%) | **1321956 (0.92%)**               | 1322398 (1.10%)   |
| CVRPLIB(5) | 48.49 (2.911%)                 | 48.36 (2.76%)   | 48.41 (2.71%)                     | **48.27 (2.18%)** |
