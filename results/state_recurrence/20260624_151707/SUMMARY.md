# State Recurrence Experiment

- Problem: cvrp
- Domains: uniform-1k, clustered-1k
- Domain specs: {"clustered-1k": {"clusters": 8, "kind": "clustered", "n": 1000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=30
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (14 train instances, 6 test instances)
- States: train=35840, test=15360
- Domain-classifier test accuracy: 0.5072 [0.4996, 0.5150]
- Majority-class chance: 0.5000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9916, PC2=0.0084

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.7352 [0.7238, 0.7469], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.5000 [0.4923, 0.5081], chance=0.5000

## Behavior classification (head ID)
- Behavior classifier: accuracy=0.8000 [0.7646, 0.8375], chance=0.1250
- Head pairwise cosine similarity (off-diagonal mean): 0.9982
- Head top-5% edge overlap (off-diagonal mean): 0.9876
- Phase-behavior coupling (Cramér's V): 0.0427 (chi² p=0.001068)
- Mean win-step per head: 4.50, 4.55, 4.59, 4.57, 4.58, 4.35, 4.35, 4.50
- UMAP skipped: No module named 'umap'
