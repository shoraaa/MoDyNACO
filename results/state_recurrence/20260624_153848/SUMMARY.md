# State Recurrence Experiment

- Problem: cvrp
- Domains: uniform-1k, clustered-16k
- Domain specs: {"clustered-16k": {"clusters": 16, "kind": "clustered", "n": 16000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=50
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (14 train instances, 6 test instances)
- States: train=35840, test=15360
- Domain-classifier test accuracy: 0.5087 [0.5010, 0.5168]
- Majority-class chance: 0.5000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9918, PC2=0.0082
- Permutation test: p=0.1200 (mean=0.5033, std=0.0177, trials=50)

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.7476 [0.7365, 0.7586], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.9612 [0.9580, 0.9643], chance=0.5000

## Behavior classification (head ID)
- Behavior classifier: accuracy=0.7917 [0.7562, 0.8271], chance=0.1250
- Head residual cosine similarity (off-diagonal mean): -0.1270
- Head top-5% edge overlap (off-diagonal mean): 0.9856
- Phase-behavior coupling (Cramér's V): 0.0289 (chi² p=1)
- Mean win-step per head: 4.53, 4.48, 4.45, 4.66, 4.41, 4.49, 4.48, 4.50
- UMAP skipped: No module named 'umap'
