# State Recurrence Experiment

- Problem: cvrp
- Domains: uniform-1k, clustered-16k
- Domain specs: {"clustered-16k": {"clusters": 16, "kind": "clustered", "n": 16000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=50
- Representation: raw / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (14 train instances, 6 test instances)
- States: train=35840, test=15360
- Domain-classifier test accuracy: 0.5108 [0.5033, 0.5189]
- Majority-class chance: 0.5000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9918, PC2=0.0082
- Permutation test: p=0.0400 (mean=0.4998, std=0.0065, trials=50)

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.7493 [0.7384, 0.7604], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.9805 [0.9783, 0.9826], chance=0.5000
- UMAP skipped: No module named 'umap'
