# State Recurrence Experiment

- Problem: cvrp
- Domains: uniform-1k, clustered-10k, clustered-100k
- Domain specs: {"clustered-100k": {"clusters": 100, "kind": "clustered", "n": 100000}, "clustered-10k": {"clusters": 10, "kind": "clustered", "n": 10000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=50
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (21 train instances, 9 test instances)
- States: train=53760, test=23040
- Domain-classifier test accuracy: 0.3341 [0.3281, 0.3400]
- Majority-class chance: 0.3333
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9918, PC2=0.0082
- Permutation test: p=0.1600 (mean=0.3272, std=0.0286, trials=50)

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.7497 [0.7405, 0.7580], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.9015 [0.8977, 0.9054], chance=0.3333
- UMAP skipped: No module named 'umap'
