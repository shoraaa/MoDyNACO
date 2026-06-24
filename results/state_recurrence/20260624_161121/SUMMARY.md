# State Recurrence Experiment

- Problem: tsp
- Domains: uniform-1k, clustered-10k, clustered-100k
- Domain specs: {"clustered-100k": {"clusters": 100, "kind": "clustered", "n": 100000}, "clustered-10k": {"clusters": 10, "kind": "clustered", "n": 10000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=50
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (21 train instances, 9 test instances)
- States: train=53760, test=23040
- Domain-classifier test accuracy: 0.4003 [0.3942, 0.4067]
- Majority-class chance: 0.3333
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9918, PC2=0.0082
- Permutation test: p=0.0000 (mean=0.3342, std=0.0048, trials=50)

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.7500 [0.7409, 0.7581], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.7998 [0.7947, 0.8050], chance=0.3333
- UMAP skipped: No module named 'umap'
