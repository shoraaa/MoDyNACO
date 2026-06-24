# State Recurrence Experiment

- Problem: tsp
- Domains: uniform-1k, clustered-16k
- Domain specs: {"clustered-16k": {"clusters": 16, "kind": "clustered", "n": 16000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=50
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (14 train instances, 6 test instances)
- States: train=35840, test=15360
- Domain-classifier test accuracy: 0.5079 [0.5003, 0.5156]
- Majority-class chance: 0.5000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9919, PC2=0.0081
- Permutation test: p=0.0200 (mean=0.5053, std=0.0357, trials=50)

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.7539 [0.7428, 0.7651], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.9503 [0.9469, 0.9538], chance=0.5000
- UMAP skipped: No module named 'umap'
