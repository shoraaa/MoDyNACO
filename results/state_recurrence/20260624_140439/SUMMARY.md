# State Recurrence Experiment

- Problem: tsp
- Domains: uniform-1k, clustered-16k
- Domain specs: {"clustered-16k": {"clusters": 16, "kind": "clustered", "n": 16000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=50
- Representation: raw / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (14 train instances, 6 test instances)
- States: train=35840, test=15360
- Domain-classifier test accuracy: 0.5000
- Majority-class chance: 0.5000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9918, PC2=0.0082
- Permutation test: p=0.5800 (mean=0.4856, std=0.0933, trials=50)

## Controls
- Positive control (early-vs-late on dynamic features): accuracy=0.5995, chance=0.5000
- Geometry baseline (edge coordinates with domain labels): accuracy=0.9542, chance=0.5000
- UMAP skipped: No module named 'umap'
