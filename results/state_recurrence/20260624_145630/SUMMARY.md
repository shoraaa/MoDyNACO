# State Recurrence Experiment

- Problem: cvrp
- Domains: uniform-1k, clustered-1k
- Domain specs: {"clustered-1k": {"clusters": 8, "kind": "clustered", "n": 1000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=5, mini_H=20
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (6 train instances, 4 test instances)
- States: train=7680, test=5120
- Domain-classifier test accuracy: 0.5051 [0.4912, 0.5191]
- Majority-class chance: 0.5000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9887, PC2=0.0113

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.9948 [0.9909, 0.9980], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.5461 [0.5320, 0.5596], chance=0.5000

## Behavior classification (head ID)
- Behavior classifier: accuracy=0.5417 [0.4583, 0.6333], chance=0.1250
- UMAP skipped: No module named 'umap'
