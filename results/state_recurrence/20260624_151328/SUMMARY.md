# State Recurrence Experiment

- Problem: cvrp
- Domains: uniform-1k, clustered-1k
- Domain specs: {"clustered-1k": {"clusters": 8, "kind": "clustered", "n": 1000}, "uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=30
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (14 train instances, 6 test instances)
- States: train=35840, test=15360
- Domain-classifier test accuracy: 0.5074 [0.4998, 0.5151]
- Majority-class chance: 0.5000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9916, PC2=0.0084

## Controls
All controls use the same instances, search states, and sampled edges; only features or labels differ.
- Positive control (extremal search phase, dynamic features): accuracy=0.7479 [0.7370, 0.7590], chance=0.5000
- Geometry baseline (edge-endpoint coordinates, domain labels): accuracy=0.8132 [0.8068, 0.8193], chance=0.5000

## Behavior classification (head ID)
- Behavior classifier: accuracy=0.8042 [0.7667, 0.8376], chance=0.1250
- Phase-behavior coupling (Cramér's V): 0.0416 (chi² p=0.007588)
- Mean win-step per head: 4.50, 4.60, 4.58, 4.59, 4.57, 4.33, 4.30, 4.50
- UMAP skipped: No module named 'umap'
