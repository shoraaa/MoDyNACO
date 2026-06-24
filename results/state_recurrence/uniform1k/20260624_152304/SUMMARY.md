# State Recurrence Experiment

- Problem: cvrp
- Domains: uniform-1k
- Domain specs: {"uniform-1k": {"kind": "uniform", "n": 1000}}
- Search schedule: H=10, mini_H=30
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (7 train instances, 3 test instances)
- States: train=17920, test=7680
- Domain-classifier test accuracy: 1.0000 [1.0000, 1.0000]
- Majority-class chance: 1.0000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9918, PC2=0.0082

## Behavior classification (head ID)
- Behavior classifier: accuracy=0.8667 [0.8250, 0.9125], chance=0.1250
- Head residual cosine similarity (off-diagonal mean): -0.1278
- Head top-5% edge overlap (off-diagonal mean): 0.9888
- Phase-behavior coupling (Cramér's V): 0.0579 (chi² p=0.02362)
- Mean win-step per head: 4.46, 4.62, 4.61, 4.45, 4.62, 4.26, 4.41, 4.54
- UMAP skipped: No module named 'umap'
