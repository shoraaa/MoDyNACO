# State Recurrence Experiment

- Problem: cvrp
- Domains: clustered-16k
- Domain specs: {"clustered-16k": {"clusters": 16, "kind": "clustered", "n": 16000}}
- Search schedule: H=10, mini_H=30
- Representation: embedding / dynamic / edge
- Classifier: mlp
- Split: held_out_instances (7 train instances, 3 test instances)
- States: train=17920, test=7680
- Domain-classifier test accuracy: 1.0000 [1.0000, 1.0000]
- Majority-class chance: 1.0000
- Interpretation: classifier is near chance; overlap is stronger when accuracy stays near chance.
- PCA explained variance: PC1=0.9917, PC2=0.0083

## Behavior classification (head ID)
- Behavior classifier: accuracy=0.8875 [0.8458, 0.9250], chance=0.1250
- Head residual cosine similarity (off-diagonal mean): -0.1311
- Head top-5% edge overlap (off-diagonal mean): 0.9820
- Phase-behavior coupling (Cramér's V): 0.0534 (chi² p=1)
- Mean win-step per head: 4.36, 4.80, 4.65, 4.46, 4.48, 4.43, 4.53, 4.31
- UMAP skipped: No module named 'umap'
