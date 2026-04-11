# Analysis: DeepACO vs NGFACO for BPP

## Key Differences Found

### 1. **Training Objective (CRITICAL)**

**DeepACO (Original):**
```python
# train_instance function
costs, log_probs = aco.sample()
baseline = costs.mean()
reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
```

**NGFACO (Current):**
```python
# train_instance_reinforce function
baseline = costs_t.mean()
adv = (min_cost - costs_t).detach()  # Using min_cost instead of mean
loss = (logp_new * adv).mean()
```

**Issue:** 
- DeepACO uses `(costs - baseline)` where costs are **negative fitness values**
- NGFACO uses `(min_cost - costs_t)` which is the **opposite sign**
- DeepACO's loss: `torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants`
- NGFACO's loss: `(logp_new * adv).mean()`

**Fix:** Use the same sign as DeepACO:
```python
# Should be:
adv = (costs_t - baseline).detach()  # costs - baseline, NOT baseline - costs
loss = -(logp_new * adv).mean()  # Negative for minimization
```

### 2. **Cost Calculation**

**DeepACO (Original):**
```python
# aco.py - gen_path_costs
def gen_path_costs(self, paths:torch.Tensor):
    u = paths.permute(1, 0).numpy()
    last_zeros = count_last_zero(u)
    n_bins = u.shape[1] - last_zeros - self.problem_size + 1
    fit = cal_fitness(u, self.demand_numpy, n_bins)
    return -torch.tensor(fit)  # Returns NEGATIVE fitness
```

**NGFACO (Current):**
```python
# faco_extended.py - sample method
# Uses C++ costs directly (negative fitness values)
costs_np, trace = self._solver.sample_trace()
costs = torch.as_tensor(costs_np, device=self.device, dtype=torch.float32)
```

**Status:** ✅ **CORRECT** - Both use negative fitness values

### 3. **Heuristic Initialization**

**DeepACO (Original):**
```python
# aco.py - __init__
self.heuristic = self.demand.unsqueeze(0).repeat(len(demand), 1) if heuristic is None else heuristic
self.heuristic[:, 0] = 1e-5
```

**NGFACO (Current):**
```python
# faco_extended.py - __init__
demand_normalized = demand / demand.max()
heuristic_raw = 1.0 / (demand_normalized + 0.1)
heuristic_raw = heuristic_raw / heuristic_raw.max()
self.heuristic = heuristic_raw.unsqueeze(0).repeat(self._n + 1, 1).to(device)
self.heuristic[:, 0] = 0.1
```

**Issue:** 
- DeepACO uses **demand values directly** as heuristic
- NGFACO uses **inverse of normalized demand**
- These are fundamentally different!

**Fix:** Use demand values directly like DeepACO:
```python
self.heuristic = demand.unsqueeze(0).repeat(self._n + 1, 1).to(device)
self.heuristic[:, 0] = 1e-5
```

### 4. **Pheromone Initialization**

**DeepACO (Original):**
```python
# aco.py - __init__
self.pheromone = torch.ones(self.problem_size, self.problem_size) if pheromone is None else pheromone
```

**NGFACO (Current):**
```python
# faco_extended.py - __init__
self.pheromone = torch.full((self._n + 1, self._n + 1), 10.0, device=device, dtype=torch.float32)
```

**Issue:** 
- DeepACO uses **1.0** as initial pheromone
- NGFACO uses **10.0** (changed based on testing)

**Fix:** Use 1.0 like DeepACO

### 5. **ACO Parameters**

**DeepACO (Original):**
```python
# Default parameters
decay=0.9,  # High decay rate
alpha=1,
beta=1,
elitist=False,  # Non-elitist by default
n_ants=20
```

**NGFACO (Current):**
```python
# Default parameters
decay=0.01,  # Very low decay rate
alpha=1.0,
beta=1.0,
elitist=True,  # Elitist by default
n_ants=20
```

**Issue:** 
- DeepACO uses **decay=0.9** (high decay)
- NGFACO uses **decay=0.01** (very low decay)
- DeepACO uses **non-elitist** by default
- NGFACO uses **elitist** by default

**Fix:** Use decay=0.9 and elitist=False like DeepACO

### 6. **Neural Network Architecture**

**DeepACO (Original):**
```python
# net.py
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=1, units=32, act_fn='silu', agg_fn='mean'):
        # GNN with 12 layers, 32 units, silu activation
        ...

class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        # MLP with 3 layers, 32 units, silu activation
        ...
```

**NGFACO (Current):**
```python
# net_extended.py
class NetBPP(nn.Module):
    def __init__(self, feats=1, edge_feats=3, units=32, depth=12):
        # Similar architecture but with edge_feats=3 (for dynamic features)
        ...
```

**Status:** ✅ **SIMILAR** - Architecture is similar

### 7. **Training Parameters**

**DeepACO (Original):**
```python
# train.ipynb
lr = 3e-4,  # Learning rate
T = 5,  # ACO iterations for validation
n_ants = 20,
steps_per_epoch = ?,  # Not specified in code
epochs = 5,  # Only 5 epochs for BPP120
n_val = 100,  # Validation size
```

**NGFACO (Current):**
```python
# train_extended.py
lr = 5e-4,  # Learning rate
T = 5,
n_ants = 20,
steps_per_epoch = 8,
epochs = 10,
val_size = 16,
```

**Issue:** 
- DeepACO uses **lr=3e-4**
- NGFACO uses **lr=5e-4**
- DeepACO uses **n_val=100** for validation
- NGFACO uses **val_size=16**

**Fix:** Use lr=3e-4 and val_size=100 like DeepACO

### 8. **Optimizer**

**DeepACO (Original):**
```python
# train.ipynb
optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
```

**NGFACO (Current):**
```python
# train_extended.py
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
```

**Issue:** 
- DeepACO uses **AdamW**
- NGFACO uses **Adam**

**Fix:** Use AdamW like DeepACO

### 9. **Loss Calculation**

**DeepACO (Original):**
```python
# train_instance function
costs, log_probs = aco.sample()
baseline = costs.mean()
reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
```

**NGFACO (Current):**
```python
# train_instance_reinforce function
logp_new = logp_steps_new.sum(dim=0)
ndec_f = ndec_new.to(torch.float32).clamp_min(1.0)
logp_new = logp_new / ndec_f  # Normalized by number of decisions
min_cost = costs_t.min()
adv = (min_cost - costs_t).detach()
loss = (logp_new * adv).mean()
```

**Issue:** 
- DeepACO does **NOT** normalize log probabilities by number of decisions
- DeepACO uses **mean cost** as baseline, not **min cost**
- DeepACO uses **(costs - baseline)**, not **(min_cost - costs)**

**Fix:** Use the same loss calculation as DeepACO

### 10. **PyG Data Construction**

**DeepACO (Original):**
```python
# utils.py - gen_pyg_data
def gen_pyg_data(demands, device='cpu'):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = torch.ones((edge_index.size(1), 1))  # All ones
    x = demands
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data
```

**NGFACO (Current):**
```python
# utils_extended.py - build_pyg_data_bpp
def build_pyg_data_bpp(demand, device='cpu', pheromone=None, dynamic=True):
    demand = demand.to(device=device, dtype=torch.float32)
    n = demand.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = torch.ones((edge_index.size(1), 1), device=device)
    edge_attr = _augment_edge_attr_with_pheromone(edge_attr, pheromone, dynamic, device)
    x = demand.unsqueeze(1)
    pyg_data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data
```

**Issue:** 
- DeepACO uses **edge_attr = torch.ones((edge_index.size(1), 1))** (all ones)
- NGFACO uses **edge_attr with pheromone augmentation** (dynamic features)

**Fix:** Use all ones for edge_attr like DeepACO

## Summary of Critical Issues

### **Most Critical Issues (Must Fix):**

1. **Wrong loss sign**: NGFACO uses `(min_cost - costs)` but should use `(costs - baseline)`
2. **Wrong heuristic**: NGFACO uses inverse demand but should use demand directly
3. **Wrong pheromone initialization**: NGFACO uses 10.0 but should use 1.0
4. **Wrong ACO parameters**: NGFACO uses decay=0.01 but should use decay=0.9
5. **Wrong optimizer**: NGFACO uses Adam but should use AdamW
6. **Wrong edge features**: NGFACO uses dynamic pheromone features but should use all ones

### **Less Critical Issues (Should Fix):**

7. **Different learning rate**: NGFACO uses 5e-4 but should use 3e-4
8. **Different validation size**: NGFACO uses 16 but should use 100
9. **Different elitist setting**: NGFACO uses elitist=True but should use elitist=False
10. **Log probability normalization**: NGFACO normalizes by ndec but DeepACO doesn't

## Recommended Fixes

### Priority 1: Fix Loss Calculation
```python
# Change from:
min_cost = costs_t.min()
adv = (min_cost - costs_t).detach()
loss = (logp_new * adv).mean()

# To:
baseline = costs_t.mean()
adv = (costs_t - baseline).detach()  # costs - baseline
loss = -torch.sum(adv * logp_new) / aco.n_ants  # Negative for minimization
```

### Priority 2: Fix Heuristic
```python
# Change from:
demand_normalized = demand / demand.max()
heuristic_raw = 1.0 / (demand_normalized + 0.1)
heuristic_raw = heuristic_raw / heuristic_raw.max()
self.heuristic = heuristic_raw.unsqueeze(0).repeat(self._n + 1, 1).to(device)
self.heuristic[:, 0] = 0.1

# To:
self.heuristic = demand.unsqueeze(0).repeat(self._n + 1, 1).to(device)
self.heuristic[:, 0] = 1e-5
```

### Priority 3: Fix Pheromone and ACO Parameters
```python
# Change from:
self.pheromone = torch.full((self._n + 1, self._n + 1), 10.0, device=device, dtype=torch.float32)
decay=0.01
elitist=True

# To:
self.pheromone = torch.ones((self._n + 1, self._n + 1), device=device, dtype=torch.float32)
decay=0.9
elitist=False
```

### Priority 4: Fix Optimizer and Learning Rate
```python
# Change from:
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr = 5e-4

# To:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
lr = 3e-4
```

### Priority 5: Fix Edge Features
```python
# Change from:
edge_attr = _augment_edge_attr_with_pheromone(edge_attr, pheromone, dynamic, device)

# To:
edge_attr = torch.ones((edge_index.size(1), 1), device=device)
```

## Expected Results

After applying these fixes, the training should:
1. Show a clear downward trend in validation costs
2. Achieve better performance than baseline ACO
3. Match the performance of the original DeepACO implementation
