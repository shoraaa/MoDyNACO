# Plan: Unify _extended.py Files into Main Files

## Context

The codebase currently has parallel implementations for two families of problems:
- **Base problems** (TSP, CVRP): `net.py`, `utils.py`, `faco.py`
- **Extended problems** (BPP, MKP, OP): `net_extended.py`, `utils_extended.py`, `faco_extended.py`

The extended versions use different graph topologies (dense vs sparse), feature dimensions, and constraint logic. The goal is to unify these into single files that support all problem types, reducing duplication and simplifying maintenance.

## Objectives

1. Merge `net_extended.py` into `net.py`
2. Merge `utils_extended.py` into `utils.py`
3. Merge `faco_extended.py` into `faco.py`
4. Ensure all existing tests pass
5. Maintain backward compatibility for training/testing scripts

## Detailed Implementation Plan

### Phase 1: Network Unification (`net.py`)

**Strategy**: Parameterize the existing `Net` class to support all problem types, while keeping specialized named classes (`NetBPP`, `NetMKP`, `NetOP`) as thin wrappers for backward compatibility.

**Changes**:

1. **Enhance `Net` class**:
   - Add `problem_type` parameter (default=None maintains current behavior)
   - Auto-configure `feats` and `edge_feats` based on problem type if not provided:
     - `'tsp'`: feats=2, edge_feats=6 (with incumbents) or 3 (static)
     - `'cvrp'`: feats=4 (coords+demand+depot), edge_feats=6 or 3
     - `'bpp'`: feats=1, edge_feats=2
     - `'mkp'`: feats=m+1 (need m parameter), edge_feats=2
     - `'op'`: feats=2, edge_feats=2
   - Move checkpointing logic to `Net` base (use `grad_checkpointing` flag)

2. **Create specialized wrappers** (for backward compatibility):
   ```python
   class NetBPP(Net):
       def __init__(self, feats=1, edge_feats=2, **kwargs):
           super().__init__(feats=feats, edge_feats=edge_feats, problem_type='bpp', **kwargs)

   class NetMKP(Net):
       def __init__(self, m=5, feats=None, edge_feats=2, **kwargs):
           if feats is None:
               feats = m + 1
           super().__init__(feats=feats, edge_feats=edge_feats, problem_type='mkp', m=m, **kwargs)

   class NetOP(Net):
       def __init__(self, feats=2, edge_feats=2, **kwargs):
           super().__init__(feats=feats, edge_feats=edge_feats, problem_type='op', **kwargs)
   ```

3. **Unify checkpointing**: 
   - Replace separate `EmbNetCheckpoint*` classes with a single `EmbNet` that uses `grad_checkpointing` flag (like base)
   - This removes duplication of checkpoint logic

4. **Keep `get_network()` factory**: Extend to handle all problem types

**Critical Files**: `net.py` (modified), tests reference `NetBPP` etc. so wrappers needed.

### Phase 2: Utils Unification (`utils.py`)

**Strategy**: Consolidate all data generation and graph building functions into a single module with problem dispatch.

**Changes**:

1. **Move extended functions** from `utils_extended.py` to `utils.py`:
   - `gen_bpp_instance()`, `gen_mkp_instance()`, `gen_op_instance()`
   - `build_pyg_data_bpp()`, `build_pyg_data_mkp()`, `build_pyg_data_op()`, `build_pyg_data_op_dense()`
   - `load_bpp_test_dataset()`, `load_mkp_test_dataset()`, `load_op_test_dataset()`
   - Helper: `gen_op_distance_matrix()`, `gen_op_prizes()`

2. **Create unified factory functions**:
   ```python
   def gen_instance(problem: str, n: int, device, **kwargs):
       if problem == 'tsp':
           return generate_tsp_instance(n)
       elif problem == 'cvrp':
           return gen_cvrp_instance(n, device, **kwargs)
       elif problem == 'bpp':
           return gen_bpp_instance(n, device)
       elif problem == 'mkp':
           m = kwargs.get('m', 5)
           return gen_mkp_instance(n, m, device)
       elif problem == 'op':
           return gen_op_instance(n, device)
   
   def build_pyg_data(problem: str, aco, *data_args, device='cpu', dynamic=True, **kwargs):
       if problem == 'tsp':
           coords = data_args[0]
           return build_pyg_data_tsp(aco, coords, device, dynamic=dynamic, **kwargs)
       elif problem == 'cvrp':
           coords, demand, capacity = data_args
           return build_pyg_data_cvrp(aco, coords, demand, capacity, device, dynamic=dynamic, **kwargs)
       elif problem == 'bpp':
           demand = data_args[0]
           return build_pyg_data_bpp(demand, aco, device, dynamic=dynamic, **kwargs)
       # ... etc
   ```

3. **Keep existing functions** for backward compatibility (TSP/CVRP keep their names, extended ones keep theirs)

4. **Consolidate dataset loading**: Keep separate loaders but unify under `load_test_dataset()` with problem parameter

**Note**: The `infer_instance()` function in `utils.py` already handles dispatch via `problem` parameter. We'll need to update it to recognize extended problems and call appropriate builders.

### Phase 3: ACO Solver Unification (`faco.py`)

**Strategy**: Create a common base class infrastructure, then refactor extended solvers to inherit from it while preserving problem-specific logic.

**Changes**:

1. **Refactor `_BaseMFACO`** to be truly base class:
   - Move common `sample()` logic (prior handling, torch sync) to base
   - Define abstract methods: `_sample_impl()`, `_update_pheromone_impl()`, `_sync_cpp_inputs()`, `_sync_from_cpp()`
   - Add `problem_type` attribute
   - Standardize return signature: `(costs, flats, touched, logps, traces, costs_raw, flats_raw, extra, survival)`
     - For extended problems without `touched`, return empty tensor or None

2. **Refactor extended solvers to inherit**:
   ```python
   class MFACO_BPP(_BaseMFACO):
       def __init__(self, ...):
           # Custom init
           self.problem_type = 'bpp'
           
       def _sample_impl(self, prior_arg, require_prob):
           # Call C++ and return costs, paths, logps, trace
           ...
       
       def _update_pheromone_impl(self, best_flat, best_cost):
           # Problem-specific update
           ...
   ```
   - Similarly for `MFACO_MKP`, `MFACO_OP`

3. **Standardize attributes**:
   - All should have: `n`, `n_ants`, `k`, `pheromone_sparse`, `h_sparse_torch`, `nn_torch` (or provide compatible properties)
   - For dense problems (BPP/MKP/OP), `k` equals `n+1` (full connectivity)
   - Provide `nn_torch` property that returns all-to-all connectivity mask

4. **Unify torch sync infrastructure**:
   - Add `enable_torch_sync` parameter to all extended classes
   - Use base class `_init_torch_buffers()` logic but adapt for dense vs sparse

5. **Handle return signature differences**:
   - Base (TSP/CVRP) returns 9-tuple
   - Extended currently returns 4-tuple
   - Extended need to compute/return additional fields to match signature
     - `touched`: number of edges explored (can be None or computed)
     - `costs_raw`, `flats_raw`: before local search (extended doesn't have LS, so same as final)
     - `extra` (survival): currently None, but extended BPP might compute something

6. **Keep factory function `get_aco()`** (from `faco_extended`) and merge with `faco.py`:
   - Could have unified `get_aco(problem, **kwargs)` that dispatches to appropriate class
   - Or keep separate but export all classes from single module

7. **Preserve standalone ACO classes** (`ACO_TSP`, `ACO_CVRP`) unchanged (no unification needed)

### Phase 4: Update Extended Common

The `extended_common.py` file serves as an adapter. **Minimal changes needed**:

1. Update imports to use unified modules:
   ```python
   from net import NetBPP, NetMKP, NetOP  # wrappers still exist
   from faco import MFACO_BPP, MFACO_MKP, MFACO_OP
   from utils import build_pyg_data_bpp, build_pyg_data_mkp, build_pyg_data_op_dense, ...
   ```

2. Potentially rename functions to match unified naming (or keep as-is for compatibility)

### Phase 5: Update Train/Test Scripts

**Minimal changes**: The imports already use:
- `import net`, `import faco`, `import utils`
- `import utils_extended` (will be removed)
- `from net import Net` (still works with enhanced Net)
- `from extended_common import ...` (still works)

**Actions**:
1. Remove `import utils_extended` from `train.py` and `test.py` - use `utils` directly
2. Remove `from net_extended import NetBPP, NetMKP, NetOP` - use `from net import NetBPP, NetMKP, NetOP`
3. Update any direct references to `faco_extended` to use `faco` instead
4. Ensure `extended_common.py` imports from unified modules

### Phase 6: Testing & Verification

**Test Strategy**:

1. **Run existing tests** to ensure backward compatibility:
   - `tests/test_extended_common.py` - tests extended_common functions
   - `tests/test_extended_regressions.py` - tests BPP/MKP/OP specific behaviors
   - `tests/test_unified_entrypoints.py` - tests train/test dispatch logic
   - `tests/test_extended_common.py` relies on `utils_extended` and `net_extended` - after unification they'll import from `utils` and `net`

2. **Add smoke tests** for unified imports:
   - Verify `from net import NetBPP, NetMKP, NetOP` works
   - Verify `from utils import gen_bpp_instance, build_pyg_data_bpp, ...` works
   - Verify `from faco import MFACO_BPP, MFACO_MKP, MFACO_OP` works

3. **Quick sanity runs**:
   - Train a tiny BPP model (1 epoch, n=8) to verify training pipeline
   - Test inference on MKP/OP to verify ACO integration
   - Run TSP/CVRP baseline tests to ensure no regression

**Expected Outcome**: All tests pass, and both base and extended problems work from unified modules.

## Implementation Order (Recommended)

To minimize risk, perform unification in this order:

1. **Start with `net.py`**: 
   - Add `problem_type` parameter to `Net`
   - Create `NetBPP`, `NetMKP`, `NetOP` wrappers
   - Update `extended_common.py` to import from `net` instead of `net_extended`
   - Run tests to verify wrapper functionality
   - Once verified, can remove `net_extended.py` (or keep as shim that imports from net)

2. **Then `utils.py`**:
   - Add extended functions to `utils.py`
   - Create unified factory functions (optional)
   - Update `extended_common.py` to import from `utils`
   - Update `infer_instance()` in `utils.py` to dispatch to extended builders
   - Run tests
   - Remove `utils_extended.py` shim

3. **Finally `faco.py`** (most complex):
   - Refactor `_BaseMFACO` to support both sparse and dense patterns
   - Make extended classes inherit from `_BaseMFACO`
   - Standardize return signatures
   - Update `extended_common.py` to import from `faco`
   - Run tests
   - Remove `faco_extended.py` shim

4. **Cleanup**:
   - Remove `utils_extended.py`, `net_extended.py`, `faco_extended.py`
   - Or leave minimal shims that re-export from unified modules (easiest for backward compatibility)

## Backward Compatibility Strategy

To avoid breaking existing code that might import `net_extended` directly:

**Option A (Shim)**: Create `net_extended.py` that re-exports from `net`:
```python
from net import NetBPP, NetMKP, NetOP, Net, GNNLayer, EmbNet, ParNet, move_pyg_to_module_device
__all__ = ['NetBPP', 'NetMKP', 'NetOP', 'Net', ...]
```

**Option B (Remove)**: Delete the files and update all internal imports. External users must update.

**Recommendation**: Use shims initially, then deprecate with warnings. This gives users time to migrate.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing checkpoints | High | Keep class names/`__init__` signatures compatible; test loading old checkpoints |
| API mismatch in ACO return signatures | High | Provide adapter layer or make extended classes match base signature exactly |
| Performance regression (dense vs sparse) | Medium | Preserve graph structure; don't force sparse for extended problems |
| Test failures due to subtle behavior changes | Medium | Run full test suite after each phase; fix discrepancies |
| Checkpoint loading failures (state dict keys) | Medium | Maintain `_load_compat_hook` in EmbNet for old checkpoints |

## Verification Checklist

After implementation:

- [ ] All existing unit tests pass (`pytest tests/`)
- [ ] Train script works for all problem types: `uv run train.py --problem bpp --n_node 12 ...`
- [ ] Test script works for all problem types: `uv run test.py --problem mkp --checkpoint ...`
- [ ] Import unification works: `from net import NetBPP` succeeds
- [ ] No references to `net_extended`, `utils_extended`, `faco_extended` remain in train/test (except shims)
- [ ] Old checkpoints can still be loaded (verify with saved models if available)
- [ ] Code quality: no new lint errors, type hints consistent

## Files to Modify

**Core changes**:
- `net.py` - major additions
- `utils.py` - major additions
- `faco.py` - major refactor
- `extended_common.py` - update imports

**Cleanup/Shims**:
- Create `net_extended.py` shim (re-export from net) OR delete
- Create `utils_extended.py` shim OR delete
- Create `faco_extended.py` shim OR delete

**No changes** (should work as-is):
- `train.py` (only if extended_common updated correctly)
- `test.py` (only if extended_common updated correctly)
- `test_extended_common.py`
- `test_extended_regressions.py`

## Notes

- The unification should preserve the current separation of concerns: TSP/CVRP (sparse) vs BPP/MKP/OP (dense)
- Don't force a single graph builder; keep `build_pyg_data_tsp` and `build_pyg_data_bpp` separate but in same module
- The `extended_common.py` adapter pattern has worked well; keep it but make it import from unified modules
- Consider whether we want a single `MFACO` class with mode flag or keep separate classes. Separate classes with common base is cleaner.

## Success Criteria

1. No code outside the `_extended.py` files directly imports from them (all imports go through unified modules or shims)
2. All tests pass without modification (maintaining backward compatibility)
3. Code duplication eliminated: no copy-paste logic between files
4. Single source of truth for each component (network, utils, ACO)
5. Easy to add new problem types in future by following the pattern
