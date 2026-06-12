"""
MFACO Solver - Unified interface for TSP and CVRP with C++ backend.
ACO solvers for BPP, MKP, OP also available.

Usage:
    from faco import MFACO_TSP, MFACO_CVRP, ACO_BPP, ACO_MKP, ACO_OP

    # TSP
    solver = MFACO_TSP(coords, n_ants=20, ...)
    costs, flats, ... = solver.sample()

    # CVRP
    solver = MFACO_CVRP(coords, demand, capacity, n_ants=20, ...)
    costs, perms, ... = solver.sample()

    # BPP
    solver = ACO_BPP(demand, n_ants=20, ...)
    costs, paths, logps, trace = solver.sample()

    # MKP
    solver = ACO_MKP(prize, weight, n_ants=20, ...)
    objs, paths, logps, trace = solver.sample()

    # OP
    solver = ACO_OP(distances, prizes, max_len, n_ants=20, ...)
    objs, paths, logps, trace = solver.sample()
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
from typing import Optional, List, Tuple, Any

# Ensure src is in path to find C++ extension if not installed globally
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    import faco_opt
except ImportError:
    # Try importing from current directory if compiled in-place at root
    try:
        import faco_opt
    except ImportError:
        # Check if it is in src but import failed
        raise ImportError(
            "C++ backend 'faco_opt' not found. Please build the C++ extension in src/."
        )

# Import extended C++ backend utilities
try:
    from cpp_aco_wrapper import (
        load_alphaant_cpp_module,
        to_numpy_matrix,
        to_numpy_vector,
        trace_paths_to_tensor,
        fresh_seed,
    )
except ImportError:
    # Define stubs for when the extended backend is not available
    def load_alphaant_cpp_module():
        raise ImportError(
            "C++ backend 'cpp_aco_wrapper' not found. "
            "Please ensure the AlphaAnt C++ extensions are built and in PYTHONPATH."
        )
    def trace_paths_to_tensor(trace, device):
        return torch.zeros(0, device=device, dtype=torch.long)
    def to_numpy_matrix(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    def to_numpy_vector(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    def fresh_seed(seed):
        import random
        return random.randint(0, 2**31-1) if seed is None else seed

# Constants
EPS = 1e-10


def set_faco_cpp_threads(n_threads: int) -> None:
    """Configure OpenMP thread count for the C++ backend."""
    faco_opt.set_num_threads(int(n_threads))


def get_faco_cpp_threads() -> int:
    """Return the current maximum OpenMP thread count exposed by the backend."""
    return int(faco_opt.get_max_threads())


class _BaseMFACO:
    """Common base class for MFACO solvers with shared sample/update logic."""

    def _init_torch_buffers(self, device):
        """Initialize torch tensors from C++ backend."""
        _init_torch_sparse_buffers(self, device)

    def sample(self, invtemp: float = 1.0, require_prob: bool = False, prior=None, parallel_traced: bool = True):
        # Prepare prior
        prior = _prepare_prior(prior, self.n, self.k)
        if prior is not None and prior.ndim == 2 and prior.shape == (self.n, self.n):
            prior = _project_dense_prior_to_sparse(prior, self.n, self.nn_list)

        costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival = self._cpp.sample(
            invtemp, require_prob, prior, parallel_traced
        )

        if require_prob and self._enable_torch_sync:
            self.sync_pheromone_to_torch()

        if isinstance(survival, np.ndarray):
            survival = torch.from_numpy(survival).to(self.device)

        return costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival


def _as_numpy(x, dtype):
    """Convert input to contiguous numpy array with specified dtype."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=dtype))


def _prepare_prior(prior, n, k):
    """Convert and reshape prior to (n, k) format."""
    if prior is None:
        return None
    arr = _as_numpy(prior, np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2 and arr.shape == (n, n):
        # Requires nn_list - will be handled by caller
        pass
    elif arr.ndim == 2 and arr.shape == (n * k, 1):
        arr = arr.reshape(n, k)
    elif arr.ndim == 1 and arr.size == n * k:
        arr = arr.reshape(n, k)
    return arr


def _split_ant_counts(n_ants, n_heads):
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    base = int(n_ants) // int(n_heads)
    rem = int(n_ants) % int(n_heads)
    counts = [base + (1 if h < rem else 0) for h in range(int(n_heads))]
    if any(c <= 0 for c in counts):
        raise ValueError(f"n_ants={n_ants} must be >= n_heads={n_heads}")
    return counts


def _concat_optional_arrays(parts, dtype):
    valid = [np.asarray(x, dtype=dtype) for x in parts if x is not None]
    if not valid:
        return None
    return np.concatenate(valid, axis=0)


def _sync_pheromone_to_torch(aco, device):
    """Shared implementation for syncing pheromone from C++ to torch."""
    phe_np = np.asarray(aco._cpp.pheromone_sparse_np)
    aco._pheromone_sparse.copy_(torch.from_numpy(phe_np).to(device))


def _compute_mfaco_logits(tau, h, alpha, invtemp, disable_heuristic):
    """Compute logits for MFACO solvers with invtemp scaling."""
    EPS = 1e-10
    tau = tau.clamp_min(EPS)
    logit = alpha * torch.log(tau)
    if not disable_heuristic:
        h = h.clamp_min(EPS)
        logit = logit + (float(invtemp) if invtemp != 1.0 else 1.0) * torch.log(h)
    return logit


def _compute_aco_logits(tau, h, alpha, beta):
    """Compute logits for ACO solvers with fixed beta."""
    EPS = 1e-10
    tau = tau.clamp_min(EPS)
    logit = alpha * torch.log(tau)
    h = h.clamp_min(EPS)
    logit = logit + beta * torch.log(h)
    return logit


def _ls_scope_to_int(ls_scope: str) -> int:
    mapping = {"localized": 0, "global": 1}
    if ls_scope not in mapping:
        raise ValueError(f"Unknown ls_scope={ls_scope!r}")
    return mapping[ls_scope]


def _ls_budget_to_int(ls_budget: str) -> int:
    mapping = {"truncated": 0, "full": 1}
    if ls_budget not in mapping:
        raise ValueError(f"Unknown ls_budget={ls_budget!r}")
    return mapping[ls_budget]


def _as_float_tensor(value, device):
    """Convert optional numpy/torch inputs to float tensors on target device."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _init_torch_sparse_buffers(obj, device):
    """Initialize torch sparse buffers from C++ backend."""
    obj._pheromone_sparse = torch.from_numpy(obj._cpp.pheromone_sparse_np).to(device)
    obj._h_sparse_torch = torch.from_numpy(obj._cpp.heuristic_sparse_np).to(device)
    obj._nn_torch = torch.from_numpy(obj._cpp.nn_list).to(device)


# =============================================================================
# Extended ACO Classes (BPP, MKP, OP)
# =============================================================================

class ACO_BPP:
    """
    Standard ACO for Bin Packing Problem (not focused/perturbative).

    Args:
        demand: Demand tensor of shape (n+1,) (first element is 0 for depot)
        capacity: Bin capacity
        n_ants: Number of ants
        decay: Pheromone decay rate
        alpha: Pheromone weight
        beta: Heuristic weight
        elitist: Use elitist pheromone update
        device: Device to use
        seed: Random seed
    """

    def __init__(
        self,
        demand: torch.Tensor,
        capacity: float = 150.0,
        n_ants: int = 20,
        decay: float = 0.9,  # Use 0.9 like DeepACO (not 0.01)
        alpha: float = 1.0,
        beta: float = 1.0,
        elitist: bool = False,  # Use False like DeepACO (not True)
        heuristic: Optional[torch.Tensor] = None,  # Allow custom heuristic (neural prior)
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.device = device
        self.capacity = float(capacity)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.elitist = bool(elitist)
        self.shortest_path = None
        self.best_cost = -float('inf')

        # Convert demand to numpy
        demand_np = to_numpy_vector(demand)
        self._n = len(demand_np) - 1  # Exclude depot

        # Initialize pheromone
        # Use 1.0 like DeepACO (not 10.0)
        self.pheromone = torch.ones((self._n + 1, self._n + 1), device=device, dtype=torch.float32)

        # Initialize heuristic (use provided heuristic or default to demand values)
        if heuristic is not None:
            self.heuristic = heuristic.to(device=device, dtype=torch.float32)
        else:
            # Use demand values directly like DeepACO (not inverse demand)
            self.heuristic = demand.unsqueeze(0).repeat(self._n + 1, 1).to(device)
            self.heuristic[:, 0] = 1e-5  # Depot has low heuristic

        # Load C++ solver
        module = load_alphaant_cpp_module()
        self._solver = module.ACO_BPP(
            demand_np,
            int(n_ants),
            float(decay),
            float(alpha),
            float(beta),
            bool(elitist),
            to_numpy_matrix(self.pheromone),
            to_numpy_matrix(self.heuristic),
            float(capacity),
            fresh_seed(seed),
        )

        # Sync state from C++
        self._sync_from_cpp()

    def _sync_cpp_inputs(self):
        """Sync pheromone and heuristic to C++."""
        self._solver.set_pheromone(to_numpy_matrix(self.pheromone))
        self._solver.set_heuristic(to_numpy_matrix(self.heuristic))

    def _sync_from_cpp(self):
        """Sync state from C++ backend."""
        self.pheromone = torch.as_tensor(
            self._solver.get_pheromone(), device=self.device, dtype=torch.float32
        )
        self.heuristic = torch.as_tensor(
            self._solver.get_heuristic(), device=self.device, dtype=torch.float32
        )
        # BPP has explicit getters
        self.best_cost = float(self._solver.get_best_fitness())
        shortest = self._solver.get_shortest_path()
        self.shortest_path = torch.as_tensor(shortest, device=self.device, dtype=torch.long) if shortest else None

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = True,
    ):
        """
        Sample solutions with optional neural prior.

        Args:
            invtemp: Inverse temperature for heuristic
            require_prob: Whether to require log probabilities
            prior: Neural prior tensor of shape (n+1, n+1)
            parallel_traced: Whether to use parallel tracing

        Returns:
            costs: Cost for each ant
            flats: Flat solution representation
            logps: Log probabilities
            traces: Trace information
        """
        self._sync_cpp_inputs()

        # Handle prior
        if prior is not None:
            if isinstance(prior, torch.Tensor):
                prior = prior.detach().to("cpu", dtype=torch.float32).numpy()
            else:
                prior = np.asarray(prior, dtype=np.float32)

        costs_np, trace = self._solver.sample_trace()
        paths = trace_paths_to_tensor(trace, self.device)

        # Use C++ costs (fitness-based: -fitness / n_bins)
        # C++ optimizes for fitness (better packing), which should correlate with fewer bins
        costs = torch.as_tensor(costs_np, device=self.device, dtype=torch.float32)

        # Compute log probabilities
        if require_prob:
            logps = self._replay_logp_from_paths(paths, prior)
        else:
            logps = None

        # Sync pheromone if needed
        if require_prob:
            self._sync_from_cpp()

        return costs, paths, logps, trace

    def _replay_logp_from_paths(
        self,
        paths: torch.Tensor,
        prior: Optional[np.ndarray] = None,
        pheromone: Optional[torch.Tensor] = None,
        heuristic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replay log probabilities from paths."""
        device = self.device
        pheromone_t = _as_float_tensor(pheromone, device)
        heuristic_t = _as_float_tensor(heuristic, device)
        if pheromone_t is None:
            pheromone_t = self.pheromone.to(device=device, dtype=torch.float32)
        if heuristic_t is None:
            heuristic_t = self.heuristic.to(device=device, dtype=torch.float32)
        prior_t = _as_float_tensor(prior, device)

        actions = paths[0].to(device=device)
        visit_mask = torch.ones((actions.size(0), self.n + 1), device=device, dtype=torch.float32)

        # Update visit mask
        visit_mask[torch.arange(actions.size(0), device=device), actions] = 0
        visit_mask[:, 0] = 1
        visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = 0

        # Update capacity mask
        used_capacity = torch.zeros((actions.size(0),), device=device, dtype=torch.float32)
        used_capacity, capacity_mask = self._update_capacity_mask(
            self.demand.to(device=device, dtype=torch.float32),
            self.capacity,
            actions,
            used_capacity,
        )

        prev = actions
        log_probs = []

        for step in range(1, paths.size(0)):
            log_weights = self.alpha * torch.log(pheromone_t[prev].clamp_min(EPS))
            log_weights = log_weights + self.beta * torch.log(heuristic_t[prev].clamp_min(EPS))
            if prior_t is not None:
                log_weights = log_weights + prior_t[prev]

            valid_mask = (visit_mask * capacity_mask) > 0
            masked_log_weights = log_weights.masked_fill(~valid_mask, float("-inf"))
            actions = paths[step].to(device=device)
            chosen = masked_log_weights.gather(1, actions.unsqueeze(1)).squeeze(1)
            normalizer = torch.logsumexp(masked_log_weights, dim=1)
            step_logp = torch.zeros_like(chosen)
            chosen_valid = valid_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
            valid_rows = valid_mask.any(dim=1) & chosen_valid
            step_logp[valid_rows] = chosen[valid_rows] - normalizer[valid_rows]
            log_probs.append(step_logp)

            visit_mask = visit_mask.clone()
            visit_mask[torch.arange(actions.size(0), device=device), actions] = 0
            visit_mask[:, 0] = 1
            visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = 0

            used_capacity, capacity_mask = self._update_capacity_mask(
                self.demand.to(device=device, dtype=torch.float32),
                self.capacity,
                actions,
                used_capacity,
            )
            prev = actions

        if not log_probs:
            return torch.zeros((0, paths.size(1)), device=device, dtype=torch.float32)
        return torch.stack(log_probs)

    def _update_capacity_mask(self, demand: torch.Tensor, capacity: float, cur_nodes: torch.Tensor, used_capacity: torch.Tensor):
        """Update capacity mask."""
        capacity_mask = torch.ones((cur_nodes.size(0), demand.size(0)), device=cur_nodes.device, dtype=torch.float32)
        used_capacity[cur_nodes == 0] = 0
        used_capacity = used_capacity + demand[cur_nodes]
        capacity_mask[demand.unsqueeze(0) > (capacity - used_capacity).unsqueeze(1)] = 0
        return used_capacity, capacity_mask

    def update_pheromone(self, paths: torch.Tensor, costs: Any) -> None:
        """Update pheromone from a full sampled path batch."""
        if isinstance(paths, torch.Tensor):
            paths_np = paths.detach().to(device="cpu", dtype=torch.long).contiguous().numpy()
        else:
            paths_np = np.asarray(paths, dtype=np.int64)

        # For BPP, we need to compute the negative fitness value for pheromone update
        # The C++ code expects: costs = -fitness / n_bins (as returned by gen_path_costs_vector)
        # Then it negates again for deposit: deposit = -costs
        # So we need to pass the negative fitness value
        if isinstance(costs, torch.Tensor):
            costs_t = costs.detach()
        else:
            costs_t = torch.as_tensor(costs, dtype=torch.float32)

        # Compute fitness values from paths
        seq_len = paths_np.shape[0]
        problem_size = self._n
        fitness_list = []
        for ant_idx in range(paths_np.shape[1]):
            ant_path = paths_np[:, ant_idx]
            # Count trailing zeros
            trailing_zeros = 0
            for step in range(seq_len - 1, -1, -1):
                if ant_path[step] == 0:
                    trailing_zeros += 1
                else:
                    break
            n_bins = max(1, seq_len - trailing_zeros - problem_size + 1)

            # Compute fitness (sum of squared load ratios)
            fitness = 0.0
            sub_fitness = 0.0
            for step in range(1, seq_len):
                node = ant_path[step]
                if node != 0:
                    sub_fitness += self.demand[node].item()
                else:
                    fitness += (sub_fitness / self.capacity) ** 2
                    sub_fitness = 0.0

            # Cost = -fitness / n_bins (negative value, as expected by C++)
            # The C++ code will negate this again for deposit
            cost = -fitness / n_bins
            fitness_list.append(cost)

        costs_np = np.array(fitness_list, dtype=np.float64)

        if paths_np.ndim != 2 or paths_np.shape[1] != self.n_ants:
            raise ValueError(f"paths must have shape (seq_len, {self.n_ants})")
        if costs_np.ndim != 1 or costs_np.shape[0] != self.n_ants:
            raise ValueError(f"costs must have shape ({self.n_ants},)")

        self._sync_cpp_inputs()
        self._solver.update_pheromone_from_paths(
            paths_np,
            costs_np,
        )
        # Update best seen in Python since C++ update_pheromone_from_paths doesn't update incumbent
        # BPP costs are -fitness / n_bins, so we minimize to maximize fitness
        best_idx = np.argmin(costs_np)
        iteration_best_cost = -costs_np[best_idx]  # fitness
        if iteration_best_cost > self.best_cost:
            self.best_cost = iteration_best_cost
            self.shortest_path = torch.as_tensor(paths_np[:, best_idx], device=self.device, dtype=torch.long)

        self._sync_from_cpp()

    def run(self, n_iterations: int) -> float:
        """Run ACO for n_iterations and return best fitness."""
        self._sync_cpp_inputs()
        best_fitness = float(self._solver.run(int(n_iterations)))
        self._sync_from_cpp()
        return best_fitness

    @property
    def demand(self) -> torch.Tensor:
        """Get demand tensor."""
        return self.heuristic[0, :]

    @property
    def n(self) -> int:
        """Get problem size."""
        return self._n


class ACO_MKP:
    """
    Standard ACO for Multi-dimensional Knapsack Problem (not focused/perturbative).

    Args:
        prize: Prize tensor of shape (n,)
        weight: Weight tensor of shape (n, m)
        n_ants: Number of ants
        decay: Pheromone decay rate
        alpha: Pheromone weight
        beta: Heuristic weight
        elitist: Use elitist pheromone update
        min_max: Use min-max pheromone bounds
        device: Device to use
        seed: Random seed
    """

    def __init__(
        self,
        prize: torch.Tensor,
        weight: torch.Tensor,
        n_ants: int = 20,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        elitist: bool = False,
        min_max: bool = False,
        heuristic: Optional[torch.Tensor] = None,  # Allow custom heuristic (neural prior)
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.device = device
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.elitist = bool(elitist)
        self.min_max = bool(min_max)
        self.shortest_path = None
        self.best_cost = -float('inf')

        # Convert to numpy
        prize_np = to_numpy_vector(prize)
        weight_np = to_numpy_matrix(weight)
        self._n = len(prize_np)
        self._m = weight_np.shape[1]

        # Initialize pheromone
        n_plus_1 = self._n + 1
        if min_max:
            self.min = 0.1
            self.max = 20.0
            self.pheromone = torch.ones((n_plus_1, n_plus_1), device=device, dtype=torch.float32) * self.min
        else:
            self.pheromone = torch.ones((n_plus_1, n_plus_1), device=device, dtype=torch.float32)

        # Initialize heuristic (use provided heuristic or default to prize/weight ratio)
        if heuristic is not None:
            self.heuristic = heuristic.to(device=device, dtype=torch.float32)
            # Extract (n, n) portion for C++ solver
            heuristic_raw = self.heuristic[:self._n, :self._n]
        else:
            # Heuristic based on prize/weight ratio (C++ expects (n, n) initially)
            ratio = prize / (weight.sum(dim=1) + 1e-10)
            heuristic_raw = ratio.unsqueeze(0).repeat(self._n, 1).to(device)
            # Store the expanded version for internal use
            self.heuristic = torch.zeros((n_plus_1, n_plus_1), device=device, dtype=torch.float32)
            self.heuristic[:self._n, :self._n] = heuristic_raw

        # Store weight for later use
        self._weight = weight.to(device)

        # Load C++ solver with (n, n) heuristic
        module = load_alphaant_cpp_module()
        self._solver = module.ACO_MKP(
            prize_np,
            weight_np,
            int(n_ants),
            float(decay),
            float(alpha),
            float(beta),
            bool(elitist),
            bool(min_max),
            to_numpy_matrix(self.pheromone),
            to_numpy_matrix(heuristic_raw),  # Pass (n, n) heuristic
            float(self.min) if min_max else None,
            fresh_seed(seed),
        )

        # Sync state from C++
        self._sync_from_cpp()

    def _sync_cpp_inputs(self):
        """Sync pheromone and heuristic to C++."""
        self._solver.set_pheromone(to_numpy_matrix(self.pheromone))
        self._solver.set_heuristic(to_numpy_matrix(self.heuristic))

    def _sync_from_cpp(self):
        """
        Sync state from C++ backend.
        Note: MKP/OP do not have get_shortest_path() bindings.
        Calling run(0) retrieves (best_obj, best_sol) without new iterations.
        """
        self.pheromone = torch.as_tensor(
            self._solver.get_pheromone(), device=self.device, dtype=torch.float32
        )
        self.heuristic = torch.as_tensor(
            self._solver.get_heuristic(), device=self.device, dtype=torch.float32
        )
        best_obj, best_sol_np = self._solver.run(0)
        self.best_cost = float(best_obj)
        self.shortest_path = torch.as_tensor(best_sol_np, device=self.device, dtype=torch.long)

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = True,
    ):
        """
        Sample solutions with optional neural prior.

        Args:
            invtemp: Inverse temperature for heuristic
            require_prob: Whether to require log probabilities
            prior: Neural prior tensor of shape (n+1, n+1)
            parallel_traced: Whether to use parallel tracing

        Returns:
            objs: Objective values for each ant
            flats: Flat solution representation
            logps: Log probabilities
            traces: Trace information
        """
        self._sync_cpp_inputs()

        # Handle prior
        if prior is not None:
            if isinstance(prior, torch.Tensor):
                prior = prior.detach().to("cpu", dtype=torch.float32).numpy()
            else:
                prior = np.asarray(prior, dtype=np.float32)

        objs_np, trace = self._solver.sample_trace()
        objs = torch.as_tensor(objs_np, device=self.device, dtype=torch.float32)
        paths = trace_paths_to_tensor(trace, self.device)

        # Compute log probabilities
        if require_prob:
            logps = self._replay_logp_from_paths(paths, prior)
        else:
            logps = None

        # Sync pheromone if needed
        if require_prob:
            self._sync_from_cpp()

        return objs, paths, logps, trace

    def _replay_logp_from_paths(
        self,
        paths: torch.Tensor,
        prior: Optional[np.ndarray] = None,
        pheromone: Optional[torch.Tensor] = None,
        heuristic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replay log probabilities from paths."""
        device = self.device
        pheromone_t = _as_float_tensor(pheromone, device)
        heuristic_t = _as_float_tensor(heuristic, device)
        if pheromone_t is None:
            pheromone_t = self.pheromone.to(device=device, dtype=torch.float32)
        if heuristic_t is None:
            heuristic_t = self.heuristic.to(device=device, dtype=torch.float32)
        prior_t = _as_float_tensor(prior, device)
        n_plus_1 = self.pheromone.size(0)
        n = n_plus_1 - 1

        items = paths[0]
        mask = torch.ones((self.n_ants, n_plus_1), device=device, dtype=torch.float32)
        dummy_mask = torch.ones((self.n_ants, n_plus_1), device=device, dtype=torch.float32)
        dummy_mask[:, -1] = 0
        knapsack = torch.zeros((self.n_ants, self._m), device=device, dtype=torch.float32)

        mask, knapsack = self._update_knapsack(mask, knapsack, items)
        dummy_mask = self._update_dummy_state(mask, dummy_mask)

        log_probs = []
        prev = items

        for step in range(1, paths.size(0)):
            log_weights = self.alpha * torch.log(pheromone_t[prev].clamp_min(EPS))
            log_weights = log_weights + self.beta * torch.log(heuristic_t[prev].clamp_min(EPS))
            if prior_t is not None:
                log_weights = log_weights + prior_t[prev]

            valid_mask = (mask * dummy_mask) > 0
            masked_log_weights = log_weights.masked_fill(~valid_mask, float("-inf"))
            items = paths[step]
            chosen_weights = masked_log_weights.gather(1, items.unsqueeze(1)).squeeze(1)
            normalizer = torch.logsumexp(masked_log_weights, dim=1)
            step_logp = torch.zeros_like(chosen_weights)
            chosen_valid = valid_mask.gather(1, items.unsqueeze(1)).squeeze(1)
            valid_rows = valid_mask.any(dim=1) & chosen_valid
            step_logp[valid_rows] = chosen_weights[valid_rows] - normalizer[valid_rows]
            log_probs.append(step_logp)

            mask = mask.clone()
            dummy_mask = dummy_mask.clone()
            knapsack = knapsack.clone()
            mask, knapsack = self._update_knapsack(mask, knapsack, items)
            dummy_mask = self._update_dummy_state(mask, dummy_mask)
            prev = items

        if not log_probs:
            return torch.zeros((0, paths.size(1)), device=device, dtype=torch.float32)
        return torch.stack(log_probs)

    def _update_knapsack(self, mask: torch.Tensor, knapsack: torch.Tensor, new_item: torch.Tensor):
        """Update knapsack state."""
        ant_idx = torch.arange(self.n_ants, device=mask.device)
        mask[ant_idx, new_item] = 0

        is_real = new_item < self._n
        if is_real.any():
            knapsack[is_real] += self._weight[new_item[is_real]]

        # Check candidates for feasibility
        for ant_idx in range(self.n_ants):
            candidates = torch.nonzero(mask[ant_idx]).squeeze(-1)
            if candidates.dim() > 0 and candidates.numel() > 0:
                real_candidates = candidates[candidates < self._n]
                if real_candidates.numel() > 0:
                    candidate_weights = self._weight[real_candidates]
                    infeasible = (knapsack[ant_idx].unsqueeze(0) + candidate_weights > self._n // 2).any(dim=1)
                    mask[ant_idx, real_candidates[infeasible]] = 0
        mask[:, -1] = 1  # Dummy node is always valid until chosen
        return mask, knapsack

    def _update_dummy_state(self, mask: torch.Tensor, dummy_mask: torch.Tensor):
        """Update dummy node state."""
        finished = (mask[:, :-1] == 0).all(dim=1)
        dummy_mask[finished] = 1
        return dummy_mask

    def update_pheromone(self, paths: torch.Tensor, costs: Any) -> None:
        """Update pheromone from a full sampled path batch."""
        if isinstance(paths, torch.Tensor):
            paths_np = paths.detach().to(device="cpu", dtype=torch.long).contiguous().numpy()
        else:
            paths_np = np.asarray(paths, dtype=np.int64)

        if isinstance(costs, torch.Tensor):
            costs_np = costs.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        else:
            costs_np = np.asarray(costs, dtype=np.float64)
        if paths_np.ndim != 2 or paths_np.shape[1] != self.n_ants:
            raise ValueError(f"paths must have shape (seq_len, {self.n_ants})")
        if costs_np.ndim != 1 or costs_np.shape[0] != self.n_ants:
            raise ValueError(f"costs must have shape ({self.n_ants},)")

        self._sync_cpp_inputs()
        self._solver.update_pheromone_from_paths(
            paths_np,
            costs_np,
        )
        # Update best seen in Python since C++ update_pheromone_from_paths doesn't update incumbent
        # MKP is maximization
        best_idx = np.argmax(costs_np)
        iteration_best_cost = costs_np[best_idx]
        if iteration_best_cost > self.best_cost:
            self.best_cost = iteration_best_cost
            self.shortest_path = torch.as_tensor(paths_np[:, best_idx], device=self.device, dtype=torch.long)

        self._sync_from_cpp()

    def run(self, n_iterations: int) -> tuple[float, torch.Tensor]:
        """Run ACO for n_iterations and return best objective and solution."""
        self._sync_cpp_inputs()
        best_obj, best_sol_np = self._solver.run(int(n_iterations))
        self._sync_from_cpp()
        best_sol = torch.as_tensor(best_sol_np, device=self.device, dtype=torch.long)
        return best_obj, best_sol

    @property
    def weight(self) -> torch.Tensor:
        """Get weight tensor."""
        return getattr(self, '_weight', None)

    @property
    def n(self) -> int:
        """Get problem size."""
        return self._n


class ACO_OP:
    """
    Standard ACO for Orienteering Problem (not focused/perturbative).

    Args:
        distances: Distance matrix of shape (n, n)
        prizes: Prize tensor of shape (n,)
        max_len: Maximum route length
        n_ants: Number of ants
        decay: Pheromone decay rate
        alpha: Pheromone weight
        beta: Heuristic weight
        elitist: Use elitist pheromone update
        min_max: Use min-max pheromone bounds
        device: Device to use
        seed: Random seed
    """

    def __init__(
        self,
        distances: torch.Tensor,
        prizes: torch.Tensor,
        max_len: float,
        n_ants: int = 20,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        elitist: bool = False,
        min_max: bool = False,
        heuristic: Optional[torch.Tensor] = None,  # Allow custom heuristic (neural prior)
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.device = device
        self.max_len = float(max_len)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.elitist = bool(elitist)
        self.min_max = bool(min_max)
        self.shortest_path = None
        self.best_cost = -float('inf')

        # Convert to numpy
        distances_np = to_numpy_matrix(distances)
        prizes_np = to_numpy_vector(prizes)
        self._n = len(prizes_np)

        # Initialize pheromone
        n_plus_1 = self._n + 1
        if min_max:
            self.min = 0.1
            self.max = -1.0
            self.pheromone = torch.ones((n_plus_1, n_plus_1), device=device, dtype=torch.float32) * self.min
        else:
            self.pheromone = torch.ones((n_plus_1, n_plus_1), device=device, dtype=torch.float32)

        # Initialize heuristic (use provided heuristic or default to prize values)
        if heuristic is not None:
            self.heuristic = heuristic.to(device=device, dtype=torch.float32)
            # Extract (n, n) portion for C++ solver
            heuristic_raw = self.heuristic[:self._n, :self._n]
        else:
            # Heuristic based on prize values (C++ expects (n, n) initially)
            heuristic_raw = prizes.unsqueeze(0).repeat(self._n, 1).to(device)
            # Store the expanded version for internal use
            self.heuristic = torch.zeros((n_plus_1, n_plus_1), device=device, dtype=torch.float32)
            self.heuristic[:self._n, :self._n] = heuristic_raw

        # Store distances for later use, including the dummy node used by the solver.
        self._distances = torch.zeros((n_plus_1, n_plus_1), device=device, dtype=torch.float32)
        self._distances[:self._n, :self._n] = distances.to(device=device, dtype=torch.float32)

        # Load C++ solver with (n, n) pheromone and heuristic
        module = load_alphaant_cpp_module()
        self._solver = module.ACO_OP(
            distances_np,
            prizes_np,
            float(max_len),
            int(n_ants),
            float(decay),
            float(alpha),
            float(beta),
            bool(elitist),
            bool(min_max),
            to_numpy_matrix(self.pheromone[:self._n, :self._n]),  # Pass (n, n) pheromone
            to_numpy_matrix(heuristic_raw),  # Pass (n, n) heuristic
            float(self.min) if min_max else None,
            fresh_seed(seed),
        )

        # Sync state from C++
        self._sync_from_cpp()

    def _sync_cpp_inputs(self):
        """Sync pheromone and heuristic to C++."""
        self._solver.set_pheromone(to_numpy_matrix(self.pheromone))
        self._solver.set_heuristic(to_numpy_matrix(self.heuristic))

    def _sync_from_cpp(self):
        """
        Sync state from C++ backend.
        Note: MKP/OP do not have get_shortest_path() bindings.
        Calling run(0) retrieves (best_obj, best_sol) without new iterations.
        """
        self.pheromone = torch.as_tensor(
            self._solver.get_pheromone(), device=self.device, dtype=torch.float32
        )
        self.heuristic = torch.as_tensor(
            self._solver.get_heuristic(), device=self.device, dtype=torch.float32
        )
        best_obj, best_sol_np = self._solver.run(0)
        self.best_cost = float(best_obj)
        self.shortest_path = torch.as_tensor(best_sol_np, device=self.device, dtype=torch.long)

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = True,
    ):
        """
        Sample solutions with optional neural prior.

        Args:
            invtemp: Inverse temperature for heuristic
            require_prob: Whether to require log probabilities
            prior: Neural prior tensor of shape (n+1, n+1)
            parallel_traced: Whether to use parallel tracing

        Returns:
            objs: Objective values for each ant
            flats: Flat solution representation
            logps: Log probabilities
            traces: Trace information
        """
        self._sync_cpp_inputs()

        # Handle prior
        if prior is not None:
            if isinstance(prior, torch.Tensor):
                prior = prior.detach().to("cpu", dtype=torch.float32).numpy()
            else:
                prior = np.asarray(prior, dtype=np.float32)

        objs_np, trace = self._solver.sample_trace()
        objs = torch.as_tensor(objs_np, device=self.device, dtype=torch.float32)
        paths = trace_paths_to_tensor(trace, self.device)

        # Compute log probabilities
        if require_prob:
            logps = self._replay_logp_from_paths(paths, prior)
        else:
            logps = None

        # Sync pheromone if needed
        if require_prob:
            self._sync_from_cpp()

        return objs, paths, logps, trace

    def _replay_logp_from_paths(
        self,
        paths: torch.Tensor,
        prior: Optional[np.ndarray] = None,
        pheromone: Optional[torch.Tensor] = None,
        heuristic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replay log probabilities from paths."""
        device = self.device
        pheromone_t = _as_float_tensor(pheromone, device)
        heuristic_t = _as_float_tensor(heuristic, device)
        if pheromone_t is None:
            pheromone_t = self.pheromone.to(device=device, dtype=torch.float32)
        if heuristic_t is None:
            heuristic_t = self.heuristic.to(device=device, dtype=torch.float32)
        prior_t = _as_float_tensor(prior, device)
        n_plus_1 = self.pheromone.size(0)
        n = n_plus_1 - 1

        cur_node = paths[0]
        mask = torch.ones((self.n_ants, n_plus_1), device=device, dtype=torch.float32)
        travel_dis = torch.zeros((self.n_ants,), device=device, dtype=torch.float32)

        mask = self._update_mask(travel_dis, cur_node, mask)
        log_probs = []

        for step in range(1, paths.size(0)):
            prev = paths[step - 1]
            log_weights = self.alpha * torch.log(pheromone_t[prev].clamp_min(EPS))
            log_weights = log_weights + self.beta * torch.log(heuristic_t[prev].clamp_min(EPS))
            if prior_t is not None:
                log_weights = log_weights + prior_t[prev]

            valid_mask = mask > 0
            masked_log_weights = log_weights.masked_fill(~valid_mask, float("-inf"))
            cur_node = paths[step]
            chosen_weights = masked_log_weights.gather(1, cur_node.unsqueeze(1)).squeeze(1)
            normalizer = torch.logsumexp(masked_log_weights, dim=1)
            step_logp = torch.zeros_like(chosen_weights)
            chosen_valid = valid_mask.gather(1, cur_node.unsqueeze(1)).squeeze(1)
            valid_rows = valid_mask.any(dim=1) & chosen_valid
            step_logp[valid_rows] = chosen_weights[valid_rows] - normalizer[valid_rows]
            log_probs.append(step_logp)

            travel_dis += self._distances[prev, cur_node]
            mask = mask.clone()
            mask = self._update_mask(travel_dis, cur_node, mask)

        if not log_probs:
            return torch.zeros((0, paths.size(1)), device=device, dtype=torch.float32)
        return torch.stack(log_probs)

    def _update_mask(self, travel_dis: torch.Tensor, cur_node: torch.Tensor, mask: torch.Tensor):
        """Update mask based on budget constraint."""
        mask[torch.arange(self.n_ants, device=mask.device), cur_node] = 0
        for ant_id in range(self.n_ants):
            if cur_node[ant_id] != self._n:
                _m = mask[ant_id]
                candidates = torch.nonzero(_m).squeeze(-1)
                if candidates.dim() > 0:
                    trails = travel_dis[ant_id] + self._distances[cur_node[ant_id], candidates] + self._distances[candidates, 0]
                    fail_idx = candidates[trails > self.max_len]
                    _m[fail_idx] = 0
        mask[:, -1] = 0
        go2dummy = (mask[:, :-1] == 0).all(dim=1)
        mask[go2dummy, -1] = 1
        return mask

    def update_pheromone(self, paths: torch.Tensor, costs: Any) -> None:
        """Update pheromone from a full sampled path batch."""
        if isinstance(paths, torch.Tensor):
            paths_np = paths.detach().to(device="cpu", dtype=torch.long).contiguous().numpy()
        else:
            paths_np = np.asarray(paths, dtype=np.int64)

        if isinstance(costs, torch.Tensor):
            costs_np = costs.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        else:
            costs_np = np.asarray(costs, dtype=np.float64)
        if paths_np.ndim != 2 or paths_np.shape[1] != self.n_ants:
            raise ValueError(f"paths must have shape (seq_len, {self.n_ants})")
        if costs_np.ndim != 1 or costs_np.shape[0] != self.n_ants:
            raise ValueError(f"costs must have shape ({self.n_ants},)")

        self._sync_cpp_inputs()
        self._solver.update_pheromone_from_paths(
            paths_np,
            costs_np,
        )
        # Update best seen in Python since C++ update_pheromone_from_paths doesn't update incumbent
        # OP is maximization
        best_idx = np.argmax(costs_np)
        iteration_best_cost = costs_np[best_idx]
        if iteration_best_cost > self.best_cost:
            self.best_cost = iteration_best_cost
            self.shortest_path = torch.as_tensor(paths_np[:, best_idx], device=self.device, dtype=torch.long)

        self._sync_from_cpp()

    def run(self, n_iterations: int) -> tuple[float, torch.Tensor]:
        """Run ACO for n_iterations and return best objective and solution."""
        self._sync_cpp_inputs()
        best_obj, best_sol_np = self._solver.run(int(n_iterations))
        self._sync_from_cpp()
        best_sol = torch.as_tensor(best_sol_np, device=self.device, dtype=torch.long)
        return best_obj, best_sol

    @property
    def distances(self) -> torch.Tensor:
        """Get distance tensor."""
        return getattr(self, '_distances', None)

    @property
    def n(self) -> int:
        """Get problem size."""
        return self._n


class MFACO_TSP(_BaseMFACO):
    """
    Unified MFACO TSP solver wrapping C++ backend.
    """

    def __init__(
        self,
        coords: torch.Tensor,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 32,
        min_new_edges: int = 8,
        decay: float = 0.1,
        alpha: float = 1.0,
        p_best: float = 0.05,
        use_local_search: bool = True,
        extend_ls: bool = False,
        smooth_mmas: bool = False,
        enable_torch_sync: bool = True,
        device: str = "cuda",
        disable_heuristic: bool = False,
        normalized_heuristic: bool = False,
        fixed_steps: int = 0,
        nls: bool = False,
        T_nls: int = 10,
        ls_scope: str = "localized",
        ls_budget: str = "truncated",
        ls_max_opt: int = 0,
        **kwargs
    ):
        self.device = device
        self.disable_heuristic = bool(disable_heuristic)
        self.extend_ls = bool(extend_ls)
        self.smooth_mmas = bool(smooth_mmas)
        self.normalized_heuristic = bool(normalized_heuristic)
        self.fixed_steps = int(fixed_steps)
        self.ls_scope = ls_scope
        self.ls_budget = ls_budget
        self.ls_max_opt = int(ls_max_opt)
        self.alpha = alpha
        self._enable_torch_sync = enable_torch_sync

        # Handle PyG Data object
        if not isinstance(coords, torch.Tensor) and hasattr(coords, "x"):
            coords = coords.x

        coords_np = _as_numpy(coords, np.float32)
        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords_np.shape}")
        self._coords_np = coords_np.copy()
        self._init_kwargs = dict(
            cand_list_size=cand_list_size,
            backup_list_size=backup_list_size,
            min_new_edges=min_new_edges,
            decay=decay,
            alpha=alpha,
            p_best=p_best,
            use_local_search=use_local_search,
            extend_ls=extend_ls,
            smooth_mmas=smooth_mmas,
            enable_torch_sync=enable_torch_sync,
            device=device,
            disable_heuristic=disable_heuristic,
            normalized_heuristic=normalized_heuristic,
            fixed_steps=fixed_steps,
            nls=nls,
            T_nls=T_nls,
            ls_scope=ls_scope,
            ls_budget=ls_budget,
            ls_max_opt=ls_max_opt,
        )

        self._cpp = faco_opt.MFACO_TSP(
            coords_np,
            n_ants,
            cand_list_size,
            backup_list_size,
            min_new_edges,
            decay,
            alpha,
            p_best,
            use_local_search,
            self.disable_heuristic,
            self.extend_ls,
            self.smooth_mmas,
            self.fixed_steps,
            nls,
            int(T_nls),
            _ls_scope_to_int(ls_scope),
            _ls_budget_to_int(ls_budget),
            self.ls_max_opt,
        )

        if self.normalized_heuristic and not self.disable_heuristic:
            h = np.asarray(self._cpp.heuristic_sparse_np)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)

        # Torch buffers
        self._init_torch_buffers(device)

    # Delegate properties
    @property
    def n(self) -> int: return self._cpp.n
    @property
    def n_ants(self) -> int: return self._cpp.n_ants
    @property
    def k(self) -> int: return self._cpp.k
    @property
    def bl(self) -> int: return self._cpp.bl
    @property
    def source_cost(self) -> float: return self._cpp.source_cost
    @property
    def best_cost(self) -> float: return self._cpp.best_cost
    @property
    def tau_min(self) -> float: return self._cpp.tau_min
    @property
    def tau_max(self) -> float: return self._cpp.tau_max

    @property
    def source_route(self) -> np.ndarray: return np.asarray(self._cpp.source_route)
    @property
    def best_route(self) -> np.ndarray: return np.asarray(self._cpp.best_route)

    @property
    def pheromone_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.pheromone_sparse_np)
    @pheromone_sparse_np.setter
    def pheromone_sparse_np(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        self._cpp.set_pheromone(arr)
        if self.enable_torch_sync:
            self._pheromone_sparse.copy_(torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(self.device))

    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)
    @property
    def backup_list(self) -> np.ndarray: return np.asarray(self._cpp.backup_list)
    @property
    def source_positions(self) -> np.ndarray: return np.asarray(self._cpp.source_positions)

    @property
    def h_sparse_torch(self) -> torch.Tensor: return self._h_sparse_torch
    @property
    def nn_torch(self) -> torch.Tensor: return self._nn_torch
    @property
    def pheromone_sparse(self) -> torch.Tensor: return self._pheromone_sparse

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(seed)


    def update_pheromone(self, best_flat, best_cost: float) -> None:
        """Update pheromone with best solution (unified API)."""
        self._cpp._update_pheromone_from_flat(np.asarray(best_flat, dtype=np.int32), float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def _update_pheromone_from_flat(self, best_flat: np.ndarray, best_cost: float) -> None:
        """Alias for update_pheromone (backward compatibility)."""
        self.update_pheromone(best_flat, best_cost)

    def sync_pheromone_to_torch(self) -> None:
        _sync_pheromone_to_torch(self, self.device)

    def sample_mixed_priors(
        self,
        priors,
        require_prob: bool = False,
        parallel_traced: bool = True,
        head_counts=None,
    ):
        """AlphaAnt-style mixed-ant deployment for sparse DyNACO priors.

        Each head prior gets a disjoint group of ants inside the same C++
        colony/sample call. The caller updates pheromone once with the global
        best ant from the returned population.
        """
        prior_arr = _as_numpy(priors, np.float32)
        if prior_arr.ndim != 3:
            raise ValueError(f"priors must have shape (heads, n, k) or (n_ants, n, k), got {prior_arr.shape}")
        if prior_arr.shape[1:] != (self.n, self.k):
            raise ValueError(f"priors trailing shape must be (n, k)=({self.n}, {self.k}), got {prior_arr.shape}")
        if prior_arr.shape[0] == self.n_ants:
            return self._cpp.sample_ant_priors(require_prob, prior_arr, parallel_traced)
        if prior_arr.shape[0] > self.n_ants:
            raise ValueError(
                f"number of heads must be <= n_ants ({self.n_ants}), got {prior_arr.shape[0]}"
            )
        counts_arr = None
        if head_counts is not None:
            counts_arr = np.asarray(head_counts, dtype=np.int32)
            if counts_arr.shape != (prior_arr.shape[0],):
                raise ValueError(
                    f"head_counts must have shape ({prior_arr.shape[0]},), got {counts_arr.shape}"
                )
            if int(counts_arr.sum()) != self.n_ants:
                raise ValueError(f"head_counts must sum to n_ants ({self.n_ants})")
        return self._cpp.sample_head_priors(
            require_prob, prior_arr, parallel_traced, counts_arr
        )

    @property
    def enable_torch_sync(self) -> bool:
        return self._enable_torch_sync

    def reset_timings(self) -> None:
        if hasattr(self._cpp, 'reset_timings'):
            self._cpp.reset_timings()

    def get_timings(self) -> dict:
        if hasattr(self._cpp, 'get_timings'):
            return self._cpp.get_timings()
        return {}

    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        logit = _compute_mfaco_logits(self._pheromone_sparse, self._h_sparse_torch, self._cpp.alpha, invtemp, self.disable_heuristic)
        if prior is not None:
            logit = logit + prior
        return torch.exp(logit)


class ACO_TSP:
    """
    Standard MMAS ACO TSP solver.
    """
    def __init__(
        self,
        coords: torch.Tensor,
        n_ants: int,
        cand_list_size: int = 32,
        decay: float = 0.9,
        alpha: float = 1.0, 
        beta: float = 1.0, # Heuristic weight
        p_best: float = 0.05,
        min_max: bool = True,
        device: str = "cuda",
        enable_torch_sync: bool = True,
        **kwargs
    ):
        self.device = device
        
        # Handle PyG Data object
        if not isinstance(coords, torch.Tensor) and hasattr(coords, "x"):
            coords = coords.x
        
        coords_np = _as_numpy(coords, np.float32)
        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords_np.shape}")

        self._cpp = faco_opt.ACO_TSP(
            coords_np,
            int(n_ants),
            int(cand_list_size),
            float(decay),
            float(alpha),
            float(beta),
            float(p_best),
            bool(min_max)
        )
        
        # Torch buffers
        _init_torch_sparse_buffers(self, device)
        self._enable_torch_sync = enable_torch_sync
        
        self.disable_heuristic = False # For compatibility

    # Delegate properties
    @property
    def n(self) -> int: return self._cpp.n
    @property
    def n_ants(self) -> int: return self._cpp.n_ants
    @property
    def k(self) -> int: return self._cpp.k
    
    @property
    def best_cost(self) -> float: return self._cpp.best_cost
    @property
    def tau_min(self) -> float: return self._cpp.tau_min
    @property
    def tau_max(self) -> float: return self._cpp.tau_max
    
    @property
    def best_route(self) -> np.ndarray: return np.asarray(self._cpp.best_route)

    @property
    def source_route(self) -> np.ndarray: return self.best_route
    
    @property
    def pheromone_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.pheromone_sparse_np)
    
    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)

    @property 
    def h_sparse_torch(self) -> torch.Tensor: return self._h_sparse_torch
    @property
    def nn_torch(self) -> torch.Tensor: return self._nn_torch
    @property
    def pheromone_sparse(self) -> torch.Tensor: return self._pheromone_sparse

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(seed)

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = True,
    ):
        # Prepare prior
        prior = _prepare_prior(prior, self.n, self.k)
        if prior is not None and prior.ndim == 2 and prior.shape == (self.n, self.n):
            prior = _project_dense_prior_to_sparse(prior, self.n, self.nn_list)

        # Returns: costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival
        ret = self._cpp.sample(require_prob, prior, parallel_traced)
        
        if require_prob and self._enable_torch_sync:
            self.sync_pheromone_to_torch()
            
        return ret 

    def update_pheromone(self, best_flat: np.ndarray, best_cost: float) -> None:
        self._cpp.update_pheromone(best_flat.astype(np.int32), float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def sync_pheromone_to_torch(self) -> None:
        _sync_pheromone_to_torch(self, self.device)
        
    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        # Reimplementation of MMAS prob calculation for PPO / Debug
        logit = _compute_aco_logits(self._pheromone_sparse, self._h_sparse_torch, self._cpp.alpha, self._cpp.beta)
        if prior is not None:
            logit = logit + prior
        return torch.exp(logit)

    def evaluate_log_prob_torch(self, paths: torch.Tensor, prior: torch.Tensor = None, invtemp: float = 1.0) -> torch.Tensor:
        """
        Evaluate log probability of paths (B, N).
        paths: (B, N) int64 tensor of node indices.
        prior: (N, K) tensor or None.
        Returns: (B,) log probability.
        """
        B, N = paths.shape
        device = paths.device
        
        # 1. Compute logits (N, K)
        # Logits are unnormalized log-weights.
        # logit[u, j] corresponds to edge (u, nn[u,j])
        logits = torch.log(self.prob_sparse_torch(invtemp=invtemp, prior=prior) + 1e-20) 
        # prob_sparse_torch returns exp(logit), so taking log returns logit back? 
        # prob_sparse_torch computes alpha*log(tau) + beta*log(eta) + prior.
        # Yes.
        
        # 2. Iterate steps
        # Maintain visited mask (B, N)
        visited = torch.zeros((B, N), dtype=torch.bool, device=device)
        
        log_prob_sum = torch.zeros(B, device=device)
        
        # nn indices: (N, K)
        nn = self.nn_torch # (N, K)
        
        # Initial step: Start node. Prob = 1/N? 
        # Usually ACO starts random uniform. Log prob depends if we model start choice.
        # Often ignored or treated as log(1/N).
        # We start from paths[:, 0].
        # Mark visited
        visited.scatter_(1, paths[:, 0:1], True)
        
        prev = paths[:, 0]
        
        # Pre-gather nn for all rows? No, distinct per row.
        # We process steps.
        
        range_b = torch.arange(B, device=device)
        
        for t in range(N - 1):
            curr = prev # (B,)
            next_node = paths[:, t+1] # (B,)
            
            # Identify which neighbor index 'j' corresponds to 'next_node'
            # We can lookup in nn[curr].
            # efficient way? 
            # nn is (N, K). nn[curr] is (B, K).
            # next_node is (B,).
            # match = (nn[curr] == next_node.unsqueeze(1)) -> (B, K) bool
            # idx = match.nonzero(something).
            # If not found (fallback case), probability is complicated (greedy fallback).
            # Assuming found in NN list.
            
            nn_curr = nn[curr] # (B, K)
            logits_curr = logits[curr] # (B, K)
            
            # Mask visited
            # neighbors = nn_curr. Flatten?
            # is_visited = visited.gather(1, nn_curr) # (B, K)
            # mask invalid neighbors
            is_visited = visited.gather(1, nn_curr)
            
            # Mask logits
            logits_valid = logits_curr.clone()
            logits_valid[is_visited] = float('-inf')
            
            # LogSumExp
            log_denom = torch.logsumexp(logits_valid, dim=1) # (B,)
            
            # Numerator: logit of the chosen 'next_node'
            # We need index j.
            # match mask
            is_next = (nn_curr == next_node.unsqueeze(1)) # (B, K)
            # If next_node not in nn_curr, is_next is all False.
            # In that case, we assume greedy fallback or zero prob? 
            # If MMAS C++ did greedy fallback, we should assign prob=1.0 (log=0)? 
            # Or -inf?
            # If we are training, we hope paths are within NN.
            # If fallback happened, strict log prob is -inf under "NN-only" policy.
            # But we can approximate.
            
            # To extract value efficiently:
            # sum(logits_valid * is_next) ? No, logits can be negative.
            # logits_valid[is_next] ? is_next might be sparse.
            # If multiple next (imposible), pick one.
            
            # Use max(logits_curr masked by is_next)? 
            # If is_next all false, we have a problem.
            
            # Assume unique match
            # We replace -inf with something safe for 'max'
            # But easier:
            
            has_match = is_next.any(dim=1)
            
            # For matched ones:
            log_numer = (logits_curr * is_next.float()).sum(dim=1) 
            # Note: if multiple matches (impossible for unique NN), sum might be wrong logit sum.
            # is_next is 0/1. sum gives logit value.
            
            step_logp = log_numer - log_denom
            
            # If not matched (fallback), assign logp = 0 (deterministic fallback assumption)
            # or handle gracefully.
            step_logp = torch.where(has_match, step_logp, torch.zeros_like(step_logp))
            
            log_prob_sum += step_logp
            
            # Update visited
            visited.scatter_(1, next_node.unsqueeze(1), True)
            
            prev = next_node
            
        return log_prob_sum

class MFACO_CVRP(_BaseMFACO):
    """
    Unified MFACO CVRP solver wrapping C++ backend.
    
    API is designed to match MFACO_TSP for consistent usage in training code.
    """
    
    def __init__(
        self,
        coords,          
        demand,          
        capacity: float,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 64,
        min_new_edges: int = 8,
        decay: float = 0.9,
        alpha: float = 1.0,
        p_best: float = 0.05,
        use_local_search: bool = True,
        disable_heuristic: bool = False,
        extend_ls: bool = False,
        smooth_mmas: bool = False,
        device: str = "cpu",
        enable_torch_sync: bool = True,
        normalized_heuristic: bool = False,
        fixed_steps: int = 0,
        nls: bool = False,
        T_nls: int = 10,
        ls_scope: str = "localized",
        ls_budget: str = "truncated",
        ls_max_opt: int = 0,
        **kwargs
    ):
        coords_np = _as_numpy(coords, np.float32)
        demand_np = _as_numpy(demand, np.float32)
        if demand_np.ndim != 1:
            raise ValueError("demand must be 1D")
        demand_np[0] = 0.0
        
        # Use _cpp for consistency with MFACO_TSP
        self._cpp = faco_opt.MFACO_CVRP(
            coords_np,
            demand_np,
            float(capacity),
            int(n_ants),
            int(cand_list_size),
            int(backup_list_size),
            int(min_new_edges),
            float(decay),
            float(alpha),
            float(p_best),
            bool(use_local_search),
            bool(disable_heuristic),
            bool(extend_ls),
            bool(smooth_mmas),
            int(fixed_steps),
            bool(nls),
            int(T_nls),
            _ls_scope_to_int(ls_scope),
            _ls_budget_to_int(ls_budget),
            int(ls_max_opt),
        )
        self.device = device
        self._enable_torch_sync = enable_torch_sync
        self.alpha = alpha
        self.disable_heuristic = disable_heuristic
        self.ls_scope = ls_scope
        self.ls_budget = ls_budget
        self.ls_max_opt = int(ls_max_opt)
        
        if normalized_heuristic and not disable_heuristic:
            h = np.asarray(self._cpp.heuristic_sparse_np)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)
        
        # Torch buffers (matching MFACO_TSP naming)
        self._init_torch_buffers(device)

    # Properties (matching MFACO_TSP naming)
    @property
    def n(self) -> int: return self._cpp.n
    @property
    def m(self) -> int: return self._cpp.m
    @property
    def k(self) -> int: return self._cpp.k
    @property
    def n_ants(self) -> int: return self._cpp.n_ants

    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)
    @property
    def backup_list(self) -> np.ndarray: return np.asarray(self._cpp.backup_list)
    
    @property
    def pheromone_sparse(self) -> torch.Tensor:
        return self._pheromone_sparse
    
    @property
    def h_sparse_torch(self) -> torch.Tensor:
        """Alias for heuristic tensor (matches MFACO_TSP API)."""
        return self._h_sparse_torch
    
    @property
    def nn_torch(self) -> torch.Tensor:
        return self._nn_torch

    @property
    def source_perm(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route)
    
    @property
    def source_route(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route)
    
    @property
    def enable_torch_sync(self) -> bool:
        return self._enable_torch_sync

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(int(seed))

    def sample(
        self,
        invtemp: float = 1.0,  # Kept for API compatibility (unused for CVRP)
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = False,
        return_decoded: bool = False,
    ):
        """
        Sample from C++ backend.
        
        Returns tuple matching MFACO_TSP:
            (costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival)
        
        Note: For CVRP, 'touched' contains decoded routes if return_decoded=True.
        """
        if prior is not None:
            prior = _prepare_prior(prior, self.n, self.k)
            if prior is not None and prior.ndim == 2 and prior.shape == (self.n, self.n):
                prior = _project_dense_prior_to_sparse(prior, self.n, self.nn_list)
        
        costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges_count, survival = self._cpp.sample(
            require_prob, prior, parallel_traced, return_decoded
        )

        if isinstance(survival, np.ndarray):
            survival = torch.from_numpy(survival).to(self.device)

        # Return format matches MFACO_TSP: (costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges, survival)
        return costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges_count, survival

    def sample_mixed_priors(
        self,
        priors,
        require_prob: bool = False,
        parallel_traced: bool = True,
        return_decoded: bool = False,
        head_counts=None,
    ):
        """Mixed-ant deployment for sparse CVRP head or per-ant priors."""
        prior_arr = _as_numpy(priors, np.float32)
        if prior_arr.ndim != 3:
            raise ValueError(f"priors must have shape (heads, n, k), got {prior_arr.shape}")
        if prior_arr.shape[1:] != (self.n, self.k):
            raise ValueError(f"priors trailing shape must be (n, k)=({self.n}, {self.k}), got {prior_arr.shape}")
        if prior_arr.shape[0] > self.n_ants:
            raise ValueError(
                f"number of heads must be <= n_ants ({self.n_ants}), got {prior_arr.shape[0]}"
            )
        counts_arr = None
        if head_counts is not None:
            counts_arr = np.asarray(head_counts, dtype=np.int32)
            if counts_arr.shape != (prior_arr.shape[0],):
                raise ValueError(
                    f"head_counts must have shape ({prior_arr.shape[0]},), got {counts_arr.shape}"
                )
            if int(counts_arr.sum()) != self.n_ants:
                raise ValueError(f"head_counts must sum to n_ants ({self.n_ants})")
        return self._cpp.sample_head_priors(
            require_prob, prior_arr, parallel_traced, return_decoded, counts_arr
        )

    def update_pheromone(self, best_route, best_cost: float) -> None:
        """Update pheromone with best route (matches MFACO_TSP naming)."""
        p = _as_numpy(best_route, np.int32)
        self._cpp.update_pheromone_from_route(p, float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def reset_timings(self) -> None:
        self._cpp.reset_timings()

    def get_timings(self) -> dict:
        return self._cpp.get_timings()

    def sync_pheromone_to_torch(self) -> None:
        _sync_pheromone_to_torch(self, self.device)

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        """Compute probability tensor (argument order matches MFACO_TSP)."""
        logit = _compute_mfaco_logits(self._pheromone_sparse, self._h_sparse_torch, self.alpha, invtemp, self.disable_heuristic)
        if prior is not None:
            logit = logit + prior
        return torch.exp(logit)

    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()
        


class ACO_CVRP:
    """
    Standard MMAS ACO CVRP solver.
    """
    def __init__(
        self,
        coords,
        demand,
        capacity: float,
        n_ants: int,
        cand_list_size: int = 0, # Default dense 
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        p_best: float = 0.05,
        min_max: bool = True,
        elitist: bool = False,
        use_local_search: bool = False,
        device: str = "cuda",
        enable_torch_sync: bool = True,
        **kwargs
    ):
        self.device = device
        
        # Handle PyG inputs
        if not isinstance(coords, torch.Tensor) and hasattr(coords, "x"):
            coords = coords.x
        
        coords_np = _as_numpy(coords, np.float32)
        demand_np = _as_numpy(demand, np.float32)
        
        self.capacity = float(capacity)
        
        self.n = coords_np.shape[0]
        # demand buffer
        self.demand_torch = torch.from_numpy(demand_np).to(device)

        self._cpp = faco_opt.ACO_CVRP(
            coords_np,
            demand_np,
            float(capacity),
            int(n_ants),
            int(cand_list_size),
            float(decay),
            float(alpha),
            float(beta),
            float(p_best),
            bool(min_max),
            bool(elitist),
            bool(use_local_search)
        )
        
        # Torch buffers
        _init_torch_sparse_buffers(self, device)
        self._enable_torch_sync = enable_torch_sync
        
        self.disable_heuristic = False

    # Delegate properties
    @property
    def n_ants(self) -> int: return self._cpp.n_ants
    @property
    def k(self) -> int: return self._cpp.k
    
    @property
    def best_cost(self) -> float: return self._cpp.best_cost
    @property
    def tau_min(self) -> float: return self._cpp.tau_min
    @property
    def tau_max(self) -> float: return self._cpp.tau_max
    
    @property
    def source_perm(self) -> np.ndarray: return np.asarray(self._cpp.source_perm)
    
    @property
    def best_route(self) -> np.ndarray: return self._cpp.best_route
    
    @property
    def pheromone_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.pheromone_sparse_np)
    
    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)

    @property 
    def h_sparse_torch(self) -> torch.Tensor: return self._h_sparse_torch
    @property
    def nn_torch(self) -> torch.Tensor: return self._nn_torch
    @property
    def pheromone_sparse(self) -> torch.Tensor: return self._pheromone_sparse

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(seed)

    def run(self, n_iterations: int) -> float:
        for _ in range(n_iterations):
            # Sample (costs, routes, ...)
            ret = self.sample(require_prob=False)
            costs = ret[0]
            routes = ret[1]

            # Find best in batch
            best_idx = np.argmin(costs)
            iteration_best_cost = float(costs[best_idx])
            iteration_best_route = routes[best_idx]

            # Update pheromone with iteration best; C++ handles global best tracking and pheromone deposition.
            self.update_pheromone(iteration_best_route, iteration_best_cost)
        
        return self.best_cost

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = True,
        return_decoded: bool = False,
    ):
        if prior is not None:
            prior = _prepare_prior(prior, self.n, self.k)
            # ACO_CVRP is dense; dense (n,n) prior not expected
            if prior is not None and prior.ndim == 2 and prior.shape == (self.n, self.n):
                raise AssertionError(f"Prior shape {prior.shape} is not compatible with ACO_CVRP (n={self.n}, k={self.k})")

        ret = self._cpp.sample(require_prob, prior, parallel_traced)
        
        if require_prob and self._enable_torch_sync:
            self.sync_pheromone_to_torch()
            
        # ret is (costs, routes, None, logps, None, None, None, None, None)
        # For compatibility, keep same format
        return ret

    def update_pheromone(self, best_flat: np.ndarray, best_cost: float) -> None:
        self._cpp.update_pheromone(best_flat.astype(np.int32), float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def sync_pheromone_to_torch(self) -> None:
        _sync_pheromone_to_torch(self, self.device)
        
    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        logit = _compute_aco_logits(self._pheromone_sparse, self._h_sparse_torch, self._cpp.alpha, self._cpp.beta)
        if prior is not None:
            logit = logit + prior
        return torch.exp(logit)

    def evaluate_log_prob_torch(self, paths: torch.Tensor, prior: torch.Tensor = None, invtemp: float = 1.0) -> torch.Tensor:
        """
        Evaluate log probability of CVRP paths (B, L) including depots.
        paths: (B, L) int64 tensor.
        """
        B, L = paths.shape
        device = paths.device
        N = self.n
        
        # 1. Compute logits (N, K) -> (B, N, K) or (N, K) shared
        # Since nn is dense, K=N.
        logits_base = torch.log(self.prob_sparse_torch(invtemp=invtemp, prior=prior) + 1e-20) # (N, N)
        
        # 2. Iterate
        visited = torch.zeros((B, N), dtype=torch.bool, device=device) # Customers visited
        visited[:, 0] = True # Depot usually 0
        
        current_capacity = torch.full((B,), self.capacity, device=device)
        
        log_prob_sum = torch.zeros(B, device=device)
        
        prev = paths[:, 0]
        
        for t in range(L - 1):
            curr = prev
            next_node = paths[:, t+1]
            
            # Mask logits[curr] (B, N)
            lg = logits_base[curr]
            
            # Mask visited
            mask_visited = visited.clone()
            mask_visited[:, 0] = False
            
            # Capacity constraint
            demand_ok = (self.demand_torch.unsqueeze(0) <= current_capacity.unsqueeze(1))
            
            cust_cand = (~mask_visited) & demand_ok
            cust_cand[:, 0] = False # Customers only
            
            has_candidates = cust_cand.any(dim=1)
            
            # Construct allowed mask
            allowed = cust_cand.clone()
            
            # Depot allowed logic: Allowed if curr != 0
            is_at_depot = (curr == 0)
            allowed[:, 0] = (~is_at_depot)
            
            # Apply mask
            lg_masked = lg.clone()
            lg_masked[~allowed] = float('-inf')
            
            log_denom = torch.logsumexp(lg_masked, dim=1)
            
            # Numerator
            log_numer = lg.gather(1, next_node.unsqueeze(1)).squeeze(1)
            
            step_logp = log_numer - log_denom
            
            # Zero out if step invalid (done/padded)
            valid_step = allowed.any(dim=1)
            step_logp = torch.where(valid_step, step_logp, torch.zeros_like(step_logp))
            
            log_prob_sum += step_logp
            
            # Update state
            is_depot_move = (next_node == 0)
            
            visited.scatter_(1, next_node.unsqueeze(1), True)
            visited[:, 0] = False
            
            dem = self.demand_torch[next_node]
            current_capacity = torch.where(is_depot_move, 
                                           torch.tensor(self.capacity, device=device),
                                           current_capacity - dem)
            
            prev = next_node
            
        return log_prob_sum


# =============================================================================
# Factory Function
# =============================================================================

def get_aco(problem: str, **kwargs):
    """
    Get the appropriate ACO class for a given problem.

    Args:
        problem: Problem type ('tsp', 'cvrp', 'bpp', 'mkp', 'op')
        **kwargs: Additional arguments for ACO initialization

    Returns:
        ACO instance
    """
    if problem in ('tsp', 'cvrp'):
        # Use existing MFACO classes
        if problem == 'tsp':
            return MFACO_TSP(**kwargs)
        else:
            return MFACO_CVRP(**kwargs)
    elif problem == 'bpp':
        return ACO_BPP(**kwargs)
    elif problem == 'mkp':
        return ACO_MKP(**kwargs)
    elif problem == 'op':
        return ACO_OP(**kwargs)
    else:
        raise ValueError(f"Unknown problem: {problem}")


if __name__ == "__main__":
    # Test unified ACO creation
    print("Testing unified ACO solvers...")

    print("\nTesting TSP...")
    coords = torch.rand(10, 2)
    aco_tsp = MFACO_TSP(coords, n_ants=5)
    print(f"MFACO_TSP: n={aco_tsp.n}, n_ants={aco_tsp.n_ants}")

    print("\nTesting CVRP...")
    demand = torch.rand(11)
    aco_cvrp = MFACO_CVRP(coords, demand, capacity=50, n_ants=5)
    print(f"MFACO_CVRP: n={aco_cvrp.n}, n_ants={aco_cvrp.n_ants}")

    print("\nTesting BPP...")
    demand_bpp = torch.rand(11)
    aco_bpp = ACO_BPP(demand_bpp, n_ants=5)
    print(f"  ACO_BPP: n={aco_bpp.n}, n_ants={aco_bpp.n_ants}")

    print("\nTesting MKP...")
    prize = torch.rand(10)
    weight = torch.rand(10, 5)
    aco_mkp = ACO_MKP(prize, weight, n_ants=5)
    print(f"  ACO_MKP: n={aco_mkp.n}, n_ants={aco_mkp.n_ants}")

    print("\nTesting OP...")
    distances = torch.rand(10, 10)
    prizes = torch.rand(10)
    aco_op = ACO_OP(distances, prizes, max_len=4.0, n_ants=5)
    print(f"  ACO_OP: n={aco_op.n}, n_ants={aco_op.n_ants}")

    print("\nTesting factory function...")
    aco_tsp_f = get_aco('tsp', coords=coords, n_ants=5)
    aco_cvrp_f = get_aco('cvrp', coords=coords, demand=demand, capacity=50, n_ants=5)
    aco_bpp_f = get_aco('bpp', demand=demand_bpp, n_ants=5)
    aco_mkp_f = get_aco('mkp', prize=prize, weight=weight, n_ants=5)
    aco_op_f = get_aco('op', distances=distances, prizes=prizes, max_len=4.0, n_ants=5)
    print(f"Factory: TSP={type(aco_tsp_f).__name__}, BPP={type(aco_bpp_f).__name__}, MKP={type(aco_mkp_f).__name__}, OP={type(aco_op_f).__name__}")

    print("\nAll ACO tests passed!")
