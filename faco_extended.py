#!/usr/bin/env python3
"""
MFACO Extended - Learning-guided ACO for BPP, MKP, OP

Provides learning-guided ACO classes for Bin Packing Problem (BPP),
Multi-dimensional Knapsack Problem (MKP), and Orienteering Problem (OP).
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Any

from cpp_aco_wrapper import (
    load_alphaant_cpp_module,
    to_numpy_matrix,
    to_numpy_vector,
    trace_paths_to_tensor,
    fresh_seed,
)


EPS = 1e-10


def _as_float_tensor(value: Optional[Any], device: str) -> Optional[torch.Tensor]:
    """Convert optional numpy/torch inputs to float tensors on the target device."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


# =============================================================================
# BPP (Bin Packing Problem)
# =============================================================================

class MFACO_BPP:
    """
    Learning-guided ACO for Bin Packing Problem.

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
            touched: Number of items touched
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


# =============================================================================
# MKP (Multi-dimensional Knapsack Problem)
# =============================================================================

class MFACO_MKP:
    """
    Learning-guided ACO for Multi-dimensional Knapsack Problem.

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


# =============================================================================
# OP (Orienteering Problem)
# =============================================================================

class MFACO_OP:
    """
    Learning-guided ACO for Orienteering Problem.

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


# =============================================================================
# Factory Function
# =============================================================================

def get_aco(problem: str, **kwargs):
    """
    Get the appropriate ACO class for a given problem.

    Args:
        problem: Problem type ('bpp', 'mkp', 'op')
        **kwargs: Additional arguments for ACO initialization

    Returns:
        ACO instance
    """
    if problem == 'bpp':
        return MFACO_BPP(**kwargs)
    elif problem == 'mkp':
        return MFACO_MKP(**kwargs)
    elif problem == 'op':
        return MFACO_OP(**kwargs)
    else:
        raise ValueError(f"Unknown problem: {problem}")


if __name__ == "__main__":
    # Test ACO creation
    print("Testing BPP ACO...")
    demand = torch.rand(21)
    aco_bpp = MFACO_BPP(demand, n_ants=10)
    print(f"MFACO_BPP: {aco_bpp}")

    print("\nTesting MKP ACO...")
    prize = torch.rand(20)
    weight = torch.rand(20, 5)
    aco_mkp = MFACO_MKP(prize, weight, n_ants=10)
    print(f"MFACO_MKP: {aco_mkp}")

    print("\nTesting OP ACO...")
    distances = torch.rand(20, 20)
    prizes = torch.rand(20)
    aco_op = MFACO_OP(distances, prizes, max_len=4.0, n_ants=10)
    print(f"MFACO_OP: {aco_op}")

    print("\nTesting factory function...")
    aco_bpp_factory = get_aco('bpp', demand=demand, n_ants=10)
    aco_mkp_factory = get_aco('mkp', prize=prize, weight=weight, n_ants=10)
    aco_op_factory = get_aco('op', distances=distances, prizes=prizes, max_len=4.0, n_ants=10)
    print(f"Factory BPP: {aco_bpp_factory}")
    print(f"Factory MKP: {aco_mkp_factory}")
    print(f"Factory OP: {aco_op_factory}")

    print("\nAll tests passed!")
