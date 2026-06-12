import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from tests.test_cvrp import compute_subroute_stats, resolve_inscale_checkpoint, split_route0


class CVRPCapacitySweepTests(unittest.TestCase):
    def test_split_route0_separates_depot_delimited_routes(self):
        route0 = [0, 1, 2, 0, 3, 4, 0]
        self.assertEqual(split_route0(route0), [[1, 2], [3, 4]])

    def test_split_route0_handles_empty_route(self):
        route0 = []
        self.assertEqual(split_route0(route0), [])

    def test_split_route0_handles_only_depot(self):
        route0 = [0, 0]
        self.assertEqual(split_route0(route0), [])

    def test_split_route0_handles_single_customer(self):
        route0 = [0, 1, 0]
        self.assertEqual(split_route0(route0), [[1]])

    def test_split_route0_accepts_numpy_array(self):
        route0 = np.array([0, 1, 2, 0, 3, 4, 0])
        self.assertEqual(split_route0(route0), [[1, 2], [3, 4]])

    def test_split_route0_accepts_torch_tensor(self):
        route0 = torch.tensor([0, 1, 2, 0, 3, 4, 0])
        self.assertEqual(split_route0(route0), [[1, 2], [3, 4]])

    def test_compute_subroute_stats_reports_route_means(self):
        coords = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        customer_demands = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float32)
        route0 = [0, 1, 2, 0, 3, 0]

        stats = compute_subroute_stats(coords, customer_demands, capacity=10.0, route0=route0)

        self.assertEqual(stats["route_count"], 2.0)
        self.assertAlmostEqual(stats["customers_per_subroute"], 1.5, places=6)
        self.assertAlmostEqual(stats["subroute_cost"], 3.0, places=6)
        self.assertAlmostEqual(stats["subroute_load"], 3.5, places=6)
        self.assertAlmostEqual(stats["subroute_utilization"], 0.35, places=6)

    def test_compute_subroute_stats_handles_empty_routes(self):
        coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        demands = torch.tensor([0.0, 0.0], dtype=torch.float32)  # depot + 1 customer
        route0 = [0, 0]  # no actual subroutes

        stats = compute_subroute_stats(coords, demands, capacity=10.0, route0=route0)

        self.assertEqual(stats["route_count"], 0.0)
        self.assertEqual(stats["customers_per_subroute"], 0.0)
        self.assertEqual(stats["subroute_cost"], 0.0)
        self.assertEqual(stats["subroute_load"], 0.0)
        self.assertEqual(stats["subroute_utilization"], 0.0)

    def test_compute_subroute_stats_accepts_numpy_inputs(self):
        # coords: depot (0) + 2 customers (1, 2)
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        # customer_demands: only customers, indexed by customer_id-1
        # customer 1 -> index 0 = 2.0, customer 2 -> index 1 = 3.0
        demands = np.array([2.0, 3.0], dtype=np.float32)
        route0 = [0, 1, 2, 0]

        stats = compute_subroute_stats(coords, demands, capacity=10.0, route0=route0)

        self.assertEqual(stats["route_count"], 1.0)
        # route [0,1,2,0]: cost = |0-1| + |1-2| + |2-0| = 1 + 1 + 2 = 4
        self.assertAlmostEqual(stats["subroute_cost"], 4.0, places=6)
        # load = 2.0 + 3.0 = 5.0
        self.assertAlmostEqual(stats["subroute_load"], 5.0, places=6)
        self.assertAlmostEqual(stats["subroute_utilization"], 0.5, places=6)
        self.assertAlmostEqual(stats["customers_per_subroute"], 2.0, places=6)

    def test_resolve_inscale_checkpoint_picks_capacity_specific_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "models" / "cvrp" / "n1000"
            root.mkdir(parents=True)
            best = root / "cvrp_n1000_cap250_nosmooth_best.pt"
            best.write_text("x", encoding="utf-8")

            found = resolve_inscale_checkpoint(1000, 250.0, checkpoint_root=Path(tmpdir) / "models" / "cvrp")

            self.assertEqual(found, best)

    def test_resolve_inscale_checkpoint_can_prefer_smooth_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "models" / "cvrp" / "n1000"
            root.mkdir(parents=True)
            smooth = root / "cvrp_n1000_cap250_best.pt"
            nosmooth = root / "cvrp_n1000_cap250_nosmooth_best.pt"
            smooth.write_text("smooth", encoding="utf-8")
            nosmooth.write_text("nosmooth", encoding="utf-8")

            found = resolve_inscale_checkpoint(
                1000,
                250.0,
                no_smooth_mmas=False,
                checkpoint_root=Path(tmpdir) / "models" / "cvrp",
            )

            self.assertEqual(found, smooth)


if __name__ == "__main__":
    unittest.main()
