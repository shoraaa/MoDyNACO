import unittest

import torch

from test_cvrp import compute_subroute_stats, split_route0


class CVRPCapacitySweepTests(unittest.TestCase):
    def test_split_route0_separates_depot_delimited_routes(self):
        route0 = [0, 1, 2, 0, 3, 4, 0]
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


if __name__ == "__main__":
    unittest.main()
