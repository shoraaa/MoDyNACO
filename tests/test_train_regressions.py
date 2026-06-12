import unittest
from unittest import mock

import train
import utils


class TrainRegressionTests(unittest.TestCase):
    def test_cuda_memory_helper_skips_cpu_device(self):
        with (
            mock.patch.object(train.torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                train.torch.cuda,
                "memory_allocated",
                side_effect=AssertionError("memory_allocated should not be called for CPU devices"),
            ),
        ):
            self.assertIsNone(train._gpu_memory_allocated_gb("cpu"))

    def test_generate_dataset_handles_missing_lkh(self):
        with mock.patch("baselines.solve_with_lkh", return_value=None):
            dataset = utils.generate_and_save_dataset(
                problem="tsp",
                n_node=5,
                n_instances=1,
                save_path=None,
                baseline_solver="lkh",
                baseline_runs=1,
                time_limit=0.01,
                device="cpu",
            )

        self.assertEqual(len(dataset), 1)
        coords, cost, tour, name = dataset[0]
        self.assertEqual(coords.shape, (5, 2))
        self.assertIsNone(cost)
        self.assertIsNone(tour)
        self.assertEqual(name, "Gen_0")

    def test_generate_dataset_accepts_scalar_lkh_cost(self):
        with mock.patch("baselines.solve_with_lkh", return_value=12.34):
            dataset = utils.generate_and_save_dataset(
                problem="tsp",
                n_node=5,
                n_instances=1,
                save_path=None,
                baseline_solver="lkh",
                baseline_runs=1,
                time_limit=0.01,
                device="cpu",
            )

        self.assertEqual(len(dataset), 1)
        _, cost, tour, _ = dataset[0]
        self.assertAlmostEqual(cost, 12.34)
        self.assertIsNone(tour)


if __name__ == "__main__":
    unittest.main()
