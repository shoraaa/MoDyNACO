import unittest
from argparse import Namespace
from unittest import mock

import torch

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

    def test_ppo_clipped_loss_rewards_better_than_average_cost(self):
        args = Namespace(ppo_clip=0.1, no_adv_norm=True)
        logp_old = torch.zeros(2)
        logp_new = torch.tensor([0.2, 0.0], requires_grad=True)
        costs = torch.tensor([1.0, 3.0])

        loss, approx_kl, clip_frac = train._ppo_clipped_loss(
            logp_new, logp_old, costs, args
        )

        self.assertLess(loss.item(), 0.0)
        self.assertGreater(approx_kl.item(), 0.0)
        self.assertAlmostEqual(clip_frac.item(), 0.5)

    def test_multi_head_ppo_loss_applies_head_competition_weights(self):
        args = Namespace(
            n_ants=4,
            alpha=1.0,
            beta=1.0,
            ppo_clip=0.2,
            no_adv_norm=True,
            head_score_mode="mean",
            head_gamma=2.0,
        )
        current_prior = torch.zeros(2, 1, 2, requires_grad=True)
        tau_nk = torch.ones(1, 2)
        eta_nk = torch.ones(1, 2)
        costs_by_head = [
            torch.tensor([1.0, 1.5]),
            torch.tensor([5.0, 6.0]),
        ]
        logp_old_by_head = [
            torch.zeros(2),
            torch.zeros(2),
        ]

        with mock.patch.object(
            train,
            "replay_logp_from_cpp_batch_trace_ant_priors",
            return_value=(torch.zeros(4), torch.ones(4)),
        ):
            loss, approx_kl, clip_frac = train._multi_head_ppo_loss(
                current_prior,
                tau_nk,
                eta_nk,
                traces=object(),
                head_counts=[2, 2],
                costs_by_head=costs_by_head,
                logp_old_by_head=logp_old_by_head,
                args=args,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(approx_kl.item(), 0.0)
        self.assertAlmostEqual(clip_frac.item(), 0.0)

    def test_polynet_multi_head_loss_updates_only_best_head(self):
        args = Namespace(
            n_ants=4,
            alpha=1.0,
            beta=1.0,
            ppo_clip=0.2,
            no_adv_norm=True,
            head_score_mode="mean",
            head_gamma=2.0,
            head_training="polynet",
        )
        current_prior = torch.zeros(2, 1, 2, requires_grad=True)
        tau_nk = torch.ones(1, 2)
        eta_nk = torch.ones(1, 2)
        costs_by_head = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([10.0, 12.0]),
        ]
        logp_old_by_head = [
            torch.zeros(2),
            torch.zeros(2),
        ]
        replayed_logp = torch.tensor([0.3, 0.1, 5.0, 5.0])

        with mock.patch.object(
            train,
            "replay_logp_from_cpp_batch_trace_ant_priors",
            return_value=(replayed_logp, torch.ones(4)),
        ):
            loss, approx_kl, clip_frac = train._multi_head_ppo_loss(
                current_prior,
                tau_nk,
                eta_nk,
                traces=object(),
                head_counts=[2, 2],
                costs_by_head=costs_by_head,
                logp_old_by_head=logp_old_by_head,
                args=args,
            )

        expected, expected_kl, expected_clip = train._ppo_clipped_loss(
            replayed_logp[:2],
            logp_old_by_head[0],
            costs_by_head[0],
            args,
            baseline=torch.tensor([(1.0 + 2.0) / 2, (10.0 + 12.0) / 2]).mean(),
        )
        self.assertAlmostEqual(loss.item(), expected.item())
        self.assertAlmostEqual(approx_kl.item(), expected_kl.item())
        self.assertAlmostEqual(clip_frac.item(), expected_clip.item())


if __name__ == "__main__":
    unittest.main()
