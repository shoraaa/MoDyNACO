import unittest
from argparse import Namespace
from unittest import mock

import torch
from torch_geometric.data import Data

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

    def test_robust_capacity_uses_domain_randomized_generator(self):
        args = Namespace(
            problem="cvrp",
            n_node=1000,
            capacity_override=None,
            robust_capacity=True,
        )

        capacity = train.resolve_cvrp_generation_capacity(args)

        self.assertIsNone(capacity)

    def test_robust_capacity_rejects_non_1k_cvrp(self):
        args = Namespace(
            problem="cvrp",
            n_node=500,
            capacity_override=None,
            robust_capacity=True,
        )

        with self.assertRaisesRegex(ValueError, "n_node=1000"):
            train.resolve_cvrp_generation_capacity(args)

    def test_robust_cvrp_instance_randomizes_raw_demand_and_capacity(self):
        raw_demand = torch.arange(1, 1001, dtype=torch.float32).remainder(80).add(1)

        with (
            mock.patch.object(train, "_sample_robust_cvrp_raw_demands", return_value=raw_demand),
            mock.patch.object(train.random, "choice", return_value=10.0),
        ):
            coords, demand, capacity = train.gen_robust_cvrp_1k_instance(1000, "cpu")

        self.assertEqual(coords.shape, (1001, 2))
        self.assertEqual(demand.shape, (1001,))
        self.assertEqual(capacity, 1.0)
        self.assertEqual(float(demand[0]), 0.0)
        self.assertTrue(torch.all(demand[1:] > 0))
        self.assertLessEqual(float(demand[1:].max()), 1.0)
        self.assertGreater(float(demand[1:].max()), float(demand[1:].min()))

    def test_train_epoch_uses_robust_generator_instead_of_default_cvrp_generator(self):
        args = Namespace(
            problem="cvrp",
            n_node=1000,
            device="cpu",
            capacity_override=None,
            robust_capacity=True,
            steps_per_epoch=2,
            algo="ppo",
        )
        coords = torch.zeros(1001, 2)
        demand = torch.full((1001,), 0.2)
        captured_instances = []

        def capture_train_instance(_net, _optimizer, instance_data, _args):
            captured_instances.append(instance_data)
            return 1.0, 0.5, {}

        with (
            mock.patch.object(train, "gen_robust_cvrp_1k_instance", return_value=(coords, demand, 1.0)) as robust_mock,
            mock.patch.object(utils, "gen_cvrp_instance") as gen_mock,
            mock.patch.object(train, "train_instance_ppo", side_effect=capture_train_instance),
            mock.patch.object(train, "get_logger") as logger_mock,
        ):
            logger_mock.return_value.set_step.return_value = None
            logger_mock.return_value.log_train_step.return_value = None
            train.train_epoch(mock.Mock(), mock.Mock(), 0, 1, args)

        self.assertEqual(robust_mock.call_count, 2)
        gen_mock.assert_not_called()
        self.assertEqual(len(captured_instances), 2)
        self.assertTrue((captured_instances[0][1] == 0.2).all())

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

    def test_multi_head_jensen_diversity_is_zero_for_identical_heads(self):
        priors = torch.tensor(
            [
                [[2.0, -2.0], [0.5, -0.5]],
                [[2.0, -2.0], [0.5, -0.5]],
            ]
        )

        diversity = train._multi_head_jensen_diversity(priors)

        self.assertAlmostEqual(diversity.item(), 0.0, places=6)

    def test_loss_js_rewards_diverse_polynet_heads(self):
        args = Namespace(loss_js=0.5)
        priors = torch.tensor(
            [
                [[4.0, -4.0], [-4.0, 4.0]],
                [[-4.0, 4.0], [4.0, -4.0]],
            ],
            requires_grad=True,
        )

        diversity = train._multi_head_jensen_diversity(priors)
        loss_term = train._multi_head_js_loss_term(priors, args)

        self.assertGreater(diversity.item(), 0.0)
        self.assertLess(loss_term.item(), 0.0)
        loss_term.backward()
        self.assertIsNotNone(priors.grad)

    def test_multi_head_ppo_loss_uses_polynet_best_mean_head(self):
        args = Namespace(
            n_ants=4,
            alpha=1.0,
            beta=1.0,
            ppo_clip=0.2,
            no_adv_norm=True,
            head_loss_mode="winner",
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
            "replay_logp_from_cpp_batch_trace_ant_slice",
            return_value=(torch.zeros(2), torch.ones(2)),
        ) as replay_mock:
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

        replay_mock.assert_called_once()
        self.assertEqual(replay_mock.call_args.args[2:], (0, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(approx_kl.item(), 0.0)
        self.assertAlmostEqual(clip_frac.item(), 0.0)

    def test_multi_head_ppo_loss_can_average_all_heads(self):
        args = Namespace(
            n_ants=4,
            alpha=1.0,
            beta=1.0,
            ppo_clip=0.2,
            no_adv_norm=True,
            head_loss_mode="mean",
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
        replay_returns = [
            (torch.tensor([0.3, 0.1]), torch.ones(2)),
            (torch.tensor([0.2, 0.4]), torch.ones(2)),
        ]

        with mock.patch.object(
            train,
            "replay_logp_from_cpp_batch_trace_ant_slice",
            side_effect=replay_returns,
        ) as replay_mock:
            loss, approx_kl, clip_frac = train._multi_head_ppo_loss(
                current_prior,
                tau_nk,
                eta_nk,
                traces=object(),
                head_counts=[2, 2],
                costs_by_head=costs_by_head,
                logp_old_by_head=logp_old_by_head,
                args=args,
                selected_head=0,
            )

        self.assertEqual([call.args[2:] for call in replay_mock.call_args_list], [(0, 2), (2, 4)])
        baseline = torch.tensor([(1.0 + 2.0) / 2, (10.0 + 12.0) / 2]).mean()
        expected_0, expected_kl_0, expected_clip_0 = train._ppo_clipped_loss(
            replay_returns[0][0],
            logp_old_by_head[0],
            costs_by_head[0],
            args,
            baseline=baseline,
        )
        expected_1, expected_kl_1, expected_clip_1 = train._ppo_clipped_loss(
            replay_returns[1][0],
            logp_old_by_head[1],
            costs_by_head[1],
            args,
            baseline=baseline,
        )
        self.assertAlmostEqual(loss.item(), torch.stack([expected_0, expected_1]).mean().item())
        self.assertAlmostEqual(approx_kl.item(), torch.stack([expected_kl_0, expected_kl_1]).mean().item())
        self.assertAlmostEqual(clip_frac.item(), torch.stack([expected_clip_0, expected_clip_1]).mean().item())

    def test_polynet_multi_head_loss_updates_only_best_head(self):
        args = Namespace(
            n_ants=4,
            alpha=1.0,
            beta=1.0,
            ppo_clip=0.2,
            no_adv_norm=True,
            head_loss_mode="winner",
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
            "replay_logp_from_cpp_batch_trace_ant_slice",
            return_value=(replayed_logp[:2], torch.ones(2)),
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
                selected_head=0,
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

    def test_infer_instance_preserves_learned_head_counts(self):
        class DummyModel:
            def eval(self):
                return self

            def forward_with_alloc(self, pyg_data):
                num_edges = pyg_data.edge_attr.shape[0]
                priors = torch.zeros(num_edges, 2)
                logits = torch.tensor([10.0, -10.0])
                return priors, logits

        class DummyACO:
            recorded_head_counts = None
            recorded_prior_scale = None

            def __init__(self, **kwargs):
                self.n = 3
                self.k = 2
                self.n_ants = int(kwargs["n_ants"])
                self.nn_torch = torch.tensor([[1, 2], [0, 2], [0, 1]])
                self.pheromone_sparse = torch.ones(self.n, self.k)
                self.source_route = torch.tensor([0, 1, 0, 2, 0])

            def seed_rng(self, seed):
                self.seed = seed

            def reset_timings(self):
                pass

            def sample_mixed_priors(
                self,
                priors,
                require_prob=False,
                parallel_traced=True,
                return_decoded=False,
                head_counts=None,
                prior_scale=1.0,
            ):
                DummyACO.recorded_head_counts = list(head_counts)
                DummyACO.recorded_prior_scale = float(prior_scale)
                costs = torch.arange(1, self.n_ants + 1, dtype=torch.float32)
                routes = [torch.tensor([0, 1, 0, 2, 0]) for _ in range(self.n_ants)]
                survival = torch.ones(self.n_ants)
                return costs, routes, None, None, None, None, None, 0, survival

            def update_pheromone(self, best_route, best_cost):
                pass

        def build_fn(aco, coords, demand, device, **kwargs):
            return Data(
                x=torch.zeros(3, 4),
                edge_index=torch.zeros(2, 6, dtype=torch.long),
                edge_attr=torch.zeros(6, 3),
            )

        args = Namespace(
            disable_heuristic=False,
            no_local_search=True,
            rho=0.5,
            device="cpu",
            no_smooth_mmas=True,
            min_new_edges=1,
            no_extend_ls=True,
            no_normalized_heuristic=True,
            L=0,
            ls_scope="localized",
            ls_budget="truncated",
            ls_max_opt=0,
            euc_2d_cost=False,
            no_dynamic_feats=False,
            head_router="learned",
            multi_head=True,
            num_heads=2,
            n_ants=8,
            head_router_min_frac=0.0,
            allocator_temperature=1.0,
            head_ant_weights=None,
            head_input_transform="none",
            edge_feature_set="compact3",
            no_anneal=True,
            mini_H=1,
            H=1,
            gamma=1.0,
            min_gamma=0.0,
            iter_log=False,
            iter_print=False,
            stage_metrics=False,
            runtime_limit=None,
            verify=False,
            verify_final_only=False,
            timed=False,
        )
        coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        demand = torch.tensor([0.0, 0.5, 0.5])

        utils.infer_instance(
            "cvrp",
            DummyACO,
            build_fn,
            DummyModel(),
            (coords, demand, 1.0),
            k_sparse=2,
            n_ants=8,
            dynamic=True,
            args=args,
            seed=1234,
        )

        self.assertEqual(DummyACO.recorded_head_counts, [7, 1])
        self.assertIsNotNone(DummyACO.recorded_prior_scale)

    def test_iter_stats_include_outer_step_time(self):
        class DummyACO:
            def __init__(self, **kwargs):
                self.n = 3
                self.k = 2
                self.n_ants = int(kwargs["n_ants"])
                self.nn_torch = torch.tensor([[1, 2], [0, 2], [0, 1]])
                self.pheromone_sparse = torch.ones(self.n, self.k)

            def seed_rng(self, seed):
                self.seed = seed

            def reset_timings(self):
                pass

            def sample(self, require_prob=False, prior=None, parallel_traced=True):
                costs = torch.arange(1, self.n_ants + 1, dtype=torch.float32)
                flats = [torch.tensor([0, 1, 2, 0]) for _ in range(self.n_ants)]
                survival = torch.ones(self.n_ants)
                return costs, flats, None, None, None, None, None, 0, survival

            def _update_pheromone_from_flat(self, best_route, best_cost):
                pass

        def build_fn(aco, coords, device, **kwargs):
            return Data(
                x=torch.zeros(3, 4),
                edge_index=torch.zeros(2, 6, dtype=torch.long),
                edge_attr=torch.zeros(6, 3),
            )

        args = Namespace(
            disable_heuristic=False,
            no_local_search=True,
            rho=0.5,
            device="cpu",
            no_smooth_mmas=True,
            min_new_edges=1,
            no_extend_ls=True,
            no_normalized_heuristic=True,
            L=0,
            ls_scope="localized",
            ls_budget="truncated",
            ls_max_opt=0,
            euc_2d_cost=False,
            no_dynamic_feats=False,
            head_router="static",
            multi_head=False,
            num_heads=1,
            n_ants=4,
            head_router_min_frac=0.0,
            allocator_temperature=1.0,
            head_ant_weights=None,
            head_input_transform="none",
            edge_feature_set="compact3",
            no_anneal=True,
            mini_H=2,
            H=2,
            gamma=1.0,
            min_gamma=0.0,
            iter_log=True,
            iter_print=False,
            stage_metrics=False,
            runtime_limit=None,
            verify=False,
            verify_final_only=False,
            timed=False,
        )

        _, _, _, extra = utils.infer_instance(
            "tsp",
            DummyACO,
            build_fn,
            model=None,
            instance_data=torch.zeros(3, 2),
            k_sparse=2,
            n_ants=4,
            dynamic=True,
            args=args,
            use_heuristic_only=True,
            seed=1234,
        )

        iter_stats = extra["iter_stats"]
        self.assertEqual(len(iter_stats), 4)
        self.assertEqual([row["outer_step_done"] for row in iter_stats], [0, 1, 0, 1])
        self.assertTrue(all(row["elapsed_s"] >= 0.0 for row in iter_stats))
        self.assertTrue(all(row["outer_elapsed_s"] >= 0.0 for row in iter_stats))
        self.assertLessEqual(iter_stats[0]["outer_elapsed_s"], iter_stats[1]["outer_elapsed_s"])
        self.assertLessEqual(iter_stats[2]["outer_elapsed_s"], iter_stats[3]["outer_elapsed_s"])


if __name__ == "__main__":
    unittest.main()
