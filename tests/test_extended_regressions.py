import unittest
import tempfile
from pathlib import Path

import torch

import faco
import test as test_script
import train
import utils
from extended_common import create_model, prior_output_to_heuristic
from net import NetBPP, NetMKP, NetOP


class ExtendedRegressionTests(unittest.TestCase):
    def test_bpp_pheromone_update_uses_full_sample_batch(self):
        demand = utils.gen_bpp_instance(12, "cpu")
        aco = faco.ACO_BPP(demand=demand, n_ants=6, device="cpu")

        costs, paths, _, _ = aco.sample(require_prob=False, parallel_traced=True)

        self.assertEqual(paths.ndim, 2)
        self.assertEqual(paths.shape[1], aco.n_ants)
        self.assertEqual(costs.shape[0], aco.n_ants)

        aco.update_pheromone(paths, costs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-device regression coverage")
    def test_bpp_model_moves_cpu_graph_to_cuda(self):
        model = NetBPP(feats=1, edge_feats=1).to("cuda")
        demand = utils.gen_bpp_instance(8, "cpu")
        class MockACO:
            def __init__(self, n):
                self.pheromone = torch.ones((n, n))
        pyg_data = utils.build_pyg_data_bpp(demand, MockACO(9), "cpu", dynamic=False)

        out = model(pyg_data)

        self.assertEqual(out.device.type, "cuda")

    def test_bpp_dynamic_toggle_changes_extra_edge_features(self):
        demand = utils.gen_bpp_instance(4, "cpu")
        n = len(demand)
        pheromone = torch.arange(n*n, dtype=torch.float32).view(n, n) + 1.0
        class MockACO:
            def __init__(self, pher):
                self.pheromone = pher

        mock_aco = MockACO(pheromone)
        dynamic_graph = utils.build_pyg_data_bpp(
            demand, mock_aco, "cpu", dynamic=True
        )
        static_graph = utils.build_pyg_data_bpp(
            demand, mock_aco, "cpu", dynamic=False
        )

        # Dynamic mode: 1 base + 2 dynamic (pheromone, incumbent) = 3 total
        self.assertEqual(dynamic_graph.edge_attr.shape[1], 3)
        # Static mode: only 1 base feature
        self.assertEqual(static_graph.edge_attr.shape[1], 1)
        # Check that at least some dynamic features are non-zero (pheromone should always be)
        self.assertGreater(dynamic_graph.edge_attr[:, 1:].abs().sum().item(), 0.0)

    def test_bpp_demand_is_not_overwritten_by_neural_heuristic(self):
        demand = utils.gen_bpp_instance(6, "cpu")
        aco = faco.ACO_BPP(demand=demand, n_ants=4, device="cpu")
        original_demand = aco.demand.detach().clone()

        aco.heuristic = torch.rand_like(aco.heuristic)

        self.assertTrue(torch.equal(aco.demand, original_demand))

    def test_mkp_builder_matches_model_feature_count(self):
        n = 4
        m = 5
        prize, weight = utils.gen_mkp_instance(n, m, "cpu")
        class MockACO:
            def __init__(self, n_nodes):
                self.pheromone = torch.ones((n_nodes, n_nodes))
        mock_aco = MockACO(n + 1)
        pyg_data = utils.build_pyg_data_mkp(prize, weight, mock_aco, "cpu", dynamic=False)
        model = NetMKP(m=m, feats=m+1, edge_feats=1)

        self.assertEqual(pyg_data.x.shape[1], m+1)
        self.assertEqual(pyg_data.x.shape[0], prize.shape[0] + 1)
        self.assertEqual(pyg_data.edge_attr.shape[0], (prize.shape[0] + 1) ** 2)
        self.assertEqual(pyg_data.edge_attr.shape[1], 1)

        out = model(pyg_data)

        self.assertEqual(out.shape[0], pyg_data.edge_attr.shape[0])

    def test_op_dense_builder_creates_directed_graph(self):
        n = 4
        coords, prizes = utils.gen_op_instance(n, "cpu")
        distances = torch.cdist(coords, coords)
        class MockACO:
            def __init__(self, n_nodes):
                self.pheromone = torch.ones((n_nodes, n_nodes))
        mock_aco = MockACO(n + 1)
        pyg_data = utils.build_pyg_data_op(distances, prizes, mock_aco, "cpu", dynamic=False)
        model = NetOP(feats=2, edge_feats=1)

        self.assertEqual(pyg_data.x.shape[0], coords.shape[0] + 1)
        self.assertEqual(pyg_data.edge_index.shape[1], (coords.shape[0] + 1) ** 2)
        self.assertEqual(pyg_data.edge_attr.shape[1], 1)

        out = model(pyg_data)

        self.assertEqual(out.shape[0], pyg_data.edge_attr.shape[0])

    def test_single_head_extended_factory_forwards_logit_net(self):
        logit_model = create_model("bpp", edge_feats=1, logit_net=True)
        sigmoid_model = create_model("bpp", edge_feats=1, logit_net=False)

        self.assertFalse(logit_model.par_net_heu.sigmoid_output)
        self.assertTrue(sigmoid_model.par_net_heu.sigmoid_output)

    def test_extended_checkpoint_kwargs_legacy_path_is_explicit(self):
        kwargs, source = test_script._extended_model_kwargs_from_checkpoint({"config": {"problem": "bpp"}})

        self.assertEqual(source, "legacy_sigmoid_no_model_kwargs")
        self.assertEqual(kwargs["logit_net"], False)

    def test_extended_checkpoint_kwargs_prefer_checkpoint_metadata(self):
        kwargs, source = test_script._extended_model_kwargs_from_checkpoint(
            {"model_kwargs": {"logit_net": True, "grad_checkpointing": False}}
        )

        self.assertEqual(source, "checkpoint_model_kwargs")
        self.assertEqual(kwargs["logit_net"], True)

    def test_prior_output_modulates_base_heuristic_with_positive_weights(self):
        prior = torch.tensor([[-2.0, 0.0], [1.0, 3.0]], requires_grad=True)
        base = torch.tensor([[2.0, 0.0], [0.5, 4.0]])

        heuristic = prior_output_to_heuristic(prior, base, logit_net=True)

        self.assertTrue(torch.all(heuristic > 0))
        self.assertLess(heuristic[0, 0].item(), heuristic[1, 1].item())
        heuristic.sum().backward()
        self.assertIsNotNone(prior.grad)
        self.assertGreater(prior.grad.abs().sum().item(), 0.0)

    def test_bpp_pure_aco_rollout_keeps_run_fitness_objective(self):
        class FakeBPPACO:
            def run(self, n_iterations):
                self.n_iterations = n_iterations
                return 7.0

        objective = test_script._run_extended_aco_rollout(FakeBPPACO(), "bpp", 3)

        self.assertEqual(objective, 7.0)

    def test_dynamic_model_can_initialize_from_static_checkpoint(self):
        static_model = create_model("bpp", edge_feats=1, logit_net=True)
        dynamic_model = create_model("bpp", edge_feats=3, logit_net=True)
        source_weight = static_model.state_dict()["emb_net.e_lin0.weight"].detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "static.pt"
            torch.save({"model_state_dict": static_model.state_dict()}, ckpt_path)
            train._init_extended_dynamic_from_static(dynamic_model, str(ckpt_path), "cpu")

        widened = dynamic_model.state_dict()["emb_net.e_lin0.weight"]
        self.assertTrue(torch.equal(widened[:, :1], source_weight))
        self.assertTrue(torch.equal(widened[:, 1:], torch.zeros_like(widened[:, 1:])))


if __name__ == "__main__":
    unittest.main()
