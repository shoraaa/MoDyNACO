import unittest

import torch

import faco_extended
import utils_extended
from net_extended import NetBPP, NetMKP, NetOP


class ExtendedRegressionTests(unittest.TestCase):
    def test_bpp_pheromone_update_uses_full_sample_batch(self):
        demand = utils_extended.gen_bpp_instance(12, "cpu")
        aco = faco_extended.MFACO_BPP(demand=demand, n_ants=6, device="cpu")

        costs, paths, _, _ = aco.sample(require_prob=False, parallel_traced=True)

        self.assertEqual(paths.ndim, 2)
        self.assertEqual(paths.shape[1], aco.n_ants)
        self.assertEqual(costs.shape[0], aco.n_ants)

        aco.update_pheromone(paths, costs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-device regression coverage")
    def test_bpp_model_moves_cpu_graph_to_cuda(self):
        model = NetBPP(feats=1, edge_feats=2).to("cuda")
        demand = utils_extended.gen_bpp_instance(8, "cpu")
        pyg_data = utils_extended.build_pyg_data_bpp(demand, "cpu")

        out = model(pyg_data)

        self.assertEqual(out.device.type, "cuda")

    def test_bpp_dynamic_toggle_changes_extra_edge_features(self):
        demand = utils_extended.gen_bpp_instance(4, "cpu")
        pheromone = torch.arange(25, dtype=torch.float32).view(5, 5) + 1.0

        dynamic_graph = utils_extended.build_pyg_data_bpp(
            demand, "cpu", pheromone=pheromone, dynamic=True
        )
        static_graph = utils_extended.build_pyg_data_bpp(
            demand, "cpu", pheromone=pheromone, dynamic=False
        )

        self.assertEqual(dynamic_graph.edge_attr.shape[1], 2)
        self.assertTrue(torch.allclose(static_graph.edge_attr[:, 1:], torch.zeros_like(static_graph.edge_attr[:, 1:])))
        self.assertGreater(dynamic_graph.edge_attr[:, 1:].abs().sum().item(), 0.0)

    def test_mkp_builder_matches_model_feature_count(self):
        prize, weight = utils_extended.gen_mkp_instance(4, 5, "cpu")
        pyg_data = utils_extended.build_pyg_data_mkp(prize, weight, "cpu")
        model = NetMKP(m=5, feats=6, edge_feats=2)

        self.assertEqual(pyg_data.x.shape[1], 6)
        self.assertEqual(pyg_data.x.shape[0], prize.shape[0] + 1)
        self.assertEqual(pyg_data.edge_attr.shape[0], (prize.shape[0] + 1) ** 2)
        self.assertEqual(pyg_data.edge_attr.shape[1], 2)

        out = model(pyg_data)

        self.assertEqual(out.shape[0], pyg_data.edge_attr.shape[0])

    def test_op_dense_builder_creates_directed_graph(self):
        coords = utils_extended.gen_op_instance(4, "cpu")
        distances = utils_extended.gen_op_distance_matrix(coords)
        prizes = utils_extended.gen_op_prizes(coords)
        pyg_data = utils_extended.build_pyg_data_op_dense(distances, prizes, "cpu")
        model = NetOP(feats=2, edge_feats=2)

        self.assertEqual(pyg_data.x.shape[0], coords.shape[0] + 1)
        self.assertEqual(pyg_data.edge_index.shape[1], (coords.shape[0] + 1) ** 2)
        self.assertEqual(pyg_data.edge_attr.shape[1], 2)

        out = model(pyg_data)

        self.assertEqual(out.shape[0], pyg_data.edge_attr.shape[0])


if __name__ == "__main__":
    unittest.main()
