import unittest

import torch

import faco
import utils
from net import Net


class CVRPGraphRegressionTests(unittest.TestCase):
    def test_cvrp_builder_emits_four_node_features_expected_by_checkpointed_models(self):
        coords, demand, capacity = utils.gen_cvrp_instance(8, "cpu", capacity=20)
        aco = faco.MFACO_CVRP(
            coords=coords,
            demand=demand,
            capacity=capacity,
            n_ants=4,
            cand_list_size=4,
            device="cpu",
            use_local_search=False,
        )

        pyg_data = utils.build_pyg_data_cvrp(aco, coords, demand, "cpu", dynamic=True)

        self.assertEqual(pyg_data.x.shape, (9, 4))
        self.assertTrue(torch.allclose(pyg_data.x[:, :2], coords))
        self.assertTrue(torch.allclose(pyg_data.x[:, 2], demand))
        self.assertEqual(pyg_data.x[0, 3].item(), 1.0)
        self.assertTrue(torch.allclose(pyg_data.x[1:, 3], torch.zeros(8)))

        model = Net(feats=4, edge_feats=pyg_data.edge_attr.shape[1])
        out = model(pyg_data)

        self.assertEqual(out.shape[0], pyg_data.edge_attr.shape[0])


if __name__ == "__main__":
    unittest.main()
