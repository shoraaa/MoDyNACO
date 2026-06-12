import unittest
from argparse import Namespace

import torch

import l2c_core
import test_l2c
import train_l2c


class L2CPipelineTests(unittest.TestCase):
    def test_normalize_tsp_dataset_supports_tuple_entries(self):
        coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        dataset = [(coords, 3.14, [0, 1, 2], "tri")]

        normalized = l2c_core.normalize_tsp_dataset(dataset)

        self.assertEqual(len(normalized), 1)
        self.assertTrue(torch.equal(normalized.coords[0], coords))
        self.assertAlmostEqual(normalized.costs[0], 3.14)
        self.assertTrue(torch.equal(normalized.tours[0], torch.tensor([0, 1, 2])))
        self.assertEqual(normalized.names[0], "tri")

    def test_normalize_tsp_dataset_supports_tensor_batch(self):
        dataset = torch.rand(2, 5, 2)
        normalized = l2c_core.normalize_tsp_dataset(dataset)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized.coords[0].shape, (5, 2))

    def test_train_l2c_build_model_name_appends_l2c_suffix(self):
        args = Namespace(
            problem="tsp",
            n_node=20,
            k_sparse=32,
            n_ants=10,
            H=5,
            mini_H=5,
            rho=0.1,
            min_new_edges=12,
            algo="ppo",
            lr=1e-4,
            capacity_override=None,
            train_anneal=False,
            gamma=1.0,
            min_gamma=0.0,
            L=0,
            train_warmup=False,
            warmup_ratio=0.5,
            train_deepaco=False,
            no_dynamic_feats=False,
            no_smooth_mmas=False,
            disable_heuristic=False,
            no_local_search=False,
            no_extend_ls=False,
            ls_scope="localized",
            ls_budget="truncated",
            ls_max_opt=0,
            no_normalized_heuristic=False,
            ablation_pheromone_features=False,
            ablation_incumbent_features=False,
            alg="faco",
        )
        self.assertTrue(train_l2c.build_model_name(args).endswith("_l2c"))

    def test_train_l2c_parser_defaults_to_original_style_values(self):
        args = train_l2c._parse_args([])
        self.assertEqual(args.problem, "tsp")
        self.assertEqual(args.encoder_layer_num, 6)
        self.assertEqual(args.k_nearest_num, 1000)
        self.assertEqual(args.train_batch_size, 256)
        self.assertEqual(args.max_subtour_length, 1000)

    def test_test_l2c_wrapper_injects_defaults(self):
        argv = test_l2c._inject_l2c_defaults(["--checkpoint", "none"])
        self.assertIn("--backend", argv)
        self.assertIn("l2c", argv)
        self.assertIn("--problem", argv)
        self.assertIn("tsp", argv)


if __name__ == "__main__":
    unittest.main()
