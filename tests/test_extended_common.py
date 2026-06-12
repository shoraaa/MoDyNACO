import argparse
import unittest

import torch

import extended_common
from net import NetBPP, NetMKP, NetOP


class ExtendedCommonTests(unittest.TestCase):
    def test_extract_problem_data_normalizes_dict_and_tuple_inputs(self):
        demand = torch.tensor([0.0, 1.0, 2.0])
        prize = torch.tensor([1.0, 2.0])
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        distances = torch.eye(3)
        prizes = torch.tensor([0.0, 1.0, 2.0])

        self.assertTrue(torch.equal(
            extended_common.extract_problem_data("bpp", {"demand": demand})["demand"],
            demand,
        ))
        mkp = extended_common.extract_problem_data("mkp", (prize, weight))
        self.assertTrue(torch.equal(mkp["prize"], prize))
        self.assertTrue(torch.equal(mkp["weight"], weight))
        op = extended_common.extract_problem_data("op", {"distances": distances, "prizes": prizes})
        self.assertTrue(torch.equal(op["distances"], distances))
        self.assertTrue(torch.equal(op["prizes"], prizes))

    def test_create_model_returns_problem_specific_network(self):
        self.assertIsInstance(extended_common.create_model("bpp", edge_feats=2), NetBPP)
        self.assertIsInstance(extended_common.create_model("mkp", m=7, edge_feats=2), NetMKP)
        self.assertIsInstance(extended_common.create_model("op", edge_feats=2), NetOP)

    def test_compute_relative_improvement_handles_raw_and_objective_modes(self):
        self.assertAlmostEqual(
            extended_common.compute_relative_improvement(8.0, 10.0, "bpp"),
            20.0,
        )
        self.assertAlmostEqual(
            extended_common.compute_relative_improvement(12.0, 10.0, "mkp"),
            20.0,
        )
        self.assertAlmostEqual(
            extended_common.compute_relative_improvement(12.0, 10.0, "mkp", objective_mode=True),
            20.0,
        )

    def test_add_extended_problem_args_registers_common_flags(self):
        parser = argparse.ArgumentParser()
        extended_common.add_extended_problem_args(parser)
        args = parser.parse_args(["--problem", "op", "--n_node", "9", "--max_len", "3.5"])
        self.assertEqual(args.problem, "op")
        self.assertEqual(args.n_node, 9)
        self.assertAlmostEqual(args.max_len, 3.5)


if __name__ == "__main__":
    unittest.main()
