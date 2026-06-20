import argparse
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import train

test_script = importlib.import_module("test")


class UnifiedEntrypointTests(unittest.TestCase):
    def test_train_dispatches_extended_problem_to_base_extended_path(self):
        parsed = argparse.Namespace(problem="bpp", n_node=12)

        with (
            mock.patch.object(train, "_parse_extended_train_args", return_value=parsed) as parse_mock,
            mock.patch.object(train, "_train_extended_main", return_value="extended-train") as train_mock,
        ):
            result = train.main(["--problem", "bpp", "--n_node", "12"])

        parse_mock.assert_called_once_with(["--problem", "bpp", "--n_node", "12"])
        train_mock.assert_called_once_with(parsed)
        self.assertEqual(result, "extended-train")

    def test_train_rejects_extended_only_problem_with_base_only_flag(self):
        with self.assertRaises(SystemExit):
            train.parse_args(["--problem", "bpp", "--ls_scope", "global"])

    def test_train_rejects_base_problem_with_extended_only_flag(self):
        with self.assertRaises(SystemExit):
            train.parse_args(["--problem", "tsp", "--capacity", "150"])

    def test_train_accepts_robust_capacity_for_base_cvrp(self):
        parsed = train.parse_args(["--problem", "cvrp", "--robust-capacity"])

        self.assertTrue(parsed.robust_capacity)

    def test_test_dispatches_extended_problem_to_base_extended_path(self):
        parsed = argparse.Namespace(problem="mkp", checkpoint="dummy.pt")

        with (
            mock.patch.object(test_script, "_parse_extended_test_args", return_value=parsed) as parse_mock,
            mock.patch.object(test_script, "_test_extended_main", return_value="extended-test") as test_mock,
        ):
            result = test_script.main(["--problem", "mkp", "--checkpoint", "dummy.pt"])

        parse_mock.assert_called_once_with(["--problem", "mkp", "--checkpoint", "dummy.pt"])
        test_mock.assert_called_once_with(parsed)
        self.assertEqual(result, "extended-test")

    def test_test_infers_extended_problem_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "bpp.pt"
            torch.save({"config": {"problem": "bpp"}}, checkpoint_path)
            parsed = argparse.Namespace(problem="bpp", checkpoint=str(checkpoint_path))

            with (
                mock.patch.object(test_script, "_parse_extended_test_args", return_value=parsed) as parse_mock,
                mock.patch.object(test_script, "_test_extended_main", return_value="extended-test") as test_mock,
            ):
                result = test_script.main(["--checkpoint", str(checkpoint_path)])

        parse_mock.assert_called_once_with(["--problem", "bpp", "--checkpoint", str(checkpoint_path)])
        test_mock.assert_called_once_with(parsed)
        self.assertEqual(result, "extended-test")


if __name__ == "__main__":
    unittest.main()
