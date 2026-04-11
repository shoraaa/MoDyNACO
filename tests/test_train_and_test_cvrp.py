import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

from train_and_test_cvrp import (
    build_test_command,
    build_train_command,
    find_latest_best_checkpoint,
    parse_capacities,
)


def make_args(**overrides):
    base = dict(
        n_node=1000,
        device="cpu",
        save_dir=Path("models"),
        output_dir=None,
        python="python",
        seed=1234,
        epochs=10,
        steps_per_epoch=32,
        k_sparse=32,
        n_ants=100,
        H=10,
        mini_H=100,
        rho=0.1,
        min_new_edges=12,
        lr=None,
        ppo_lr=5e-6,
        reinforce_lr=1e-4,
        algo="ppo",
        alg="faco",
        no_smooth_mmas=True,
        no_local_search=False,
        no_extend_ls=False,
        ls_scope="localized",
        ls_budget="truncated",
        ls_max_opt=0,
        disable_heuristic=False,
        no_normalized_heuristic=False,
        no_dynamic_feats=False,
        no_logit_net=False,
        baseline="none",
        baseline_time_limit=0.5,
        val_size=16,
        threads=None,
        train_warmup=False,
        warmup_ratio=0.5,
        train_anneal=False,
        no_anneal=True,
        gamma=1.0,
        min_gamma=0.0,
        L=0,
        run_name=None,
        summary_json=None,
        metadata_json=None,
        skip_train=False,
        skip_test=False,
        checkpoint=None,
        capacities=None,
        capacity_override=None,
        sweep_csv=None,
        dry_run=False,
        train_extra=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TrainAndTestCVRPTests(unittest.TestCase):
    def test_build_train_command_targets_cvrp_training(self):
        args = make_args()

        cmd = build_train_command(args)

        self.assertEqual(cmd[:4], ["python", "train.py", "--problem", "cvrp"])
        self.assertIn("--n_node", cmd)
        self.assertIn("--save_dir", cmd)
        self.assertIn("--no_smooth_mmas", cmd)
        self.assertIn("--baseline", cmd)
        self.assertIn("--generate_val", cmd)

    def test_build_test_command_uses_resolved_checkpoint(self):
        args = make_args()
        checkpoint = Path("models/cvrp/n1000/example_best.pt")
        summary_json = Path("output/train_and_test_cvrp/evaluation_summary.json")

        cmd = build_test_command(args, checkpoint, summary_json)

        self.assertEqual(cmd[:4], ["python", "test.py", "--problem", "cvrp"])
        self.assertIn(str(checkpoint), cmd)
        self.assertIn(str(summary_json), cmd)
        self.assertIn("--no_anneal", cmd)
        self.assertIn("--generate_val", cmd)

    def test_parse_capacities_accepts_trailing_comma(self):
        self.assertEqual(parse_capacities("100,200,250,"), [100.0, 200.0, 250.0])

    def test_build_commands_include_capacity_override(self):
        args = make_args(capacity_override=250.0)

        train_cmd = build_train_command(args)
        test_cmd = build_test_command(args, Path("x.pt"), Path("y.json"))

        self.assertIn("--capacity_override", train_cmd)
        self.assertIn("250.0", train_cmd)
        self.assertIn("--capacity_override", test_cmd)
        self.assertIn("250.0", test_cmd)

    def test_find_latest_best_checkpoint_prefers_newest_best_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "models" / "cvrp" / "n1000"
            root.mkdir(parents=True)
            older = root / "older_best.pt"
            newer = root / "newer_best.pt"
            older.write_text("x", encoding="utf-8")
            time.sleep(0.01)
            newer.write_text("y", encoding="utf-8")

            found = find_latest_best_checkpoint(Path(tmpdir) / "models", 1000)

            self.assertEqual(found, newer)


if __name__ == "__main__":
    unittest.main()
