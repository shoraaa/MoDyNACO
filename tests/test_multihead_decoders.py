import unittest

import torch
from torch_geometric.data import Data

import net


def tiny_pyg(edge_feats=6):
    x = torch.rand(4, 2)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]],
        dtype=torch.long,
    )
    edge_attr = torch.rand(edge_index.shape[1], edge_feats)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MultiHeadDecoderTests(unittest.TestCase):
    def _model(self, decoder_type, **kwargs):
        torch.manual_seed(1234)
        return net.MultiHeadNet(
            feats=2,
            edge_feats=6,
            depth=2,
            units=8,
            num_heads=3,
            head_zdim=4,
            rank=2,
            head_decoder_type=decoder_type,
            head_adapter_init="anchored",
            **kwargs,
        )

    def test_decoder_variants_return_edges_by_heads(self):
        pyg = tiny_pyg()
        for decoder_type in ["lora", "deep_lora", "film", "per_head_mlp", "lowrank"]:
            with self.subTest(decoder_type=decoder_type):
                model = self._model(decoder_type)
                out = model(pyg)
                self.assertEqual(tuple(out.shape), (pyg.edge_attr.shape[0], 3))

    def test_deep_lora_anchored_init_keeps_head0_base_but_perturbs_others(self):
        layer = net.MultiHeadLoRALinear(
            in_features=5,
            out_features=5,
            num_heads=3,
            rank=2,
            init_mode="anchored",
            init_std=0.1,
        )
        x = torch.rand(7, 5)
        out = layer(x)
        base = layer.base(x)

        self.assertTrue(torch.allclose(out[:, 0], base, atol=1e-6))
        self.assertGreater((out[:, 1] - base).abs().max().item(), 0.0)

    def test_small_init_randomizes_every_lora_head(self):
        layer = net.MultiHeadLoRALinear(
            in_features=5,
            out_features=5,
            num_heads=3,
            rank=2,
            init_mode="small",
            init_std=0.1,
        )
        x = torch.rand(7, 5)
        out = layer(x)
        base = layer.base(x)

        for head_idx in range(3):
            self.assertGreater((out[:, head_idx] - base).abs().max().item(), 0.0)

    def test_random_init_randomizes_every_lora_head_without_std(self):
        layer = net.MultiHeadLoRALinear(
            in_features=5,
            out_features=5,
            num_heads=3,
            rank=2,
            init_mode="random",
        )
        x = torch.rand(7, 5)
        out = layer(x)
        base = layer.base(x)

        for head_idx in range(3):
            self.assertGreater((out[:, head_idx] - base).abs().max().item(), 0.0)

    def test_deep_lora_chunked_eval_matches_full_eval(self):
        torch.manual_seed(1234)
        decoder = net.ParNetCondDeepLoRA(
            units=5,
            num_heads=3,
            rank=2,
            init_mode="random",
        )
        decoder.eval()
        emb = torch.rand(11, 5)

        decoder.decode_chunk_edges = 0
        full = decoder(emb)
        decoder.decode_chunk_edges = 3
        chunked = decoder(emb)

        self.assertTrue(torch.allclose(full, chunked, atol=1e-6))

    def test_output_to_multi_sparse_priors_keeps_head_order(self):
        output = torch.arange(12, dtype=torch.float32).view(6, 2)
        priors = net.output_to_multi_sparse_priors(output, n=3, k=2)

        self.assertEqual(tuple(priors.shape), (2, 3, 2))
        self.assertTrue(torch.equal(priors[0].reshape(-1), output[:, 0]))
        self.assertTrue(torch.equal(priors[1].reshape(-1), output[:, 1]))

    def test_current_lora_checkpoint_load_still_works(self):
        model = self._model("lora")
        state = model.state_dict()
        target = self._model("lora")

        result = net.load_multihead_state_dict(target, state)

        self.assertFalse(getattr(result, "missing_keys", []))
        self.assertFalse(getattr(result, "unexpected_keys", []))


if __name__ == "__main__":
    unittest.main()
