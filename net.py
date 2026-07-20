import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
import math
import os


def move_pyg_to_module_device(module: nn.Module, pyg):
    """Normalize a PyG batch onto the same device as the module parameters."""
    try:
        device = next(module.parameters()).device
    except StopIteration:
        return pyg

    if hasattr(pyg, "to"):
        x = getattr(pyg, "x", None)
        edge_index = getattr(pyg, "edge_index", None)
        edge_attr = getattr(pyg, "edge_attr", None)
        needs_move = any(
            tensor is not None and hasattr(tensor, "device") and tensor.device != device
            for tensor in (x, edge_index, edge_attr)
        )
        if needs_move:
            return pyg.to(device)
    return pyg


# =============================================================================
# GNN Layer (shared by all problems)
# =============================================================================

class GNNLayer(nn.Module):
    def __init__(self, units, act_fn, agg_fn):
        super().__init__()
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.v_lin1 = nn.Linear(units, units)
        self.v_lin2 = nn.Linear(units, units)
        self.v_lin3 = nn.Linear(units, units)
        self.v_lin4 = nn.Linear(units, units)
        self.v_bn = gnn.BatchNorm(units)
        self.e_lin0 = nn.Linear(units, units)
        self.e_bn = gnn.BatchNorm(units)

    def forward(self, x, w, edge_index):
        x0 = x
        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)
        w0 = w
        w1 = self.e_lin0(w0)
        w2 = torch.sigmoid(w0)
        x = x0 + self.act_fn(self.v_bn(x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w


# =============================================================================
# Embedding Networks (with optional gradient checkpointing)
# =============================================================================

class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=6, units=32, act_fn='silu', agg_fn='mean', grad_checkpointing=False):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.grad_checkpointing = grad_checkpointing

        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)

        self.layers = nn.ModuleList([
            GNNLayer(self.units, self.act_fn, self.agg_fn) for _ in range(self.depth)
        ])

        self._register_load_state_dict_pre_hook(self._load_compat_hook)

    def _load_compat_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Backward compatibility hook to map old state_dict keys to new GNNLayer structure.
        Old: v_lins1.0.weight -> New: layers.0.v_lin1.weight
        """
        keys_to_rewrite = []
        for key in state_dict.keys():
            if not key.startswith(prefix):
                continue

            local_key = key[len(prefix):]

            if local_key.startswith("v_lins1."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin1.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_lins2."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin2.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_lins3."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin3.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_lins4."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin4.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("e_lins0."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.e_lin0.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_bns."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_bn.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("e_bns."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.e_bn.{suffix}"
                keys_to_rewrite.append((key, new_key))

        for old_key, new_key in keys_to_rewrite:
            if new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

    def forward(self, x, edge_index, edge_attr):
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)

        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                 x, w = torch.utils.checkpoint.checkpoint(layer, x, w, edge_index, use_reentrant=False)
            else:
                 x, w = layer(x, w, edge_index)
        return w


class EmbNetCheckpoint(nn.Module):
    """Checkpointed version of EmbNet for memory efficiency."""
    def __init__(self, depth=12, feats=2, edge_feats=6, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')

        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)

        self.layers = nn.ModuleList([
            GNNLayer(self.units, self.act_fn, self.agg_fn) for _ in range(self.depth)
        ])

    def forward(self, x, edge_index, edge_attr):
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)

        for layer in self.layers:
            if self.training:
                 x, w = torch.utils.checkpoint.checkpoint(layer, x, w, edge_index, use_reentrant=False)
            else:
                 x, w = layer(x, w, edge_index)
        return w


# =============================================================================
# Parameter Network (MLP)
# =============================================================================

class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn, sigmoid_output=True):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                if self.sigmoid_output:
                    x = torch.sigmoid(x)
        return x


class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu', logit_net=False):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn, sigmoid_output=not logit_net)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)


class ParNetCondFiLM(nn.Module):
    """Head-conditioned edge decoder used by MultiHeadNet."""

    def __init__(
        self,
        depth=3,
        units=32,
        num_heads=4,
        zdim=16,
        act_fn='silu',
        logit_net=False,
        init_mode="anchored",
        init_std=0.02,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("ParNetCondFiLM requires depth >= 2")
        self.units = units
        self.num_heads = num_heads
        self.zdim = zdim
        self.act_fn = getattr(F, act_fn)
        self.sigmoid_output = not logit_net
        self.input = nn.Linear(units, units)
        self.hidden = nn.ModuleList([nn.Linear(units, units) for _ in range(depth - 2)])
        self.output = nn.Linear(units, 1)
        self.gamma = nn.ModuleList([nn.Linear(zdim, units) for _ in range(depth - 1)])
        self.beta = nn.ModuleList([nn.Linear(zdim, units) for _ in range(depth - 1)])
        self.decode_chunk_edges = int(os.getenv("DYNACO_DECODE_CHUNK_EDGES", "0"))
        self._init_conditioners(init_mode, init_std)

    def _init_conditioners(self, init_mode, init_std):
        mode = str(init_mode or "anchored").lower()
        if mode not in {"anchored", "zero", "small", "random"}:
            raise ValueError("head_adapter_init must be one of: anchored, zero, small, random")
        if mode == "random":
            return
        for layer in list(self.gamma) + list(self.beta):
            nn.init.zeros_(layer.bias)
            if mode == "zero":
                nn.init.zeros_(layer.weight)
            else:
                nn.init.normal_(layer.weight, std=init_std)

    def _apply_film(self, x, z, layer_idx):
        gamma = self.gamma[layer_idx](z).unsqueeze(0)
        beta = self.beta[layer_idx](z).unsqueeze(0)
        return x * (1.0 + gamma) + beta

    def _forward_impl(self, emb, z):
        # emb: (E, units), z: (K, zdim), output: (E, K)
        e = emb.unsqueeze(1).expand(-1, z.shape[0], -1)
        x = self.input(e)
        x = self._apply_film(x, z, 0)
        x = self.act_fn(x)
        for layer_idx, layer in enumerate(self.hidden, start=1):
            x = layer(x)
            x = self._apply_film(x, z, layer_idx)
            x = self.act_fn(x)
        x = self.output(x).squeeze(-1)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

    def forward(self, emb, z):
        chunk = int(getattr(self, "decode_chunk_edges", 0) or 0)
        if self.training or chunk <= 0 or emb.shape[0] <= chunk:
            return self._forward_impl(emb, z)
        outs = [self._forward_impl(emb[start:start + chunk], z) for start in range(0, emb.shape[0], chunk)]
        return torch.cat(outs, dim=0)


# =============================================================================
# Unified Network Class (supports all problem types)
# =============================================================================

class Net(nn.Module):
    """
    Unified neural network for all problem types (TSP, CVRP, BPP, MKP, OP).

    Args:
        problem_type: Problem type ('tsp', 'cvrp', 'bpp', 'mkp', 'op')
        feats: Number of node features (auto-configured if problem_type given)
        edge_feats: Number of edge features (auto-configured if problem_type given)
        m: Number of constraints for MKP (required if problem_type='mkp' and feats not given)
        grad_checkpointing: Enable gradient checkpointing for memory savings
        **kwargs: Additional arguments
    """

    # Default feature configurations for each problem type
    PROBLEM_CONFIGS = {
        'tsp': {'feats': 2, 'edge_feats': 6, 'embed_units': 32},
        'cvrp': {'feats': 4, 'edge_feats': 6, 'embed_units': 32},  # coords(2) + demand(1) + depot_flag(1)
        'bpp': {'feats': 1, 'edge_feats': 2, 'embed_units': 32},
        'mkp': {'feats': None, 'edge_feats': 2, 'embed_units': 32},  # feats = m+1, m required
        'op': {'feats': 2, 'edge_feats': 2, 'embed_units': 32},
    }

    def __init__(
        self,
        problem_type: str = None,
        feats: int = None,
        edge_feats: int = None,
        m: int = 5,
        grad_checkpointing: bool = False,
        logit_net: bool = False,
        **kwargs
    ):
        super().__init__()

        # Determine problem configuration
        if problem_type is not None:
            if problem_type not in self.PROBLEM_CONFIGS:
                raise ValueError(f"Unknown problem_type: {problem_type}. Must be one of {list(self.PROBLEM_CONFIGS.keys())}")

            config = self.PROBLEM_CONFIGS[problem_type]
            # Use provided feats/edge_feats if given, otherwise use defaults
            self.feats = feats if feats is not None else config['feats']
            self.edge_feats = edge_feats if edge_feats is not None else config['edge_feats']

            # Special handling for MKP
            if problem_type == 'mkp' and self.feats is None:
                self.feats = m + 1
        else:
            # Backward compatibility: if no problem_type, require explicit feats/edge_feats
            if feats is None or edge_feats is None:
                raise ValueError("Must specify either problem_type or both feats and edge_feats")
            self.feats = feats
            self.edge_feats = edge_feats

        # Create embedding network with optional checkpointing
        if grad_checkpointing:
            self.emb_net = EmbNetCheckpoint(
                feats=self.feats,
                edge_feats=self.edge_feats,
                **kwargs
            )
        else:
            self.emb_net = EmbNet(
                feats=self.feats,
                edge_feats=self.edge_feats,
                **kwargs
            )

        # Parameter network (shared across all problems)
        self.par_net_heu = ParNet(logit_net=logit_net)

    def forward(self, pyg):
        pyg = move_pyg_to_module_device(self, pyg)
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)
        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False


# class MultiHeadNet(Net):
#     """DyNACO edge-prior model with K cheap head-conditioned decoders."""

#     def __init__(
#         self,
#         *args,
#         num_heads: int = 4,
#         head_zdim: int = 16,
#         logit_net: bool = False,
#         **kwargs
#     ):
#         self.num_heads = num_heads
#         self.head_zdim = head_zdim
#         super().__init__(*args, logit_net=logit_net, **kwargs)
#         # self.head_codes = nn.Parameter(torch.randn(num_heads, head_zdim) * 0.02)
#         codes = torch.randn(num_heads, head_zdim)
#         codes = F.normalize(codes, dim=-1)
#         self.register_buffer("head_codes", codes)
#         self.par_net_heu = ParNetCondFiLM(
#             num_heads=num_heads,
#             zdim=head_zdim,
#             logit_net=logit_net,
#         )

#     def forward(self, pyg):
#         pyg = move_pyg_to_module_device(self, pyg)
#         x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
#         emb = self.emb_net(x, edge_index, edge_attr)
#         return self.par_net_heu(emb, self.head_codes)

class ParNetCondLowRank(nn.Module):
    """
    Memory-efficient head-conditioned residual decoder.

    Input:
        emb: (E, units)
        head_codes: (H, zdim)

    Output:
        logits: (E, H)
    """

    def __init__(
        self,
        units=32,
        num_heads=4,
        zdim=16,
        rank=16,
        act_fn='silu',
        logit_net=True,
        zero_init=True,
    ):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        self.zdim = zdim
        self.rank = rank
        self.act_fn = getattr(F, act_fn)
        self.sigmoid_output = not logit_net

        # Normal DyNACO base decoder.
        self.base = ParNet(depth=3, units=units, preds=1, act_fn=act_fn, logit_net=True)

        # Low-rank residual adapter.
        self.edge_proj = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, rank),
        )
        self.head_proj = nn.Linear(zdim, rank, bias=False)
        self.head_bias = nn.Linear(zdim, 1, bias=False)

        if zero_init:
            nn.init.zeros_(self.edge_proj[-1].weight)
            nn.init.zeros_(self.edge_proj[-1].bias)
            nn.init.zeros_(self.head_bias.weight)

    def forward(self, emb, head_codes):
        # base: (E,)
        base = self.base(emb)

        # edge_factor: (E, rank)
        edge_factor = self.edge_proj(emb)

        # head_factor: (H, rank)
        head_factor = self.head_proj(head_codes)

        # delta: (E, H)
        delta = edge_factor @ head_factor.t()

        # head-specific bias: (H,)
        bias = self.head_bias(head_codes).squeeze(-1)

        out = base[:, None] + delta + bias[None, :]

        if self.sigmoid_output:
            out = torch.sigmoid(out)

        return out

class MultiHeadLoRALinear(nn.Module):
    """Linear layer with one shared base weight and head-specific LoRA updates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = True,
        init_mode: str = "anchored",
        init_std: float = 0.02,
        freeze_base: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        init_mode = str(init_mode or "anchored").lower()
        if init_mode not in {"anchored", "zero", "small", "random"}:
            raise ValueError("head_adapter_init must be one of: anchored, zero, small, random")
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.rank = rank
        self.scaling = alpha / rank

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.empty(num_heads, rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(num_heads, out_features, rank))

        if init_mode == "random":
            for h in range(num_heads):
                nn.init.kaiming_uniform_(self.lora_A[h], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_B[h], a=math.sqrt(5))
        elif init_mode == "zero":
            nn.init.normal_(self.lora_A, std=init_std)
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.normal_(self.lora_A, std=init_std)
            nn.init.normal_(self.lora_B, std=init_std)
            if init_mode == "anchored":
                with torch.no_grad():
                    self.lora_B[0].zero_()

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: (E, D) or (E, H, D); out: (E, H, out_features)
        if x.dim() == 2:
            base = self.base(x).unsqueeze(1)
            low = torch.einsum("ed,hrd->ehr", x, self.lora_A)
        elif x.dim() == 3:
            if x.shape[1] != self.num_heads:
                raise ValueError(f"Expected {self.num_heads} heads, got {x.shape[1]}")
            base = self.base(x)
            low = torch.einsum("ehd,hrd->ehr", x, self.lora_A)
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")
        delta = torch.einsum("ehr,hor->eho", low, self.lora_B)
        if base.shape == delta.shape:
            return base.add_(delta, alpha=self.scaling)
        return delta.mul_(self.scaling).add_(base)

    def forward_head(self, x, head_idx: int):
        """Evaluate one LoRA head without materializing the full head axis."""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input for one-head decode, got shape {tuple(x.shape)}")
        if head_idx < 0 or head_idx >= self.num_heads:
            raise ValueError(f"head_idx must be in [0, {self.num_heads}), got {head_idx}")
        base = self.base(x)
        low = F.linear(x, self.lora_A[head_idx])
        delta = F.linear(low, self.lora_B[head_idx])
        return base + self.scaling * delta


class ParNetCondLoRA(nn.Module):
    """
    Multi-head LoRA-MLP edge decoder.

    The MLP trunk is shared. The final linear layer is W_h = W_0 + B_h A_h,
    giving each head a small trainable low-rank adapter while preserving the
    existing multi-head output contract: (E, H).
    """

    def __init__(
        self,
        depth: int = 3,
        units: int = 32,
        num_heads: int = 4,
        rank: int = 8,
        alpha: float = 1.0,
        act_fn: str = 'silu',
        logit_net: bool = True,
        init_mode: str = "anchored",
        init_std: float = 0.02,
        freeze_base: bool = False,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("ParNetCondLoRA requires depth >= 2")
        self.units = units
        self.num_heads = num_heads
        self.rank = rank
        self.act_fn = getattr(F, act_fn)
        self.sigmoid_output = not logit_net
        self.hidden = nn.ModuleList([
            nn.Linear(units, units) for _ in range(depth - 1)
        ])
        self.out = MultiHeadLoRALinear(
            in_features=units,
            out_features=1,
            num_heads=num_heads,
            rank=rank,
            alpha=alpha,
            bias=True,
            init_mode=init_mode,
            init_std=init_std,
            freeze_base=freeze_base,
        )

    def forward(self, emb):
        x = emb
        for layer in self.hidden:
            x = self.act_fn(layer(x))
        out = self.out(x).squeeze(-1)
        if self.sigmoid_output:
            out = torch.sigmoid(out)
        return out


class ParNetCondDeepLoRA(nn.Module):
    """Head-specific LoRA adapters at every decoder layer."""

    def __init__(
        self,
        depth: int = 3,
        units: int = 32,
        num_heads: int = 4,
        rank: int = 4,
        alpha: float = 1.0,
        act_fn: str = 'silu',
        logit_net: bool = True,
        init_mode: str = "anchored",
        init_std: float = 0.02,
        freeze_base: bool = False,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("ParNetCondDeepLoRA requires depth >= 2")
        self.units = units
        self.num_heads = num_heads
        self.rank = rank
        self.decode_chunk_edges = int(os.getenv("DYNACO_DECODE_CHUNK_EDGES", "0"))
        self.decode_headwise = os.getenv("DYNACO_DECODE_HEADWISE", "0").lower() in {"1", "true", "yes"}
        self.act_fn = getattr(F, act_fn)
        self.sigmoid_output = not logit_net
        self.hidden = nn.ModuleList([
            MultiHeadLoRALinear(
                units,
                units,
                num_heads=num_heads,
                rank=rank,
                alpha=alpha,
                bias=True,
                init_mode=init_mode,
                init_std=init_std,
                freeze_base=freeze_base,
            )
            for _ in range(depth - 1)
        ])
        self.out = MultiHeadLoRALinear(
            units,
            1,
            num_heads=num_heads,
            rank=rank,
            alpha=alpha,
            bias=True,
            init_mode=init_mode,
            init_std=init_std,
            freeze_base=freeze_base,
        )

    def _forward_impl(self, emb):
        x = emb
        for layer in self.hidden:
            x = self.act_fn(layer(x))
        out = self.out(x).squeeze(-1)
        if self.sigmoid_output:
            out = torch.sigmoid(out)
        return out

    def _forward_head(self, emb, head_idx: int):
        x = emb
        for layer in self.hidden:
            x = self.act_fn(layer.forward_head(x, head_idx))
        out = self.out.forward_head(x, head_idx).squeeze(-1)
        if self.sigmoid_output:
            out = torch.sigmoid(out)
        return out

    def _forward_impl_headwise(self, emb):
        outs = [self._forward_head(emb, head_idx) for head_idx in range(self.num_heads)]
        return torch.stack(outs, dim=1)

    def forward(self, emb):
        chunk = int(getattr(self, "decode_chunk_edges", 0) or 0)
        if self.training:
            return self._forward_impl(emb)
        if self.decode_headwise:
            return self._forward_impl_headwise(emb)
        if chunk <= 0 or emb.shape[0] <= chunk:
            return self._forward_impl(emb)
        outs = [self._forward_impl(emb[start:start + chunk]) for start in range(0, emb.shape[0], chunk)]
        return torch.cat(outs, dim=0)


class ParNetMultiDecoder(nn.Module):
    """Independent full decoder per head over the shared GNN embedding."""

    def __init__(
        self,
        depth: int = 3,
        units: int = 32,
        num_heads: int = 4,
        act_fn: str = 'silu',
        logit_net: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            ParNet(depth=depth, units=units, preds=1, act_fn=act_fn, logit_net=logit_net)
            for _ in range(num_heads)
        ])

    def forward(self, emb):
        return torch.stack([head(emb) for head in self.heads], dim=1)


ParNetMultiMLP = ParNetMultiDecoder


def make_polynet_codes(num_heads: int, zdim: int) -> torch.Tensor:
    """Create fixed, unique binary strategy codes for PolyNet heads."""
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if zdim <= 0:
        raise ValueError("head_zdim must be positive")
    bits = max(1, math.ceil(math.log2(max(num_heads, 2))))
    if zdim < bits:
        raise ValueError(f"head_zdim={zdim} is too small for {num_heads} unique binary codes; need at least {bits}")
    codes = torch.zeros(num_heads, zdim)
    for head_idx in range(num_heads):
        for bit_idx in range(bits):
            codes[head_idx, bit_idx] = float((head_idx >> bit_idx) & 1)
    return codes


class ParNetPolyNet(nn.Module):
    """Single shared decoder with a PolyNet-style code-conditioned residual block."""

    def __init__(
        self,
        depth: int = 3,
        units: int = 32,
        zdim: int = 16,
        poly_units: int = 256,
        act_fn: str = 'silu',
        logit_net: bool = True,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("ParNetPolyNet requires depth >= 2")
        self.units = units
        self.zdim = zdim
        self.poly_units = poly_units
        self.decode_chunk_edges = int(os.getenv("DYNACO_DECODE_CHUNK_EDGES", "0"))
        self.act_fn = getattr(F, act_fn)
        self.sigmoid_output = not logit_net
        self.hidden = nn.ModuleList([
            nn.Linear(units, units) for _ in range(depth - 1)
        ])
        self.poly_residual = nn.Sequential(
            nn.Linear(units + zdim, poly_units),
            nn.ReLU(),
            nn.Linear(poly_units, units),
        )
        self.out = nn.Linear(units, 1)
        nn.init.zeros_(self.poly_residual[-1].weight)
        nn.init.zeros_(self.poly_residual[-1].bias)

    def _forward_impl(self, emb, head_codes):
        x = emb
        for layer in self.hidden:
            x = self.act_fn(layer(x))
        e = x.shape[0]
        h = head_codes.shape[0]
        xh = x.unsqueeze(1).expand(e, h, self.units)
        zh = head_codes.unsqueeze(0).expand(e, h, self.zdim)
        xh = xh + self.poly_residual(torch.cat([xh, zh], dim=-1))
        out = self.out(xh).squeeze(-1)
        if self.sigmoid_output:
            out = torch.sigmoid(out)
        return out

    def forward(self, emb, head_codes):
        chunk = int(getattr(self, "decode_chunk_edges", 0) or 0)
        if self.training or chunk <= 0 or emb.shape[0] <= chunk:
            return self._forward_impl(emb, head_codes)
        outs = [
            self._forward_impl(emb[start:start + chunk], head_codes)
            for start in range(0, emb.shape[0], chunk)
        ]
        return torch.cat(outs, dim=0)


class MultiHeadNet(Net):
    """DyNACO edge-prior model with a shared encoder and multi-head decoder."""

    def __init__(
        self,
        *args,
        num_heads: int = 4,
        head_zdim: int = 16,
        rank: int = 8,
        lora_alpha: float = 1.0,
        freeze_lora_base: bool = False,
        head_decoder_type: str = "lora",
        poly_units: int = 256,
        head_adapter_init: str = "anchored",
        head_adapter_init_std: float = 0.02,
        logit_net: bool = True,
        fixed_head_codes: bool = True,
        alloc_mode: str = "mlp",
        **kwargs
    ):
        self.num_heads = num_heads
        self.head_zdim = head_zdim
        self.rank = rank
        self.alloc_mode = str(alloc_mode or "mlp").lower()
        self.head_decoder_type = str(head_decoder_type or "lora").lower()
        if self.head_decoder_type in {"head_code", "head-code", "residual"}:
            self.head_decoder_type = "lowrank"
        if self.head_decoder_type in {"deep-lora", "deeplora"}:
            self.head_decoder_type = "deep_lora"
        if self.head_decoder_type in {"per-head-mlp", "perhead_mlp", "perhead"}:
            self.head_decoder_type = "per_head_mlp"
        if self.head_decoder_type in {"multi-decoder", "multidecoder", "multi_mlp", "stacked_decoder"}:
            self.head_decoder_type = "multi_decoder"
        if self.head_decoder_type == "per_head_mlp":
            self.head_decoder_type = "multi_decoder"
        if self.head_decoder_type in {"poly-net", "poly_net", "code_decoder", "code_conditioned_decoder"}:
            self.head_decoder_type = "polynet"
        valid_decoders = {"lora", "deep_lora", "film", "multi_decoder", "lowrank", "polynet"}
        if self.head_decoder_type not in valid_decoders:
            raise ValueError(f"head_decoder_type must be one of {sorted(valid_decoders)}")
        super().__init__(*args, logit_net=True, **kwargs)

        units = self.emb_net.units

        if self.head_decoder_type == "lowrank":
            if fixed_head_codes:
                codes = torch.randn(num_heads, head_zdim)
                codes = F.normalize(codes, dim=-1)
                self.register_buffer("head_codes", codes)
            else:
                codes = torch.randn(num_heads, head_zdim) * 0.02
                self.head_codes = nn.Parameter(codes)
            self.par_net_heu = ParNetCondLowRank(
                units=units,
                num_heads=num_heads,
                zdim=head_zdim,
                rank=rank,
                logit_net=logit_net,
                zero_init=True,
            )
        elif self.head_decoder_type == "polynet":
            codes = make_polynet_codes(num_heads, head_zdim)
            self.register_buffer("head_codes", codes)
            self.par_net_heu = ParNetPolyNet(
                units=units,
                zdim=head_zdim,
                poly_units=poly_units,
                logit_net=logit_net,
            )
        elif self.head_decoder_type == "film":
            adapter_init = str(head_adapter_init or "anchored").lower()
            if fixed_head_codes:
                codes = torch.randn(num_heads, head_zdim)
                if adapter_init == "anchored":
                    codes[0].zero_()
                    if num_heads > 1:
                        codes[1:] = F.normalize(codes[1:], dim=-1)
                else:
                    codes = F.normalize(codes, dim=-1)
                self.register_buffer("head_codes", codes)
            else:
                codes = torch.randn(num_heads, head_zdim)
                if adapter_init != "random":
                    codes = codes * head_adapter_init_std
                if adapter_init == "anchored":
                    codes[0].zero_()
                self.head_codes = nn.Parameter(codes)
            self.par_net_heu = ParNetCondFiLM(
                units=units,
                num_heads=num_heads,
                zdim=head_zdim,
                logit_net=logit_net,
                init_mode=head_adapter_init,
                init_std=head_adapter_init_std,
            )
        elif self.head_decoder_type == "deep_lora":
            self.par_net_heu = ParNetCondDeepLoRA(
                units=units,
                num_heads=num_heads,
                rank=rank,
                alpha=lora_alpha,
                logit_net=logit_net,
                init_mode=head_adapter_init,
                init_std=head_adapter_init_std,
                freeze_base=freeze_lora_base,
            )
        elif self.head_decoder_type == "multi_decoder":
            self.par_net_heu = ParNetMultiDecoder(
                units=units,
                num_heads=num_heads,
                logit_net=logit_net,
            )
        else:
            self.par_net_heu = ParNetCondLoRA(
                units=units,
                num_heads=num_heads,
                rank=rank,
                alpha=lora_alpha,
                logit_net=logit_net,
                init_mode=head_adapter_init,
                init_std=head_adapter_init_std,
                freeze_base=freeze_lora_base,
            )
        alloc_in = 2 * units if self.alloc_mode == "mlp_meanstd" else units
        self.alloc_net = nn.Sequential(
            nn.Linear(alloc_in, units),
            nn.SiLU(),
            nn.Linear(units, num_heads),
        )
        nn.init.zeros_(self.alloc_net[-1].weight)
        nn.init.zeros_(self.alloc_net[-1].bias)

    def _decode_heads(self, emb):
        if self.head_decoder_type in {"lowrank", "film", "polynet"}:
            return self.par_net_heu(emb, self.head_codes)
        return self.par_net_heu(emb)

    def _alloc_input(self, emb):
        if self.alloc_mode == "mlp_meanstd":
            return torch.cat([emb.mean(dim=0), emb.std(dim=0)], dim=0)
        return emb.mean(dim=0)

    def forward_with_alloc(self, pyg):
        pyg = move_pyg_to_module_device(self, pyg)
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        prior = self._decode_heads(emb)
        alloc_logits = self.alloc_net(self._alloc_input(emb))
        return prior, alloc_logits

    def forward(self, pyg):
        prior, _ = self.forward_with_alloc(pyg)
        return prior


def lowrank_head_adapter_is_dead(model: nn.Module, atol: float = 1e-12) -> bool:
    """Return True for old low-rank head adapters with both bilinear factors zero."""
    decoder = getattr(model, "par_net_heu", None)
    if not isinstance(decoder, ParNetCondLowRank):
        return False
    edge_last = decoder.edge_proj[-1]
    return (
        torch.allclose(edge_last.weight, torch.zeros_like(edge_last.weight), atol=atol)
        and torch.allclose(edge_last.bias, torch.zeros_like(edge_last.bias), atol=atol)
        and torch.allclose(decoder.head_proj.weight, torch.zeros_like(decoder.head_proj.weight), atol=atol)
    )


def repair_dead_lowrank_head_adapter(model: nn.Module) -> bool:
    """Reinitialize the head factor for resumable old checkpoints.

    The edge-side residual stays zero, so the resumed model initially preserves
    the base decoder output, but gradients can now flow into the head adapter.
    """
    if not lowrank_head_adapter_is_dead(model):
        return False
    decoder = model.par_net_heu
    nn.init.xavier_uniform_(decoder.head_proj.weight)
    return True


def load_multihead_state_dict(model: nn.Module, state_dict: dict):
    """Load current LoRA checkpoints and migrate old residual-head checkpoints.

    Older MultiHeadNet checkpoints stored the shared decoder as
    par_net_heu.base.lins.{0,1,2}.  The LoRA decoder stores the same reusable
    trunk/base weights as par_net_heu.hidden.{0,1} and par_net_heu.out.base.
    Old head-code and residual-adapter tensors are intentionally ignored.
    """
    if not isinstance(model, MultiHeadNet):
        return model.load_state_dict(state_dict)
    if getattr(model, "head_decoder_type", "lora") == "lowrank":
        return model.load_state_dict(state_dict, strict=False)

    migrated = dict(state_dict)
    old_prefix = "par_net_heu.base.lins."
    if (
        f"{old_prefix}0.weight" in migrated
        and "par_net_heu.hidden.0.weight" not in migrated
    ):
        mapping = {
            f"{old_prefix}0.weight": "par_net_heu.hidden.0.weight",
            f"{old_prefix}0.bias": "par_net_heu.hidden.0.bias",
            f"{old_prefix}1.weight": "par_net_heu.hidden.1.weight",
            f"{old_prefix}1.bias": "par_net_heu.hidden.1.bias",
            f"{old_prefix}2.weight": "par_net_heu.out.base.weight",
            f"{old_prefix}2.bias": "par_net_heu.out.base.bias",
        }
        for old_key, new_key in mapping.items():
            if old_key in migrated:
                migrated[new_key] = migrated[old_key]

    # Migrate alloc_net when switching to mlp_meanstd from an old checkpoint
    alloc_mode = getattr(model, "alloc_mode", "mlp")
    alloc_w_key = "alloc_net.0.weight"
    if (
        alloc_mode == "mlp_meanstd"
        and alloc_w_key in migrated
        and migrated[alloc_w_key].shape[1] != 2 * model.emb_net.units
    ):
        alloc_keys = [k for k in list(migrated) if k.startswith("alloc_net.")]
        for k in alloc_keys:
            del migrated[k]
        print("Migrating checkpoint: dropped old alloc_net weights for mlp_meanstd mode")

    is_current_lora = "par_net_heu.out.lora_A" in migrated
    if is_current_lora:
        return model.load_state_dict(migrated, strict=False)

    return model.load_state_dict(migrated, strict=False)

def output_to_sparse_prior(output, n: int, k: int):
    """Convert a single-head edge output into one (n, k) sparse prior."""
    if output.dim() == 2:
        raise ValueError("Multi-head output should use output_to_multi_sparse_priors.")
    return output.view(-1).view(n, k)


def output_to_multi_sparse_priors(output, n: int, k: int):
    """Convert model output into (heads, n, k); single-head output becomes one head."""
    if output.dim() == 1:
        return output.view(1, n, k)
    if output.dim() == 2:
        return output.transpose(0, 1).contiguous().view(output.shape[1], n, k)
    raise ValueError(f"Expected 1D or 2D model output, got shape {tuple(output.shape)}")


# =============================================================================
# Backward-Compatible Specialized Wrappers
# =============================================================================

class NetBPP(Net):
    """Neural network for Bin Packing Problem (backward compatibility wrapper)."""
    def __init__(self, feats=1, edge_feats=2, **kwargs):
        super().__init__(problem_type='bpp', feats=feats, edge_feats=edge_feats, **kwargs)


class NetMKP(Net):
    """Neural network for Multi-dimensional Knapsack Problem (backward compatibility wrapper)."""
    def __init__(self, m=5, feats=None, edge_feats=2, **kwargs):
        if feats is None:
            feats = m + 1
        super().__init__(problem_type='mkp', feats=feats, edge_feats=edge_feats, m=m, **kwargs)


class NetOP(Net):
    """Neural network for Orienteering Problem (backward compatibility wrapper)."""
    def __init__(self, feats=2, edge_feats=2, **kwargs):
        super().__init__(problem_type='op', feats=feats, edge_feats=edge_feats, **kwargs)


# =============================================================================
# Factory Function
# =============================================================================

def get_network(problem: str, **kwargs):
    """
    Get the appropriate network for a given problem.

    Args:
        problem: Problem type ('tsp', 'cvrp', 'bpp', 'mkp', 'op')
        **kwargs: Additional arguments for network initialization

    Returns:
        Network instance
    """
    if problem == 'tsp':
        return Net(problem_type='tsp', **kwargs)
    elif problem == 'cvrp':
        return Net(problem_type='cvrp', **kwargs)
    elif problem == 'bpp':
        return NetBPP(**kwargs)
    elif problem == 'mkp':
        m = kwargs.pop('m', 5)
        return NetMKP(m=m, **kwargs)
    elif problem == 'op':
        return NetOP(**kwargs)
    else:
        raise ValueError(f"Unknown problem: {problem}")


# =============================================================================
# Standalone Test Functionality
# =============================================================================

if __name__ == "__main__":
    # Test network creation for all problem types
    print("Testing unified network creation...")

    print("\nTesting TSP network...")
    net_tsp = Net(problem_type='tsp')
    print(f"Net TSP: {net_tsp}")

    print("\nTesting CVRP network...")
    net_cvrp = Net(problem_type='cvrp')
    print(f"Net CVRP: {net_cvrp}")

    print("\nTesting BPP network (wrapper)...")
    net_bpp = NetBPP()
    print(f"NetBPP: {net_bpp}")

    print("\nTesting MKP network (wrapper)...")
    net_mkp = NetMKP(m=5)
    print(f"NetMKP: {net_mkp}")

    print("\nTesting OP network (wrapper)...")
    net_op = NetOP()
    print(f"NetOP: {net_op}")

    print("\nTesting factory function...")
    net_tsp_factory = get_network('tsp')
    net_cvrp_factory = get_network('cvrp')
    net_bpp_factory = get_network('bpp')
    net_mkp_factory = get_network('mkp', m=5)
    net_op_factory = get_network('op')
    print(f"Factory TSP: {net_tsp_factory}")
    print(f"Factory CVRP: {net_cvrp_factory}")
    print(f"Factory BPP: {net_bpp_factory}")
    print(f"Factory MKP: {net_mkp_factory}")
    print(f"Factory OP: {net_op_factory}")

    print("\nAll tests passed!")
