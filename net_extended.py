#!/usr/bin/env python3
"""
Network Architectures for BPP, MKP, OP

Provides neural network architectures for Bin Packing Problem (BPP),
Multi-dimensional Knapsack Problem (MKP), and Orienteering Problem (OP).
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn

# Import base classes from net.py
from net import GNNLayer, EmbNet, ParNet, move_pyg_to_module_device


# =============================================================================
# BPP Network
# =============================================================================

class NetBPP(nn.Module):
    """
    Neural network for Bin Packing Problem.

    Node features: demand (1D)
    Edge features: base score + pheromone (2D)
    """

    def __init__(self, feats=1, edge_feats=2, logit_net=False, grad_checkpointing=False):
        super().__init__()
        if grad_checkpointing:
            self.emb_net = EmbNetCheckpoint(feats=feats, edge_feats=edge_feats)
        else:
            self.emb_net = EmbNet(feats=feats, edge_feats=edge_feats)

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


class EmbNetCheckpoint(nn.Module):
    """Checkpointed version of EmbNet for memory efficiency."""
    def __init__(self, depth=12, feats=1, edge_feats=2, units=32, act_fn='silu', agg_fn='mean'):
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
# MKP Network
# =============================================================================

class NetMKP(nn.Module):
    """
    Neural network for Multi-dimensional Knapsack Problem.

    Node features: prize + weights (m+1D)
    Edge features: base score + pheromone (2D)
    """

    def __init__(self, m=5, feats=6, edge_feats=2, logit_net=False, grad_checkpointing=False):
        super().__init__()
        self.m = m
        if grad_checkpointing:
            self.emb_net = EmbNetCheckpointMKP(feats=feats, edge_feats=edge_feats)
        else:
            self.emb_net = EmbNetMKP(feats=feats, edge_feats=edge_feats)

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


class EmbNetMKP(nn.Module):
    """Embedding network for MKP."""
    def __init__(self, depth=12, feats=6, edge_feats=2, units=32, act_fn='silu', agg_fn='mean', grad_checkpointing=False):
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


class EmbNetCheckpointMKP(nn.Module):
    """Checkpointed version of EmbNetMKP."""
    def __init__(self, depth=12, feats=6, edge_feats=2, units=32, act_fn='silu', agg_fn='mean'):
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
# OP Network
# =============================================================================

class NetOP(nn.Module):
    """
    Neural network for Orienteering Problem.

    Node features: distance to depot + prize (2D)
    Edge features: base score + pheromone (2D)
    """

    def __init__(self, feats=2, edge_feats=2, logit_net=False, grad_checkpointing=False):
        super().__init__()
        if grad_checkpointing:
            self.emb_net = EmbNetCheckpointOP(feats=feats, edge_feats=edge_feats)
        else:
            self.emb_net = EmbNetOP(feats=feats, edge_feats=edge_feats)

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


class EmbNetOP(nn.Module):
    """Embedding network for OP."""
    def __init__(self, depth=12, feats=2, edge_feats=2, units=32, act_fn='silu', agg_fn='mean', grad_checkpointing=False):
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


class EmbNetCheckpointOP(nn.Module):
    """Checkpointed version of EmbNetOP."""
    def __init__(self, depth=12, feats=2, edge_feats=2, units=32, act_fn='silu', agg_fn='mean'):
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
# Factory Function
# =============================================================================

def get_network(problem: str, **kwargs):
    """
    Get the appropriate network for a given problem.

    Args:
        problem: Problem type ('bpp', 'mkp', 'op')
        **kwargs: Additional arguments for network initialization

    Returns:
        Network instance
    """
    if problem == 'bpp':
        return NetBPP(feats=1, edge_feats=2, **kwargs)
    elif problem == 'mkp':
        m = kwargs.pop('m', 5)
        return NetMKP(m=m, feats=m+1, edge_feats=2, **kwargs)
    elif problem == 'op':
        return NetOP(feats=2, edge_feats=2, **kwargs)
    else:
        raise ValueError(f"Unknown problem: {problem}")


if __name__ == "__main__":
    # Test network creation
    print("Testing BPP network...")
    net_bpp = NetBPP()
    print(f"NetBPP: {net_bpp}")

    print("\nTesting MKP network...")
    net_mkp = NetMKP(m=5)
    print(f"NetMKP: {net_mkp}")

    print("\nTesting OP network...")
    net_op = NetOP()
    print(f"NetOP: {net_op}")

    print("\nTesting factory function...")
    net_bpp_factory = get_network('bpp')
    net_mkp_factory = get_network('mkp', m=5)
    net_op_factory = get_network('op')
    print(f"Factory BPP: {net_bpp_factory}")
    print(f"Factory MKP: {net_mkp_factory}")
    print(f"Factory OP: {net_op_factory}")

    print("\nAll tests passed!")
