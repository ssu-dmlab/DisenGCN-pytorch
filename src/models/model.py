import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseInputLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SparseInputLayer, self).__init__()

        weight = torch.zeros((in_dim, out_dim), dtype=torch.float32)
        bias = torch.zeros(out_dim, dtype=torch.float32)

        std = 1. / np.sqrt(out_dim)
        weight = nn.init.uniform_(weight, -std, std)
        bias = nn.init.uniform_(bias, -std, std)
        self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)

    def forward(self, feat):
        # mm은 broadcasting X
        return torch.mm(feat, self.weight) + self.bias


class RoutingLayer(nn.Module):
    def __init__(self, hyperpm):
        super(RoutingLayer, self).__init__()
        self.k = hyperpm['ncaps']
        self.routit = hyperpm['routit']
        self.tau = hyperpm['tau']

    def forward(self, x, src_trg):
        pass


class DisenGCN(nn.Module):
    def __init__(self, in_dim, out_dim, nclass, hyperpm):
        super(DisenGCN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.pca = SparseInputLayer(in_dim, out_dim)

        self.conv_ls = []
        for i in range(hyperpm['nlayer']):
            conv = RoutingLayer(hyperpm)
            self.conv_ls.append(conv)

        self.mlp = nn.Linear(in_dim, nclass)
        self.dropout = hyperpm['dropout']

    def forward(self, feat, src_trg_edges):

        x = torch.relu(self.pca(feat))
        for conv in self.conv_ls:
            x = conv(x, src_trg_edges)

        x = self.mlp(x)
        return F.softmax(x)
