import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


# Make initial feature dim to embedding dim
class DimReduceLayer(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DimReduceLayer, self).__init__()

        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feat):
        return torch.mm(feat, self.weight) + self.bias


# Feature disentangle layer
class RoutingLayer(nn.Module):
    def __init__(self, hyperpm):
        super(RoutingLayer, self).__init__()
        self.k = hyperpm['init_k']
        self.routit = hyperpm['routit']
        self.dim = hyperpm['ndim']

    def forward(self, x, src_trg):
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k

        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)  # neighbors' feature
        c = x  # node-neighbor attention aspect factor

        for t in range(self.routit):
            p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)  # update node-neighbor attention aspect factor
            p = F.softmax(p, dim=1)
            p = p.view(-1, 1).repeat(1, delta_d).view(m, k, delta_d)

            weight_sum = (p * z).view(m, d)  # weight sum (node attention * neighbors feature)
            c = c.index_add_(0, trg, weight_sum)  # update output embedding
            if t < self.routit - 1:
                c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d)  # embedding normalize aspect factor
        return c


class DisenGCN(nn.Module):
    def __init__(self, in_dim, nclass, hyperpm):
        super(DisenGCN, self).__init__()

        conv_ls = []
        for i in range(hyperpm['nlayer']):
            conv = RoutingLayer(hyperpm)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)

        self.pca = DimReduceLayer(in_dim, hyperpm['ndim'])
        self.conv_ls = conv_ls
        self.dropout = hyperpm['dropout']
        self.mlp = nn.Linear(hyperpm['ndim'], nclass)

    def _dropout(self, x):
        return F.dropout(x, self.dropout, training=self.training)

    def forward(self, feat, src_trg_edges):
        x = F.relu(self.pca(feat))
        for conv in tqdm(self.conv_ls, position=0, leave=False, desc='RoutingLayer', disable=not self.training):
            x = self._dropout(F.relu(conv(x, src_trg_edges)))
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)
