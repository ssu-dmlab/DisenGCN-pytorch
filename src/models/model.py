import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


<<<<<<< HEAD
class DimReduceLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DimReduceLayer, self).__init__()
=======
class SparseInputLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SparseInputLayer, self).__init__()
>>>>>>> 6f77d6b881540f28c3443b62d852cc57d3d6ec58

        weight = torch.zeros((in_dim, out_dim), dtype=torch.float32)
        bias = torch.zeros(out_dim, dtype=torch.float32)

        std = 1. / np.sqrt(out_dim)
        weight = nn.init.uniform_(weight, -std, std)
        bias = nn.init.uniform_(bias, -std, std)
<<<<<<< HEAD
        #weight = nn.init.normal_(weight, 0, std)
        #bias = nn.init.normal_(bias, 0, std)
=======
>>>>>>> 6f77d6b881540f28c3443b62d852cc57d3d6ec58
        self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)

    def forward(self, feat):
        # mm은 broadcasting X
        return torch.mm(feat, self.weight) + self.bias


class RoutingLayer(nn.Module):
<<<<<<< HEAD
    def __init__(self, k, hyperpm):
        super(RoutingLayer, self).__init__()
        self.k = k
=======
    def __init__(self, hyperpm):
        super(RoutingLayer, self).__init__()
        self.k = hyperpm['ncaps']
>>>>>>> 6f77d6b881540f28c3443b62d852cc57d3d6ec58
        self.routit = hyperpm['routit']
        self.tau = hyperpm['tau']

    def forward(self, x, src_trg):
<<<<<<< HEAD
        m, trg, src = src_trg.shape[1], src_trg[0], src_trg[1]
=======

        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
>>>>>>> 6f77d6b881540f28c3443b62d852cc57d3d6ec58

        src, trg = src.long(), trg.long()
        n, d = x.shape
        # k : # of factors(channels), delta_d : embedding dim of each factor
        k, delta_d = self.k, d // self.k
        # factor별 임베딩이 균일하게 나오기 위해, normalize
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        # 각각 neighbor의 factor k에 관한 피쳐
        z = x[src].view(m, k, delta_d)
        c = x
        idx = trg.view(m, 1).expand(m, d)

        for t in range(self.routit):
            p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)
            p = F.softmax(p / self.tau, dim=1)
            p = p.view(-1, 1).repeat(1, delta_d).view(m, k, delta_d)

            weight_sum = (p * z).view(m, d)
            c = c.index_add_(0, trg, weight_sum)
            c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d)
        return c

<<<<<<< HEAD
class DisenConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hyperpm):
        super(DisenConvLayer, self).__init__()

        self.k = out_dim // (hyperpm['ndim'] // hyperpm['init_k'])
        self.pca = DimReduceLayer(in_dim, out_dim)
        self.rout = RoutingLayer(self.k, hyperpm)
        self.dropout = hyperpm['dropout']

    def _dropout(self, x):
        return F.dropout(x, self.dropout, training = self.training)

    def forward(self, x, src_trg):
        x = F.leaky_relu(self.pca(x))
        x = F.normalize(x.view(x.shape[0], self.k, -1), dim = 2).view(x.shape[0],-1)
        x = self.rout(x,src_trg)
        return self._dropout(F.leaky_relu(x))

class DisenGCN(nn.Module):
    def __init__(self, in_dim, nclass, hyperpm):
        super(DisenGCN, self).__init__()

        self.pca = DimReduceLayer(in_dim, hyperpm['ndim'])

        k = hyperpm['init_k']
        d = hyperpm['ndim'] // hyperpm['init_k']
        in_dim = hyperpm['ndim']

        conv_ls = []

        for i in range(hyperpm['nlayer']):
            k -= hyperpm['delta_k']
            out_dim = k * d
            print(f'out_dim = {out_dim}')
            conv_ls.append(DisenConvLayer(in_dim, out_dim, hyperpm))
            in_dim = out_dim

        self.conv_ls = conv_ls
        self.dropout = hyperpm['dropout']
        self.mlp = nn.Linear(in_dim, nclass)

    def _dropout(self, x, dropout):
        return F.dropout(x, dropout, training = self.training)

    def forward(self, feat, src_trg_edges):
        x = self.pca(feat)
        x = F.leaky_relu(x)
        #x = self._dropout(F.leaky_relu(x), self.dropout)
        for conv in tqdm(self.conv_ls, position=0, leave=False, desc='RoutingLayer'):
            x = conv(x, src_trg_edges)
            #x = self._dropout(F.leaky_relu(x), self.dropout)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)
=======

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

        self.mlp = nn.Linear(out_dim, nclass)
        self.dropout = hyperpm['dropout']

    def forward(self, feat, src_trg_edges):
        x = F.leaky_relu(self.pca(feat))
        for conv in tqdm(self.conv_ls, position=0, leave=False, desc='RoutingLayer'):

            x = conv(x, src_trg_edges)

        x = self.mlp(x)

        return F.log_softmax(x, dim=1)

>>>>>>> 6f77d6b881540f28c3443b62d852cc57d3d6ec58
