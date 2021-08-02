import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


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
        # m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        # n, d = x.shape
        # k, delta_d = self.k, d // self.k
        # x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        # z = x[src].view(m, k, delta_d)
        # u = x
        # if (not isinstance(trg, torch.Tensor)):
        #     trg = torch.from_numpy(trg)
        # scatter_idx = trg.view(m, 1).expand(m, d)
        # for clus_iter in range(self.routit):
        #     p = (z * u[trg].view(m, k, delta_d)).sum(dim=2)
        #     p = F.softmax(p / self.tau, dim=1)
        #     scatter_src = (z * p.view(m, k, 1)).view(m, d)
        #     u = torch.zeros(n, d, device=x.device)
        #     u.scatter_add_(0, scatter_idx, scatter_src)
        #     u += x
        #     # noinspection PyArgumentList
        #     u = F.normalize(u.view(n, k, delta_d), dim=2).view(n, d)
        # return u
        # src는 neighbor
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        # k : # of factors(channels), delta_d : embedding dim of each factor
        k, delta_d = self.k, d // self.k
        #factor별 임베딩이 균일하게 나오기 위해, normalize
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        # 각각 neighbor의 factor k에 관한 피쳐
        z = x[src].view(m, k, delta_d)
        c = x
        idx = trg.view(m, 1).expand(m, d)

        for t in range(self.routit):
            p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)
            p = F.softmax(p, dim = 1)
            print(p.shape)


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

        x = torch.relu(self.pca(feat))
        for conv in tqdm(self.conv_ls, position=0, leave=False, desc='RoutingLayer'):
            x = conv(x, src_trg_edges)
        x = self.mlp(x)
        return F.softmax(x, dim=1)
