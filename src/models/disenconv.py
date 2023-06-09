import torch
import torch.nn as nn
import torch.nn.functional as F

class InitDisenLayer(nn.Module):
    def __init__(self, inp_dim, fac_dim, num_factors):
        super(InitDisenLayer, self).__init__()

        self.inp_dim = inp_dim
        self.fac_dim = fac_dim
        self.num_factors = num_factors
        
        self.factor_lins = nn.ModuleList(
            [nn.Linear(self.inp_dim, self.fac_dim) for k in range(self.num_factors)])
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        z = [self.factor_lins[k](X) for k in range(self.num_factors)] 
        z = torch.stack(z, dim=1) # (N, K, D)
        z = F.normalize(torch.relu(z), dim=2)
        return z


# Feature disentangle layer
class RoutingLayer(nn.Module):
    def __init__(self, num_factors, routit, tau):
        super(RoutingLayer, self).__init__()
        self.num_factors = num_factors
        self.routit = routit
        self.tau = tau

    def forward(self, x, edges):
        m, src, trg = len(edges), edges[0], edges[1]
        n, k, delta_d = x.shape

        z = x  # neighbors' feature
        c = x  # node-neighbor attention aspect factor

        for t in range(self.routit):
            p = (z[src] * c[trg]).sum(dim=2, keepdim=True)  # update node-neighbor attention aspect factor
            p = F.softmax(p/self.tau, dim=1) # (M, K, 1)
            weight_sum = (p * z[trg])  # weight sum (node attention * neighbors feature)
            c = c.index_add_(0, src, weight_sum)  # update output embedding
            c = F.normalize(c, dim=2)  # embedding normalize aspect factor
        return c.view(n, -1)

class DisenConv(nn.Module):
    def __init__(self, inp_dim, hid_dim, num_factors, routit, tau):
        super(DisenConv, self).__init__()
        self.init_disen = InitDisenLayer(inp_dim, hid_dim, num_factors)
        self.neigh_rout = RoutingLayer(num_factors, routit, tau)

    def forward(self, X, edges):
        z = self.init_disen(X)
        z = self.neigh_rout(z, edges)
        return z