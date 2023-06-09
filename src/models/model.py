import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .disenconv import DisenConv


class DisenGCN(nn.Module):
    def __init__(self, 
                 inp_dim,
                 hid_dim,
                 init_k,
                 delta_k,
                 routit,
                 tau,
                 dropout,
                 num_classes,
                 num_layers,
                 **kwargs):
        super(DisenGCN, self).__init__()

        self.conv_layers = nn.ModuleList()
        k = init_k
        for l in range(num_layers):
            fac_dim = hid_dim // k
            self.conv_layers.append(DisenConv(inp_dim, fac_dim, k, routit, tau))
            inp_dim = fac_dim * k
            k -= delta_k   
        
        self.dropout = dropout
        self.classifier = nn.Linear(inp_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def _dropout(self, X):
        return F.dropout(X, p=self.dropout, training=self.training)
        
    def forward(self, X, edges):
        Z = X
        for disen_conv in tqdm(self.conv_layers, position=0, leave=False, desc='DisenConv', disable=not self.training):
            Z = disen_conv(Z, edges)
            Z = self._dropout(torch.relu(Z))
        Z = self.classifier(Z)
        return Z
