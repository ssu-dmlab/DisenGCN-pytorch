import os.path
import pickle
import numpy as np
import networkx as nx
import torch

from loguru import logger
from scipy.sparse import csr_matrix
from utils import sprs_torch_from_scipy


class DataLoader:
    def __init__(self, data_dir='datasets/', data_name='Cora', device='cpu'):
        """
        Split train, validation, test index and make whole feature about vertices
        Make src_trg_edges which is edges matrix transposed
        """

        self.device = device
        data_tmp = []
        data_path = os.path.join(data_dir, data_name, f'raw/ind.{data_name.lower()}.')

        for suffix in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open(data_path + suffix, 'rb') as file:
                data_tmp.append(pickle.load(file, encoding='latin1'))
        x, y, tx, ty, allx, ally, graph = data_tmp

        with open(data_path + 'test.index') as file:
            tst_idx = [int(i) for i in file.read().split()]
        
        trn_idx = np.array(range(x.shape[0]), dtype=np.int64)
        val_idx = np.array(range(x.shape[0], allx.shape[0]), dtype=np.int64)
        tst_idx = np.array(tst_idx, dtype=np.int64)

        edges = []
        for k,v in graph.items():
            edges += [(k,item) for item in v]
        src_trg_edges = np.transpose(edges)
        src_trg_edges = torch.from_numpy(src_trg_edges).long().to(self.device)

        n, d, c = max(np.max(edges), np.max(tst_idx))+1, tx.shape[1], ty.shape[1]
        
        allx = allx.tocoo()
        tx = tx.tocoo()
        tx.row = tst_idx[tx.row]
        feat = csr_matrix((np.concatenate([allx.data, tx.data]),
                           (np.concatenate([allx.row, tx.row]), np.concatenate([allx.col, tx.col]))), (n, d))
        feat = sprs_torch_from_scipy(feat).to_dense().to(self.device)
        
        targ = np.zeros((n,))
        targ[trn_idx] = y.argmax(axis=1)
        targ[val_idx] = ally.argmax(axis=1)[val_idx]
        targ[tst_idx] = ty.argmax(axis=1)
        targ = torch.from_numpy(targ).long().to(self.device)
        logger.info(f'Dataset: {data_name}  Dim: #node X #feature ~ #class = {n} X {d} ~ {c}')

        self.trn_idx, self.tst_idx, self.val_idx = trn_idx, tst_idx, val_idx
        self.src_trg_edges, self.feat, self.targ = src_trg_edges, feat, targ
        self.inp_dim = self.feat.shape[-1]
        self.nclass = c

    def get_idx(self):
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_feat_targ(self):
        return self.feat, self.targ

    def get_src_trg_edges(self):
        return self.src_trg_edges

    def get_nclass(self):
        return self.nclass
