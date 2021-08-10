import os.path
import pickle
import numpy as np
import networkx as nx
import torch

from loguru import logger
from scipy.sparse import csr_matrix
from utils import sprs_torch_from_scipy


class DataLoader:
    def __init__(self, data_dir='datasets/', data_name='Cora', bidirection=True, device='cpu'):
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

        graph = nx.from_dict_of_lists(graph)
        n = graph.number_of_nodes()
        n, d, c = max(n, np.max(tst_idx) + 1), tx.shape[1], ty.shape[1]
        for u in range(n):
            graph.add_node(u)

        allx = allx.tocoo()
        tx = tx.tocoo()
        tx.row = tst_idx[tx.row]
        feat = csr_matrix((np.concatenate([allx.data, tx.data]),
                           (np.concatenate([allx.row, tx.row]), np.concatenate([allx.col, tx.col]))), (n, d))
        feat = sprs_torch_from_scipy(feat).to(self.device)

        targ = np.zeros((n, c))
        targ[trn_idx, :] = y
        targ[val_idx, :] = ally[val_idx, :]
        targ[tst_idx, :] = ty
        # one-hot encoding -> label number
        label = np.argwhere(targ == 1)
        targ = np.zeros(n)
        targ[label[:, 0]] = label[:, 1]
        targ = torch.from_numpy(targ).to(self.device).long()
        logger.info(f'Dataset: {data_name}  Dim: #node X #feature ~ #class = {n} X {d} ~ {c}')

        edges = np.array(graph.edges)
        if bidirection:
            fun = lambda x: (x[1], x[0])
            reverse_edges = np.apply_along_axis(fun, 1, edges)
            edges = np.concatenate((edges, reverse_edges), axis=0)
        src_trg_edges = torch.from_numpy(np.transpose(edges)).to(self.device)

        self.trn_idx, self.tst_idx, self.val_idx = trn_idx, tst_idx, val_idx
        self.graph, self.src_trg_edges, self.feat, self.targ = graph, src_trg_edges, feat, targ
        self.nclass = c


    def get_idx(self):
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ

    def get_src_trg_edges(self):
        return self.src_trg_edges

    def get_nclass(self):
        return self.nclass
