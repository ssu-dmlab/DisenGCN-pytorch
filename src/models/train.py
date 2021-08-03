import torch
import torch.nn
import numpy as np
import torch.nn.functional as F

from math import isnan
from tqdm import tqdm
from copy import deepcopy
from models.model import DisenGCN
from utils import sprs_torch_from_scipy


class MyTrainer:
    def __init__(self, in_dim, out_dim, device='cpu'):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    def train_model(self, dataset, hyperpm):
        self.hyperpm = hyperpm
        epochs = hyperpm['nepoch']

        model = DisenGCN(self.in_dim, self.out_dim, dataset.get_nclass(), self.hyperpm).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperpm['lr'], weight_decay=hyperpm['reg'])
        model.train()

        pbar = tqdm(range(epochs), position=1, leave=False, desc='epoch')
        early_count, best_acc, best_model = 0, 0, None

        trn_idx, val_idx, tst_idx = dataset.get_idx()
        _, feat, targ = dataset.get_graph_feat_targ()
        src_trg_edges = dataset.get_src_trg_edges()

        feat = sprs_torch_from_scipy(feat).to(self.device)
        targ = torch.from_numpy(targ).to(self.device)
        src_trg_edges = torch.from_numpy(src_trg_edges).to(self.device)

        for epoch in pbar:
            optimizer.zero_grad()
            pred_prob = model(feat, src_trg_edges)

            pred_label = torch.argmax(pred_prob, dim=1)
            trn_acc = (pred_label[trn_idx] == targ[trn_idx]).sum() / len(trn_idx)
            val_acc = (pred_label[val_idx] == targ[val_idx]).sum() / len(val_idx)

            if val_acc > best_acc:
                best_acc, best_model = val_acc, deepcopy(model.state_dict())
                early_count = 0
            else:
                early_count += 1

            #loss = F.nll_loss(pred_prob[trn_idx], targ[trn_idx])
            loss = -torch.log(pred_prob[(range(len(trn_idx)), targ[trn_idx])]).sum()
            #################debugging#######################
            if(isnan(loss)):
                for name, param in model.named_parameters():
                    print(name, param)
            #################################################
            loss.backward()
            optimizer.step()

            pbar.write(
                f'Epoch : {epoch + 1:02}/{epochs}    loss : {loss:.4f}    trn_acc : {trn_acc:.4f} val_acc : {val_acc:.4f}')
            pbar.update()

            if (early_count == hyperpm['early']):
                break

        model.load_state_dict(best_model)
        return model
