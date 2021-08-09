import torch
import torch.nn
import numpy as np
import torch.nn.functional as F

from math import isnan
from tqdm import tqdm
from copy import deepcopy
from models.model import DisenGCN



class MyTrainer:
    def __init__(self, in_dim, device='cpu'):
        self.in_dim = in_dim
        self.device = device

    def train_model(self, dataset, hyperpm):
        torch.autograd.set_detect_anomaly(True)
        self.hyperpm = hyperpm
        epochs = hyperpm['nepoch']

        model = DisenGCN(self.in_dim, dataset.get_nclass(), self.hyperpm).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperpm['lr'], weight_decay=hyperpm['reg'])
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                                               lr_lambda=lambda epoch: 0.95 ** epoch,
        #                                               last_epoch=-1,
        #                                               verbose=False)
        model.train()

        pbar = tqdm(range(epochs), position=1, leave=False, desc='epoch')
        early_count, best_acc, best_model = 0, 0, None

        trn_idx, val_idx, tst_idx = dataset.get_idx()
        _, feat, targ = dataset.get_graph_feat_targ()
        src_trg_edges = dataset.get_src_trg_edges()


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

            loss = F.nll_loss(pred_prob[trn_idx], targ[trn_idx])
            #loss = -torch.log(pred_prob[(range(len(trn_idx)), targ[trn_idx])]).sum()
            #################debugging#######################
            # if(True):
            #     for name, param in model.named_parameters():
            #         if(name[0:3] == 'pca'):
            #             print(name, param)
            #################################################
            loss.backward()
            optimizer.step()
            # scheduler.step()

            pbar.write(
                f'Epoch : {epoch + 1:02}/{epochs}    loss : {loss:.4f}    trn_acc : {trn_acc:.4f} val_acc : {val_acc:.4f}')
            pbar.update()

            if (early_count == hyperpm['early']):
                break

        model.load_state_dict(best_model)
        return model
