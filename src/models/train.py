import torch
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from models.model import DisenGCN
from models.eval import MyEvaluator
from loguru import logger


class MyTrainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.evaluator = MyEvaluator(device)

    def train_model(self, dataset, hyperpm):
        nepoch = hyperpm['nepoch']
        model = DisenGCN(inp_dim=dataset.inp_dim, num_classes=dataset.get_nclass(), **hyperpm).to(self.device)
        model.acc_list = []
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperpm['lr'], weight_decay=hyperpm['reg'])

        pbar = tqdm(range(nepoch), desc='epoch')
        early_count, best_acc, best_model = 0, 0, None

        trn_idx, val_idx, tst_idx = dataset.get_idx()
        feat, targ = dataset.get_feat_targ()
        src_trg_edges = dataset.get_src_trg_edges()
        ce_loss = torch.nn.CrossEntropyLoss()
        
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()

            logit = model(feat, src_trg_edges)
            loss = ce_loss(logit[trn_idx], targ[trn_idx])
            loss.backward()
            optimizer.step()

            trn_acc = self.evaluator.evaluate(model, dataset, trn_idx)
            val_acc = self.evaluator.evaluate(model, dataset, val_idx)
            model.acc_list.append((trn_acc, val_acc))

            if val_acc > best_acc:
                best_acc, best_model = val_acc, deepcopy(model.state_dict())
                early_count = 0
            else:
                early_count += 1

            pbar.set_description(
                f'Epoch : {epoch + 1:02}/{nepoch}    loss : {loss:.4f}    trn_acc : {trn_acc:.4f} val_acc : {val_acc:.4f}')

            if (hyperpm['early'] != None and early_count == hyperpm['early']):
                break

        model.load_state_dict(best_model)
        return model
