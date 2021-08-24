import torch
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from models.model import DisenGCN
from models.eval import MyEvaluator
from loguru import logger


class MyTrainer:
    def __init__(self, in_dim, device='cpu'):
        self.in_dim = in_dim
        self.device = device
        self.evaluator = MyEvaluator(device)

    def train_model(self, dataset, hyperpm):
        epochs = hyperpm['nepoch']

        model = DisenGCN(self.in_dim, dataset.get_nclass(), hyperpm).to(self.device)
        model.acc_list = []
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperpm['lr'], weight_decay=hyperpm['reg'])

        pbar = tqdm(range(epochs), position=1, leave=False, desc='epoch')
        early_count, best_acc, best_model = 0, 0, None

        trn_idx, val_idx, tst_idx = dataset.get_idx()
        _, feat, targ = dataset.get_graph_feat_targ()
        src_trg_edges = dataset.get_src_trg_edges()

        for epoch in pbar:
            model.train()
            optimizer.zero_grad()

            pred_prob = model(feat, src_trg_edges)
            loss = F.nll_loss(pred_prob[trn_idx], targ[trn_idx])
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

            pbar.write(
                f'Epoch : {epoch + 1:02}/{epochs}    loss : {loss:.4f}    trn_acc : {trn_acc:.4f} val_acc : {val_acc:.4f}')
            pbar.update()

            if (hyperpm['early'] != None and early_count == hyperpm['early']):
                break

        model.load_state_dict(best_model)
        return model
