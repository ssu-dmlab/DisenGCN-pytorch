import torch
from utils import sprs_torch_from_scipy


class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, dataset, eval_idx):
        with torch.no_grad():
            model.eval()
            feat, targ = dataset.get_feat_targ()
            src_trg = dataset.get_src_trg_edges()

            eval_pred_prob = model(feat, src_trg)[eval_idx]
            eval_pred_label = torch.argmax(eval_pred_prob, dim=1)
            eval_acc = (eval_pred_label == targ[eval_idx]).sum() / len(eval_idx)
            eval_acc = eval_acc.item()

            return eval_acc
