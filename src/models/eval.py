import torch
from utils import sprs_torch_from_scipy

class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, dataset):
        with torch.no_grad():
            model.eval()
            _, feat, targ = dataset.get_graph_feat_targ()
            src_trg = dataset.get_src_trg_edges()
            _, _, tst_idx = dataset.get_idx()

            tst_pred_prob = model(feat, src_trg)[tst_idx]
            tst_pred_label = torch.argmax(tst_pred_prob, dim = 1)
            tst_acc = (tst_pred_label == targ[tst_idx]).sum() / len(tst_idx)

            return tst_acc
