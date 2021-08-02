import torch

class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, dataset):
        with torch.no_grad():
            model.eval()
