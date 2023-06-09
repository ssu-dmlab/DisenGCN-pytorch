import random
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from loguru import logger


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info(f'{in_key:20}:{in_value:>50}')
        else:
            if (value != None):
                logger.info(f'{key:20}:{value:>50}')


def sprs_torch_from_scipy(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))


def plot_show(acc_list, tst_accuracy):
    epochs = len(acc_list)
    trn_acc = [i for i, _ in acc_list]
    val_acc = [v for _, v in acc_list]

    x = range(1, epochs + 1)
    plt.xlim(0, epochs + 5)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.plot(x, trn_acc, 'o-', color='g',
             label='train')
    plt.plot(x, val_acc, 'o-', color='b',
             label='validation')
    plt.scatter(x[-1], tst_accuracy, color='r', label='test')
    plt.legend()
    plt.savefig('result.png')
