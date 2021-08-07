import random
import torch
import numpy as np
import time

from loguru import logger


def set_rng_seed(seed):
    if (seed == -1):
        seed = int(time.time())
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
            logger.info(f'{key:20}:{value:>50}')


def sprs_torch_from_scipy(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))