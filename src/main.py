#!/usr/bin/env python3

import fire
import gc

from loguru import logger
from utils import *
from models.train import MyTrainer
from models.eval import MyEvaluator
from data import DataLoader


def run_model(device, hyperpm, dataset):
    in_dim = dataset.feat.shape[1]
    trainer = MyTrainer(device=device,
                        in_dim=in_dim)

    if (hyperpm['early'] == -1):
        logger.info("Train model without early-stopping.")
    else:
        logger.info("Train model with early-stopping.")

    model = trainer.train_model(dataset=dataset, hyperpm=hyperpm)

    evaluator = MyEvaluator(device=device)
    _, _, tst_idx = dataset.get_idx()
    accuracy = evaluator.evaluate(model, dataset, tst_idx)
    # plot_show(model.acc_list, accuracy)
    return accuracy


def main(datadir='datasets/',
         dataname='Cora',
         bidirect=True,
         seed=None,
         nepoch=200,
         early=8,
         lr=0.02,
         reg=0.0036,
         dropout=0.35,
         nlayer=5,
         init_k=8,
         ndim=64,
         routit=6):
    """
    :param datadir: directory of dataset
    :param dataname: name of the dataset
    :param bidirect : Use graph as undirected
    :param seed : seed
    :param nepoch: Max number of epochs to train
    :param early: Extra iterations before early-stopping(default : -1; not using early-stopping) //8
    :param lr: Initial learning rate
    :param reg: Weight decay (L2 loss on parameters)
    :param dropout: Dropout rate (1 - keep probability)
    :param nlayer: Number of conv layers
    :param init_k: Maximum number of capsules per layer
    :param ndim: Output embedding dimensions
    :param routit: Number of iterations when routing
    """

    logger.info("The main procedure has started with the following parameters:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    param = dict()
    param['datadir'] = datadir
    param['dataname'] = dataname
    param['device'] = device
    param['bidirect'] = bidirect
    param['seed'] = seed
    log_param(param)

    logger.info("Training the model has begun with the following hyperparameters:")
    hyperpm = dict()
    hyperpm['nepoch'] = nepoch
    hyperpm['lr'] = lr
    hyperpm['early'] = early
    hyperpm['reg'] = reg
    hyperpm['dropout'] = dropout
    hyperpm['nlayer'] = nlayer
    hyperpm['init_k'] = init_k
    hyperpm['ndim'] = ndim
    hyperpm['routit'] = routit
    log_param(hyperpm)

    dataset = DataLoader(data_dir=param['datadir'],
                         data_name=param['dataname'],
                         bidirection=bidirect,
                         device=device)

    set_rng_seed(random.randint(1, 1000))

    accuracy = run_model(device=device,
                         hyperpm=hyperpm,
                         dataset=dataset)

    logger.info(f"The model has been trained. The test accuracy is {accuracy:.4}")
    return accuracy


if __name__ == '__main__':
    fire.Fire(main)
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
