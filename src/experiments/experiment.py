#!/usr/bin/env python

import time
from itertools import product
from ..main import main


# datanames = ['Cora']
# lrs = [0.005, 0.03]
# weight_decays = [0.006 * i for i in range(1, 11)]
#
# file_name = 'lr_reg_Cora.csv'
# file_path = f'/Volumes/GoogleDrive/.shortcut-targets-by-id/107r5K0_qzMzC2U5GN3KdxA907I9lnkmC/Geonwoo Ko/Research/DisenGCN-pytorch/src/experiments/result/{file_name}'
# with open(file_path, 'w') as f:
#     f.write(f'dataname,lr,reg,accuracy,time\n')
#
# for dataname in datanames:
#     for lr in lrs:
#         for reg in weight_decays:
#             Time = time.time()
#             accuracy = main(datadir='../datasets/', dataname='Pubmed', reg=reg, cpu=False, early=10, lr=lr)
#             Time = time.time() - Time
#             with open(file_path, 'a') as f:
#                 f.write(f'{dataname},{lr},{reg},{accuracy},{Time}


def run_experiments(file_dir, hyperpms):
    hyperpm = hyperpms.keys()
    permutation = list(product(*hyperpm.items()))

    for perm in permutation:
        pass


def hyperpm_generator(start, term, len, type='difference'):  # 'rate'
    if type == 'difference':
        return [start + term * i for i in range(len)]
    elif type == 'rate':
        return [start * (term * i) for i in range(1, len + 1)]


def main():
    print('aadasd')

if __name__ == "__main__":
    main()