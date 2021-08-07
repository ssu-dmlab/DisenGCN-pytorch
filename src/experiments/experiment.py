#!/usr/bin/env python3

import time
import os, sys
import numpy as np
import time

from itertools import product

# 상위 경로 path 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from main import main as m


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


def run_experiments(data_name, times, **hyperpms):
    hpm_names = list(hyperpms.keys())
    permutation = list(product(*hyperpms.values()))

    hpm = dict()
    for key, value in zip(hpm_names, permutation):
        hpm[key] = value

    file_dir = f'/Volumes/GoogleDrive/.shortcut-targets-by-id/107r5K0_qzMzC2U5GN3KdxA907I9lnkmC/Geonwoo Ko/Research/DisenGCN-pytorch/src/experiments/result/'

    file_name = ''
    for hpm_name in hpm_names:
        file_name += f'{hpm_name}_'
    file_name += f'{data_name}.csv'
    file_path = f'{file_dir}/{data_name}/{file_name}'

    write_result_csv(file_path, [*hpm_names, 'time', 'accuracy_mean', 'accuracy_std'], 'w')

    for perm in permutation:
        hpm = dict()
        for key, value in zip(hpm_names, perm):
            hpm[key] = value

        accuracy = []
        Time = time.time()
        for i in range(times):
            accuracy.append(m(dataname=data_name, early=10, **hpm))
        Time = time.time() - Time

        write_result_csv(file_path, [*hpm.keys(), Time, np.mean(accuracy), np.std(accuracy)])


def write_result_csv(file_path, strings, mode='a'):
    with open(file_path, mode) as file:
        for i in range(len(strings)):
            if (i == len(strings) - 1):
                file.write(f'{strings[i]}\n')
            else:
                file.write(f'{strings[i]},')


def main():
    datas = ['Cora', 'Citeseer', 'Pubmed']

    hyperpms = dict()
    hyperpms['lr'] = [0.1 ** i for i in range(1, 5)]
    hyperpms['reg'] = [0.1 ** i for i in range(1, 5)]

    run_experiments(datas[0], 3, **hyperpms)

    hyperpms = dict()
    hyperpms['nlayer'] = [i for i in range(1, 7)]
    hyperpms['dropout'] = [0.3 + i * 0.05 for i in range(7)]

    run_experiments(datas[0], 3, **hyperpms)

    hyperpms = dict()
    hyperpms['ncaps'] = [4 * (2 ** i) for i in range(4)]
    hyperpms['nhidden'] = [8, 16, 32]

    run_experiments(datas[0], 3, **hyperpms)


if __name__ == "__main__":
    main()
