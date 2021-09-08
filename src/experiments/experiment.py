#!/usr/bin/env python3

import time
import os, sys
import numpy as np
import time
import fire
import gc
import scipy.stats as stats

from itertools import product


def run_grid_search(data_name, device, times, append=True, **hyperpms):
    if device == 'colab':
        file_dir = f'/content/drive/MyDrive/연구실/DisenGCN-pytorch/src/experiments/result'
    elif device == 'windows':
        file_dir = f'/content/drive/.shortcut-targets-by-id/107r5K0_qzMzC2U5GN3KdxA907I9lnkmC/Geonwoo Ko/Research/DisenGCN-pytorch/src/experiments/result'
    else:
        file_dir = f'/Volumes/GoogleDrive/.shortcut-targets-by-id/107r5K0_qzMzC2U5GN3KdxA907I9lnkmC/Geonwoo Ko/Research/DisenGCN-pytorch/src/experiments/result/'

    for data in data_name:
        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        from main import main as m

        hpm_names = list(hyperpms.keys())
        permutation = list(product(*hyperpms.values()))

        file_name = ''
        for hpm_name in hpm_names:
            file_name += f'{hpm_name}_'
        file_name += f'{data}.csv'
        file_path = f'{file_dir}/{data}/{file_name}'
        if not append:
            write_result_csv(file_path, [*hpm_names, 'time', 'accuracy_mean', 'accuracy_std'], 'w')

        for perm in permutation:
            hpm = dict()
            for key, value in zip(hpm_names, perm):
                hpm[key] = value
            print(data, hpm)
            accuracy = []
            Time = time.time()

            for i in range(times):
                accuracy.append(float(m(dataname=data, **hpm)))
                print(accuracy[i])

                for i in range(5):
                    gc.collect()

            Time = time.time() - Time
            write_result_csv(file_path, [*hpm.values(), Time, np.mean(accuracy), np.std(accuracy)])


def run_random_search(data_name, device, sample_num, times, append=True, **hyperpms):
    if device == 'colab':
        file_dir = f'/content/drive/MyDrive/연구실/DisenGCN-pytorch/src/experiments/result'
    elif device == 'windows':
        file_dir = f'/content/drive/.shortcut-targets-by-id/107r5K0_qzMzC2U5GN3KdxA907I9lnkmC/Geonwoo Ko/Research/DisenGCN-pytorch/src/experiments/result'
    else:
        file_dir = f'/Volumes/GoogleDrive/.shortcut-targets-by-id/107r5K0_qzMzC2U5GN3KdxA907I9lnkmC/Geonwoo Ko/Research/DisenGCN-pytorch/src/experiments/result/'

    for data in data_name:
        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        from main import main as m

        hpm_names = list(hyperpms.keys())
        random_hpm = sample_hyperpm(sample_num, **hyperpms)
        
        file_name = ''
        for hpm_name in hpm_names:
            file_name += f'{hpm_name}_'
        file_name += f'{data}.csv'
        file_path = f'{file_dir}/{data}/{file_name}'
        if not append:
            write_result_csv(file_path, [*hpm_names, 'time', 'accuracy_mean', 'accuracy_std'], 'w')

        for perm in random_hpm:

            hpm = dict()
            for key, value in zip(hpm_names, perm):
                hpm[key] = value
            print(data, hpm)
            accuracy = []
            Time = time.time()

            for i in range(times):
                accuracy.append(float(m(dataname=data, **hpm)))
                print(accuracy[i])

                for i in range(5):
                    gc.collect()

            Time = time.time() - Time
            write_result_csv(file_path, [*hpm.values(), Time, np.mean(accuracy), np.std(accuracy)])


def sample_hyperpm(sample_num, **hyperpm):
    sampled = []
    for hpm in hyperpm.values():
        if (isinstance(hpm, stats._distn_infrastructure.rv_frozen)):
            sampled.append(hpm.rvs(sample_num))
        else:
            sampled.append(np.random.choice(hpm, sample_num))
    hpm = [(i) for i in zip(*sampled)]
    return hpm


def write_result_csv(file_path, strings, mode='a'):
    with open(file_path, mode) as file:
        for i in range(len(strings)):
            if (i == len(strings) - 1):
                file.write(f'{strings[i]}\n')
            else:
                file.write(f'{strings[i]},')


if __name__ == '__main__':
    main()
