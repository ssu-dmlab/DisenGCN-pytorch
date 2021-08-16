#!/usr/bin/env python3

import time
import os, sys
import numpy as np
import time

from itertools import product


def run_experiments(data_name, file_dir, times, **hyperpms):
    print('start experiments')
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from main import main as m

    hpm_names = list(hyperpms.keys())
    permutation = list(product(*hyperpms.values()))

    hpm = dict()
    for key, value in zip(hpm_names, permutation):
        hpm[key] = value


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
        print(hpm)
        
        accuracy = []
        Time = time.time()
        for i in range(times):
            accuracy.append(float(m(dataname=data_name, **hpm, verbose=False)))
            print(accuracy)
        Time = time.time() - Time
        write_result_csv(file_path, [*hpm.values(), Time, np.mean(accuracy), np.std(accuracy)])


def write_result_csv(file_path, strings, mode='a'):
    with open(file_path, mode) as file:
        for i in range(len(strings)):
            if (i == len(strings) - 1):
                file.write(f'{strings[i]}\n')
            else:
                file.write(f'{strings[i]},')



