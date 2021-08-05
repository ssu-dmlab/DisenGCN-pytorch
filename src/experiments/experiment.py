import time

datanames = ['Cora']
lrs = [0.005, 0.03]
weight_decays = [0.006 * i for i in range(1, 11)]

file_name = 'lr_reg_Cora.csv'
file_path = f'/Volumes/GoogleDrive/.shortcut-targets-by-id/107r5K0_qzMzC2U5GN3KdxA907I9lnkmC/Geonwoo Ko/Research/DisenGCN-pytorch/src/experiments/result/{file_name}'
with open(file_path, 'w') as f:
  f.write(f'dataname,lr,reg,accuracy,time\n')


from main import main

for dataname in datanames:
  for lr in lrs:
    for reg in weight_decays:
      Time = time.time()
      accuracy = main(datadir = '../datasets/', dataname = 'Pubmed', reg = reg, cpu = False, early = 10, lr = lr)
      Time = time.time() - Time
      with open(file_path, 'a') as f:
        f.write(f'{dataname},{lr},{reg},{accuracy},{Time}\n')
