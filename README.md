# DisenGCN-pytorch

This is python implementation of **Disen**tanlged **G**raph **C**onvolutional **N**etwork.      
[(DisenGCN)](https://jianxinma.github.io/assets/DisenGCN.pdf)    


## Installation
To install this package, type the following:
```bash
pip3 install -r requirements.txt
```

### Requirements
* networkx==2.6.2
* torch==1.9.0
* loguru==0.5.3
* fire==0.4.0
* scipy==1.6.2
* tqdm==4.61.2
* numpy==1.21.1

## Usage
We provide the following simple command line usage:
```bash
python3 ./src/main.py  
```


## Arguments of `DisenGCN`
We summarize the input arguments of `DisenGCN` in the following table:

| Arguments     | Explanation       | Default       | 
| --------------|-------------------|:-------------:|
| `dataname` | Name of the dataset | `Cora`|
| `verbose` | Print the training log | `True` |
| `bidirect` | Make directed edges to bidirect edges | `True`|
| `seed` | Random seed about model | `None`|
| `nepoch` | Max Number of epochs to train | `200`|
| `early` | Extra iterations before early-stopping | `None`|
| `lr` | Learning Rate | `1e-3`|
| `reg` | L2 Regularization(weight decay) rate  | `3e-2`|
| `dropout` | Dropout rate ( 1 - keep probability) | `0.35`|
| `nlayer` | Number of hidden(DisenConv) layers | `4`|
| `init_k` | Initial number of channel in conv layer | `8`|
| `delta_k` | Number of reduction channel per layer  | `0`|
| `ndim` | Initial Embedding Dimention (First hidden layer input dim) | `64`|
| `routit` | Number of iterations in routing | `6` |
| `tau` | Softamx scaling rate | `1.0`|
   

[paper](https://jianxinma.github.io/assets/DisenGCN.pdf)   
[raw code](https://jianxinma.github.io/assets/DisenGCN-py3.zip)
