# DisenGCN-pytorch
This is python implementation of **Disen**tanlged **G**raph **C**onvolutional **N**etwork.   
This code only implements single node classification task.    
[(DisenGCN)](https://jianxinma.github.io/assets/DisenGCN.pdf)    

## Neighborhood Sampling
Raw code must use neighbor sampling, but this implementation doesn't use that.  
This code is likely to make embeddings by reflecting every neighbors' effects.

## Installation
* networkx==2.6.2
* torch==1.9.0
* loguru==0.5.3
* fire==0.4.0
* scipy==1.6.2
* tqdm==4.61.2
* numpy==1.21.1  

To install this package, write this on your terminal.
```bash
pip3 install -r requirements.txt
```

## Usage
We provide the following simple command line usage:
```bash
python3 ./src/main.py \
        --datadir ··· \
        --dataname ··· \
        --bidirect ··· \
        --seed ··· \
        --nepoch ··· \
        --early ··· \
        --lr ··· \
        --reg ··· \
        --dropout ··· \
        --nlayer ··· \
        --init_k ··· \
        --ndim ··· \
        --routit ··· \
```


## Arguments of `DisenGCN`
We summarize the input arguments of `DisenGCN` in the following table:

| Arguments     | Explanation       | Default       | 
| --------------|-------------------|:-------------:|
| `datadir` | Directory of dataset | `./datasets/` |
| `dataname` | Dataset name (Cora / Citeseer / Pubmed) | `Cora`|
| `bidirect` | Make directed edges to bidirect edges | `True`|
| `seed` | Random seed about model | `None`|
| `nepoch` | Max Number of epochs to train | `200`|
| `early` | Extra iterations before early-stopping | `None`|
| `lr` | Learning Rate | `0.03`|
| `reg` | L2 Regularization(weight decay) rate  | `0.003`|
| `dropout` | Dropout rate ( 1 - keep probability) | `0.2`|
| `nlayer` | Number of hidden(DisenConv) layers | `5`|
| `init_k` | Initial number of channel in conv layer | `8`|
| `ndim` | Initial Embedding Dimention (First hidden layer input dim) | `64`|
| `routit` | Number of iterations in routing | `6` |

##Experiments
### - Single node classification(semi-supervised learning)   


This implementation can obtain the best accuracy in this hyperparameters. If not write in this table, it uses default hyperparameters.

| hyperparameter |   Cora    | Citeseer   |    Pubmed   |
|----------------|----------|-------------|:-----------:|
|     lr         |    0.04  |     0.02    |      0.03   |
|     reg        |    0.004 |  0.006      |      0.02   |
|     dropout    |    0.45  |   0.2       |      0.05   |
|     nlayer     |     2    |    6        |      5      |
 

The results are here.  

|          accuracy            |             Cora          |          Citeseer        |           Pubmed           |
|------------------------------|---------------------------|--------------------------|:--------------------------:|
|      raw code(mean / std)    |     0.811 /  0.001        |  0.785 / 0.016           |            0.692 / 0.012   |
|     this code(mean / std)    |     0.814 / 0.001         |  0.795 / 0.005           |           0.705 / 0.009    |
 




[paper](https://jianxinma.github.io/assets/DisenGCN.pdf)   
[raw code](https://jianxinma.github.io/assets/DisenGCN-py3.zip)
