# DisenGCN-pytorch

This repository aims to reproduce **DisenGCN** proposed in "Disentangled graph convolutional networks (ICML 2019)". 
We refer to the original code [(link)](https://jianxinma.github.io/assets/DisenGCN-py3.zip) to implement **DisenGCN** on semi-supervised node classification task. 

## Dependencies
We use the following Python packages to implement this. 

* networkx==2.6.2
* torch==1.9.0
* loguru==0.5.3
* fire==0.4.0
* scipy==1.6.2
* tqdm==4.61.2
* numpy==1.21.1  

To install the above packages, type the following on your terminal:
```bash
pip3 install -r requirements.txt
```

## Usage
You can run this project to simply type the following in your terminal:

```bash
python3 -m src.main \
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

| Arguments     | Explanation       | Default       | 
| --------------|-------------------|:-------------:|
| `datadir` | Directory of dataset | `./datasets/` |
| `dataname` | Dataset name (Cora / Citeseer / Pubmed) | `Cora`|
| `bidirect` | Make directed edges to bidirect edges | `True`|
| `seed` | Random seed about model | `None`|
| `nepoch` | Maximum number of epochs to train | `200`|
| `early` | Extra iterations before early-stopping | `None`|
| `lr` | Learning rate | `0.03`|
| `reg` | L2 Regularization (weight decay) rate  | `0.003`|
| `dropout` | Dropout rate (1 - keep probability) | `0.2`|
| `nlayer` | Number of hidden (DisenConv) layers | `5`|
| `init_k` | Initial number of channels in conv layer | `8`|
| `ndim` | Initial embedding dimention (first hidden layer input dim) | `64`|
| `routit` | Number of iterations in routing | `6` |


## Notes on neighborhood sampling
The original code uses neighborhood sampling to make the tensor parallelization easier, but our implmentation does not use the sampling technique. Our implementation is likely to more reflect the influence of all neighbors.


## Evaluation results
We have tested our repository on the semi-supervised (single label) node classification. 
For each dataset, we use the following values in the table for `lr`, `reg`, `dropout`, and `nlayer`. 
For the other hyperparameters, we use default values as above. 

| Hyperparameters |   Cora    | Citeseer   |    Pubmed   |
|----------------|----------|-------------|:-----------:|
|     lr         |    0.04  |     0.02    |      0.03   |
|     reg        |    0.004 |  0.006      |      0.02   |
|     dropout    |    0.45  |   0.2       |      0.05   |
|     nlayer     |     2    |    6        |      5      |
 

We summarize average accuracies with their standard deviations of 10 runs in the following table, and compare them to the results reported in the original paper. 
As shown in the table, our results are slightly better and more stable than the original results. 

|          Accuracy            |             Cora          |          Citeseer        |           Pubmed           |
|------------------------------|---------------------------|--------------------------|:--------------------------:|
|   Original (mean / std)    |     0.811 /  0.001        |  0.785 / 0.016           |            0.692 / 0.012   |
|     This (mean / std)    |     0.814 / 0.001         |  0.795 / 0.005           |           0.705 / 0.009    |
 


## References 
[1] Ma, J., Cui, P., Kuang, K., Wang, X., & Zhu, W. (2019, May). Disentangled graph convolutional networks. In International Conference on Machine Learning (pp. 4212-4221). PMLR.
