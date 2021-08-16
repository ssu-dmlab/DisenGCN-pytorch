# DisenGCN-pytorch

This is python implementation of **Disen**tanlged **G**raph **C**onvolutional **N**etwork.      
[(DisenGCN)]((https://jianxinma.github.io/assets/DisenGCN.pdf)  )    


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

| Arguments     | Query Type | Explanation       | Default       | 
| --------------|:------:|-------------------|:-------------:|
| `query-type` | `common` | Query type among [rwr, ppr, pagerank] | `None`|
| `graph-type` | `common` | Graph type among [directed, undirected] | `None` |
| `input-path` | `common` | Input path for a graph | `None`|
| `output-path` | `common` | Output path for storing a query result | `None`|
| `seeds` | `rwr` | A single seed node id | `None`|
| `seeds` | `ppr` | File path for seeds (`str`) or list of seeds (`list`) | `[]`|
| `c` | `common` | Restart probablity (`rwr`) or jumping probability (otherwise) | `0.15`|
| `epsilon` | `common` | Error tolerance for power iteration | `1e-9`|
| `max-iters` | `common` | Maximum number of iterations for power iteration | `100`|
| `handles-deadend` | `common` | If true, handles the deadend issue | `True`|

The value of `Query Type` in the above table is one of the followings:
* `common`: parameter of all of `rwr`, `ppr`, and `pagerank`
* `rwr`: parameter of `rwr`
* `ppr`: parameter of `ppr`
* `pagerank`: parameter of `pagerank`

Note the followings:
* If you want to compute `pagerank` query, then do not need to specify `seeds`.
* For directed graphs, there might be deadend nodes whose outdegree is zero. In this case, a naive power iteration would incur leaking out scores. 
`handles_deadend` exists for such issue handling deadend nodes. With `handles_deadend`, you can guarantee that the sum of a score vector is 1.
Otherwise, the sum would less than 1 in directed graphs. 
The strategy `pyrwr` exploits is that whenever a random surfer visits a deadend node, go back to a seed node (or one of seed nodes), and restart.
See this for the detailed technique of the strategy.
   

[paper](https://jianxinma.github.io/assets/DisenGCN.pdf)   
[raw code](https://jianxinma.github.io/assets/DisenGCN-py3.zip)
