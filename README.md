# DisenGCN-pytorch
##1. Citation 데이터는 Directed Graph
    만약 Directed Graph로 처리한다면, 최근에 쓰여진 논문(인용횟수가 적음)은 neighbor가 없기 때문에 초기 피쳐를 그대로 가져감  
    그래서 'bidirection'을 추가해서 edge를 양방향으로 만듬

##2. Effect about neighborhood Sampling
    초기 구현 방식은 matrix vectorization을 위해 같은 neighbor Degree를 맞춰줌(sample number).
    만약 어떤 노드의 neighbor Degree가 sample 개수보다 작다면, expand(해당 이웃들의 영향이 과적합될 가능성)
    또한 random sampling이므로, 특정 이웃의 영향이 무시될 가능성도 존재하다.
    따라서 전체 이웃을 다 고려할 수 있는 구현 방식으로 변환(routing Layer)

##3. 해결되지 않은 문제
1) pca layer(SparseInputLayer)의 parameter가 nan이 되는 문제   
   1) epoch를 돌리다보면, pca layer의 parameter가 nan이 됨, lr을 줄여봐도 동일한 현상

2) Citeseer 데이터는 노드의 개수와 test_idx의 최댓값이 다름.
   1) 데이터 전처리 과정에서 tst_idx에는 없는 idx들은 0으로 맞춰줘야함

3) Loss function이 잘 작동하지 않음(torch.nn.functional.nll_loss())
   1) 이미 구현된 함수를 이용하면, softmax된 값들의 loss가 음수가 나오는 이상한 상황...
   2) 직접 negative log likelihood loss를 구하면 문제가 없음



'''   
    :param datadir: directory of dataset   
    :param dataname: name of the dataset   
    :param cpu: Insist on using CPU instead of CUDA   
    :param bidirect : Use graph as undirected   
    :param nepoch: Max number of epochs to train   
    :param early: Extra iterations before early-stopping(default : -1; not using early-stopping)   
    :param lr: Initial learning rate   
    :param reg: Weight decay (L2 loss on parameters)   
    :param drouput: Dropout rate (1 - keep probability)     
    :param nlayer: Number of conv layers   
    :param ncaps: Maximum number of capsules per layer   
    :param nhidden: Number of hidden units per capsule   
    :param routit: Number of iterations when routing   
    :param nbsz: Size of the sampled neighborhood   
    :param tau: Softmax scaling parameter   
'''