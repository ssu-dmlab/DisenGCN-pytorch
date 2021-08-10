# DisenGCN-pytorch

## 1. 해결되지 않은 문제   
1) pca layer(SparseInputLayer)의 parameter(gradient)가 nan이 되는 문제 
   1) activation function을 relu -> leaky_relu로 변환, nan이 거의 발생하지 않음.
   2) layer(row-wise) normalization을 통해 routingLayer의 output을 normalization
   3) 그러나 layer가 깊어지거나, dropout rate가 커지면 여전히 nan이 드물게 발생


2) 원 논문은 각 DisenConvLayer마다 가중치를 학습시키는데, 코드에서는 처음 pca layer만 가중치를 학습
    1) routing Layer를 통과하다보면, 값이 극단적으로 되어 gradient가 exploding 하는 문제라고 추측
    2) **모델으로 수정하여 각각의 routing Layer에 fc layer를 두어 weight를 학습**
    3) 추가적으로 delta_k를 두어, initial embedding dim(128)에서 k를 줄여 dim을 줄여나감
    4) 새로운 모델은 nan이 발생하지 않음, 그러나 하이퍼파라미터가 코드의 디폴트값으로 적용하면 결과가 좋지 않음.
    5) random search를 이용해 하이퍼파라미터 서치
   

3) random search를 할때, 고려해야 할 점
   1) 완전 생 랜덤 서치는, 좋지않은 결과들만 나온다 (하이퍼 파라미터 범위를 조정)
   2) 모델이 dropout을 적용 안하거나 적게 해야 결과가 좋음(?)
   3) lr decay 필요 여부 / early stopping의 적용 여부 (early 수)


5) ~~Loss function이 잘 작동하지 않음(torch.nn.functional.nll_loss())~~
   1) ~~이미 구현된 함수를 이용하면, softmax된 값들의 loss가 음수가 나옴~~
   2) ~~직접 negative log likelihood loss를 구하면 문제가 없음~~
   3) ***nll_loss는 log_softmax된 값을 인자로 취하는데, softmax된 값을 주었음***


3) ~~src_trg_edges가 tensor에서 ndarray로 바뀌는 문제~~
   1) ~~tonumpy로 바꿔주는 코드가 없는데 어느순간 타입이 바뀜~~
   2) ***데이터 전처리 코드를 모두 data.py에 옮겼더니 문제가 발생하지 않음***

   

[paper](https://jianxinma.github.io/assets/DisenGCN.pdf)   
[raw code](https://jianxinma.github.io/assets/DisenGCN-py3.zip)