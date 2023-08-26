
### Transformer & GPT
----

#### 1. seq2seq

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/5e9363f2-37c7-4d92-81dc-615cd3448d48)

- task 는 주로 입력 문장과 출력 문장의 길이가 다를 경우 사용 (e.g. 번역 및 텍스트 요약)
- seq2seq 는 크게 encoder 및 decoder 의 두 가지 모듈로 구성
- encoder는 입력으로 임베딩 벡터를 순차 입력 받은 뒤 rnn (또는 gru, lstm cell) 의 hidden state를 거쳐 정보를 압축해 마지막에 모든 단어 정보를 압축한 하나의 벡터인 context vector 생성. 즉 encoder 의 최종 목적은 context vector 를 만드는 것이며, context vector는 float으로 이루어진 하나의 벡터이다.
    - rnn과 마찬가지로 encoder의 hidden state 는 전체 공유되어 있으며 입력 문장의 모든 단어 토큰 정보를 요약해 담고 있음을 유의(같은 cell 에 반복해서 입력 및 출력되는 것임)

- 입력 문장 정보가 하나의 context vector 로 압축되면 decoder 로 전달, decoder는 context vector를 받아 번역된 단어를 한 개씩 순차적으로 출력
- decoder에는 초기 입력으로 문장 시작 심볼 <sos>을 입력으로 받으며, 이후 다음에 등장할 확률이 제일 높은 단어 예측함.
- rnn (또는 gru, lstm cell) 을 거치며 hidden state 는 계속 다음 step으로 연결되고 (decoder의 rnn -> rnn) dense-softmax를 거쳐 $y_0$ 이 출력됨. 이후 다시 $y_0$은 rnn의 입력으로 들어가게 됨. 즉 decoder의 rnn은 $y_i$ 와 hidden state를 입력으로 받아 $y_i+1$을 생성함.
    - decoder  다음 등장 단어를 예측할 때 선택 가능한 모든 단어들 중 하나의 단어를 골라 예측해야 하기 때문에 softmax 함수 사용

> problem  
decoder 의 input으로 encoder의 최종 출력인 context vector만을 사용하는 것에 문제점 존재 
1. 하나의 고정 크기 벡터(context vector) 에 모든 정보를 압축하여 정보 손실 발생
2. vanishing gradient 문제 존재

특히 번역 task에서 입력 문장 길이가 길면 번역 품질이 떨어지는 현상 발생            



<br/><br/><br/><br/>



  

#### 2. attention mechanism
- decoder 에서 출력 단어를 고를 때 어떤 encoder 정보를 참고할지 알려주는 데 도움을 주기 위함

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/9963f7bd-19ae-473a-bb64-dbe62f1d6904)

```
Query: 질의, 찾고자 하는 대상
Key: 키, 저장된 데이터 찾을 때 참값
Value: 키에 저장된 데이터
```

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/37a54bf9-9a4e-4653-b044-1aec262ce185)


- attention에서는 주어진 '하나의' query가 어떤 key와 유사한지 모든 key와 각각 비교해 유사도를 얻고, key와 매핑되어 있는 각각의 value를 모두 더해 attention value를 만든다. 즉, **query에 해당하는 dictionary의 key값들이 query와 얼마나 유사한지 계산**한다. 이때 유사도가 가중치 역할을 한다고 볼 수 있음
- 이때 **query는 decoder의 hidden state**가 되며, **encoder의 hidden state가 key와 value**가 됨
- 가장 기본적인 구현 방식으로 comparison은 fully connected 방식 연산, aggregate의 경우 모든 key-value에 대해 벡터의 element-wise multiplication 연산을 한 후 element-wise sum 하여 attention value 생성. 수식은 아래와 같다.
<br/>

  $$Compare(q, k_j) = q*k_j = q^Tk_j$$

  $$Aggregate(c, V) = \sum\limits_{j} c_jv_j$$
<br/>

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/4d593223-b795-4db0-8d9e-64810cd9c738)

- encoder의 hidden state를 (key, value) 로 사용, 위 그림에서 $h_i$는 key와 value로 사용됨

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/707c5277-dc4e-4e65-a8a5-95b433f175ad)

- decoder의 hidden state는 query로 사용, 따라서 위 그림에서 $s_i$는 query로 사용됨


![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/9236d423-f3d0-40ff-8c35-193486fb7698)

- decoder에서 $s_i$라는 query가 입력되고, 그 query와 모든 key값인 $h_i$와 comparison 연산을 통해 유사도를 구한 뒤, value에 해당하는 $h_i$와 각각의 유사도를 곱한 뒤 element-wise sum 하여 attention value $a_i$ 출력.
- 그림에서 사용된 연산을 수식으로 표현하면 아래와 같다. ($k_j$ 와 $v_j$ 자리에 주의할 것)
<br/>

  $$c_i = softmax(s_i^Th_j)$$

  $$a_i = \sum\limits_{j} c_jh_j$$

<br/>

- 여기서 attention value $a_i$는 context vector 라고 불리기도 함 (인코더의 문맥을 포함하고 있기 때문)

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/28406297-7dc2-4df6-a86b-9ee6a946f323)

- 마지막으로 attention value와 decoder의 t시점의 hidden state 를 concat한다. decoder의 hidden state는 rnn에서 받아 연산하여 $s_i -> s_i+1$로 만든다. 그 후 $a_i$와 $s_i+1$을 concat하여 $v_i+1$을 만든다.이후 FC-layer와 softmax를 거쳐 최종 출력인 예측 벡터 $y_i$를 출력한다. 



<br/><br/><br/><br/>



  

#### 3. transformer
- 이전 모델에서 입력 시퀀스의 정보 손실이 발생하는 seq2seq의 문제를 해결하기 위해 보정 용도로 attention이 사용되었는데, attention으로만 encoder와 decoder를 만들어보자는 아이디어

- 이전 seq2seq 구조에서는 encoder와 decoder에서 각각 **1개의 rnn**이 t개의 시점(time step)을 가지는 구조였다면 trnasformer는 encoder와 decoder의 단위가 N개로 구성되어 있음(논문에서는 6개를 default로 설정)

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/84891e1c-b361-445d-96f1-c6332770d643)


**1. input에 대해 Embedding**
w2v과 같은 방법으로 입력 텍스트를 token embedding으로 변환한다. 어텐션 메커니즘은 단어(토큰)의 상대적 위치를 알지 못하기 때문에 텍스트 순서 특징을 모델링하기 위해 각 토큰의 위치 정보가 담긴 positional embedding 을 토큰 임베딩 및 embedding with time signal 값과 더해 encoder block 의 input값으로 사용한다.

<br/>

**2. encoder block**
<br/>
![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/b6f35336-37e0-4834-a8d6-e2bc2492074c)
<br/>
1) self attention <br/>
:: **output: 각 단어의 vector들끼리 서로간의 관계가 얼마나 중요한지 점수화된 vector**  
입력 문장은 위의 과정을 거쳐 크기가 512 인 vector로 변환된 뒤 첫 번째 encoder block의 attention layer로 입력됨. 이후 weight vector $W$를 곱해 q, k, v 벡터를 생성해냄. 이렇게 한 단어에 3가지 vector가 나오게 되고, 특정 연산을 해 attention layer의 output을 만들어냄. <br/>
<br/>
2) Multi-Head Attention
<br/>

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/49bd5f47-7b02-48ea-88f8-2372ebf1e839)

<br/>
attention은 문맥 의미를 잘 파악하는 알고리즘이지만 단독으로 쓸 경우 자기자신의 의미에만 지나치게 집중할 수 있기에 8개의 attention layer를 두고 각각 다른 초기값으로 학습 진행하였음. 각 layer에서 나온 출력을 그대로 합한 뒤, 또다른 weight vector를 곱해 하나의 vector로 취합 --> multi-head attention layer의 최종 출력이 됨.
- 서로 다른 8개의 representation subspace를 생성하여 single 일 때보다 문맥을 더 잘 이해할 수 있게 함. single-head일 경우, 문장 내 한 개의 단어와의 연관성을 중시할 확률이 높지만, multi-head를 활용함으로써 유사 단어에 대한 다양한 후보군 제공.  
<br/><br/>

3) Point-Wise Feed-Forward Netorks
<br/>

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/453eec2e-f86c-4584-8a8e-f4d626530302)
<br/>
*attention layer를 통과한 값들은 FCN을 지나는데, 하나의 인코더 블록 내에서는 다른 문장/단어들마다 정확하게 동일하게 사용되지만 각 인코더 마다는 다른 값을 가지게 된다.* (이해필요)

<br/>

**3. decoder block**

![tm2](https://github.com/hihiee/Session-based-Recsys/assets/45914097/0c11941c-3066-41a0-a3c4-774595d44228)
<br/>
- self-attention 시, 현재 위치의 바로 이전 위치에 대해서만 attention 할 수 있게 하였으며 이렇게 통과된 vector 중 query만 가져오고, keydhk value vector는 인코더 블록의 출력을 가져옴
- 인코더와 마찬가지로 6개 블록 통과 후 FCN과 softmax를 거쳐 학습된 데이터 db중 가장 관련성이 높아보이는 단어를 출력함


![tm3](https://github.com/hihiee/Session-based-Recsys/assets/45914097/851dce02-ce60-4775-ab8d-9787fbe69383)
