
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



#### 2. attention mechanism

![image](https://github.com/hihiee/Session-based-Recsys/assets/45914097/9963f7bd-19ae-473a-bb64-dbe62f1d6904)

- decoder 에서 출력 단어를 고를 때 어떤 encoder 정보를 참고할지 알려주는 데 도움을 주기 위함
```
Query: 질의, 찾고자 하는 대상
Key: 키, 저장된 데이터 찾을 때 참값
Value: 키에 저장된 데이터
```


