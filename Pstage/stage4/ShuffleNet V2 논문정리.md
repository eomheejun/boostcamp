# ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

1. Introduction

   1. 이전 까지 계산복잡도 측정을 위해 FLOPs를 사용했다.

      - FLOPs(float-point operations) 와 MACs의 차이
        - MACs : a*x+b를 하나의 연산(operation)으로 처리를 하고 이 연산이 몇번 실행되었는지를 세는 것
        - FLOPs : 덧셈, 곱셈을 하나 하나의 연산으로 보고 몇 번 실행되었는지 세는 것 
        - 일반적으로 1MAC = 2 FLOPs
        - 참고 (https://bongjasee.tistory.com/3)

   2. 그러나 FLOPs는 속도나 latency 측면에서는 간접적인 지표다.

   3. 

      <a href="https://ibb.co/vcshnVZ"><img src="https://i.ibb.co/Mg8f0ZM/2021-06-02-23-54-57.png" alt="2021-06-02-23-54-57" border="0"></a>

      위의 표는 ARM과 GPU에서의 경량화 모델 속도 측정 결과다 각 모델들의 FLOPs가 같지만 batch/sec를 보면 차이가 난다. 따라서 FLOPs만 고려하기에는 무리가 있다.

   4. FLOPS와 속도와 차이가 나는 이유는 첫번 째로 몇 가지 중요한 요소(memory access cost, degree of parallelism)들이 FLOPs  연산에서 고려되지 않는다.  두번 째 이유는 GPU와 ARM같은 플랫폼에 따라 같은 FLOPs라 하더라도 러닝타임이 다르다. tensor decomposition을 사용하면 GPU상에서 FLOPs는 75%정도 줄어들지만 실제 속도는 늦어졌다.

      <a href="https://ibb.co/7JmS9Tp"><img src="https://i.ibb.co/GvSdh1M/2021-06-03-00-06-17.png" alt="2021-06-03-00-06-17" border="0"></a>

      대부분의 연산은 Conv가 차지하지만 다른 요소들의 실행 시간 역시 고려를 해야한다. 따라서 FLOPs와 각 플랫폼 마다에서 평가하는 방식으로 고려를 해야 한다. 

2. Practical Guidelines for Efficient Network Design

   위와 같은 이유로 효율적인 아키텍처를 설계하기 위해 4가지 가이드 라인을 제안 했다.

   1. G1) Equal channel width minimizes memory access cost (MAC)

      요즘 모델들은 depthwise separable convolutions 를 사용한다. 그 중 pointwise convolution(1x1 conv)이 연산량의 대부분을 차지한다. 1x1 conv의 파라미터는 입력 채널(c1), 출력 채널(c2) 두 가지 이므로 이 두가지를 바꿔가면서 실험 한다.

      <a href="https://ibb.co/CmpJwtZ"><img src="https://i.ibb.co/4KwsFfG/2021-06-03-00-20-25.png" alt="2021-06-03-00-20-25" border="0"></a>

      input channel과 output channel이 1:1일때 모든 플랫폼에서 속도가 빨랐다.

   2. G2) Excessive group convolution increases MAC.

      최근 아키텍처의 핵심인 group convolution을 사용하면 FLOPs는 줄어들지만 MACS는 늘어나게 된다. 

      <a href="https://ibb.co/c3s9pH2"><img src="https://i.ibb.co/ZXbrwPc/2021-06-03-00-48-45.png" alt="2021-06-03-00-48-45" border="0"></a>

      표에서 그룹number가 늘어남에 따라 속도가 현저히 줄어들게 된다. 

   3. G3) Network fragmentation reduces degree of parallelism.

      GoogleNet시리즈와 Auto-generated architectures에서 "multipath"구조가 사용된다. 많은 소규모 연산자(“fragmented operators”)가 큰 연산자를 사용하게 되는데 예를들어 NASNET-A에서 여러 conv, pooling 연산이 하나의 block에 포함되어 있고 반대로 ResNet은 규칙적인 구조로 이루어져 있다. 

      <a href="https://ibb.co/0qNB7pk"><img src="https://i.ibb.co/TcXKfDZ/2021-06-03-00-59-43.png" alt="2021-06-03-00-59-43" border="0"></a>

      소규모 연산자를 사용하면 정확도는 좋아지지만 GPU환경에서는 

      효율성이 떨어진다. 

      

   

   

