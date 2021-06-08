# Wrap up Report Pstage1



## 1. 점수 및 순위 : Private LB (f1 score) : 0.7236 , 98등



## 2 . 문제 정의 및 접근 방법

1) Overview

​	<a href="https://ibb.co/q7XZVrP"><img src="https://i.ibb.co/CmNrDHT/2021-06-08-23-34-37.png" alt="2021-06-08-23-34-37" border="0"></a>

<a href="https://ibb.co/xsdqJY4"><img src="https://i.ibb.co/CtDKh6r/2021-06-08-23-35-38.png" alt="2021-06-08-23-35-38" border="0"></a>

<a href="https://ibb.co/BtncSd5"><img src="https://i.ibb.co/37R4LG9/2021-06-08-23-36-49.png" alt="2021-06-08-23-36-49" border="0"></a>

한 사람당 마스크 착용 5장, 이상하게 착용 1장, 미착용 1장으로 데이터 불균형이 심할 것으로 예상되었다.

2) 데이터 EDA

<a href="https://imgbb.com/"><img src="https://i.ibb.co/d4QcyJk/2021-06-08-23-40-25.png" alt="2021-06-08-23-40-25" border="0"></a>

각 클래스 별로 Train데이터의 개수를 확인 했을 때 데이터 불균형 문제가 심했다. 마스크 미착용 혹은 이상하게 착용의 데이터가 현저히 부족 했고 2,5,8,11,14,17 번 클래스들은 전부 60세 이상으로 분류되는 노년층이었는데 노년층의 데이터 역시 부족했다.

<a href="https://ibb.co/RbKt0Rw"><img src="https://i.ibb.co/fFWZpjw/2021-06-08-23-43-17.png" alt="2021-06-08-23-43-17" border="0"></a>

Age, Gender, Mask착용 유무별로 나눠서 확인해본 결과이다. 따라서 위와 같은 불균형 문제 때문에 외부 데이터(마스크 미착용, 연령대를 60세이상들 위주로) 가져와서 추가하기로 결정하게 되었다. 또한 분류시에 60세 이상으로 클래스를 분류하지 않고 58세로 줄여서 데이터 불균형 문제를 완화할 수 있었다.



### 3. 해결 방안

1. 외부데이터의 이미지 이름들이 나이를 표시 했고 따로 기존 데이터셋을 가지고 나이만 분류하여 resnet모델로 학습을 시킨 후에 그 학습 된 모델로 외부 데이터를 inference하여 대회의 class에 맞게 labeling을 했다. 외부 데이터를 추가 했으나 특정 class(이상하게 마스크 착용)데이터들이 부족하여 여전히 데이터 불균형 문제가 남아있었다.

2. 데이터 불균형 문제를 해결하기 위해 Focal Loss에 각 class별로 분포를 확인해 weight를 주는 전략을 선택하게 되었다.

   ```
   def get_class_weight(label):
       label_unique, count = np.unique(label, return_counts=True)
       return [1-c/sum(count) for c in count]
   
   result = get_class_weight(train_index)
   result=torch.Tensor(result)
   result = result.to(device)
   
   class FocalLoss(nn.Module):
       def __init__(self, weight=None,
                    gamma=2., reduction='mean'):
           nn.Module.__init__(self)
           self.weight = weight
           self.gamma = gamma
           self.reduction = reduction
   
       def forward(self, input_tensor, target_tensor):
           log_prob = F.log_softmax(input_tensor, dim=-1)
           prob = torch.exp(log_prob)
           return F.nll_loss(
               ((1 - prob) ** self.gamma) * log_prob,
               target_tensor,
               weight=self.weight,
               reduction=self.reduction)
   ```

3. 다음으로 optimizer는 Adam을 사용했을 때보다 SGD를 사용했을 때 성능이 더 좋게 나와 SGD를 사용하게 되었고 lr_scheduler를 CosineAnnealingWarmRestarts를 사용하게 되었다.

   

4. backbone model은 pretrained된 efficientnet-b3를 사용하게 되었다. 데이터가 충분하다고 판단되어 overfitting 현상이 잘 일어나지 않을 거라는 판단하에 dropout을 0.5 밑으로 주면서 실험 했다.

5. classifier를 붙힐 때 ReLU함수에 맞는 가중치 초기화를 사용하는 전략을 사용했는데 실제로 성능이 약간은 올라갔다.

   ```
   def init_weights(m):
       if type(m) == nn.Linear:
           torch.nn.init.kaiming_uniform_(m.weight)
           
           
   model = EfficientNet.from_pretrained('efficientnet-b3')
   num = model._fc.in_features
   classifier = nn.Sequential(
                              nn.Dropout(p=0.4),
                              nn.Linear(num,18),
                              nn.ReLU()
                             )
   model._fc = classifier
   model.apply(init_weights)
   model.to(device)
   ```



### 4. 실패 했던 전략

1. 한창 토론 글에 mask 따로, gender 따로, age 따로 학습 시킨 뒤에 최종 적으로 결과를 합산하면 성능이 좋을 것이라는 논쟁이 있어서 적용해보았으나 성능이 오히려 떨어졌다.

   <a href="https://imgbb.com/"><img src="https://i.ibb.co/ZXRW1wq/2021-06-09-00-00-23.png" alt="2021-06-09-00-00-23" border="0"></a>

2. 랜덤 피어세션 시간에 train과 val 데이터를 나누지 않고 전부 train에 넣어 학습을 시키면 성능 향상이 많이 된다는 얘기를 듣고 적용해보았으나 역시 성능이 떨어지는 현상이 발생했다.

3. score를 높히기 위해 다양한 augmentation을 적용해 보았으나 성능이 오히려 떨어져 이미지를 224 X 224로 Resize후에 150 X 150으로 CenterCrop만 적용했을 때 성능이 올라갔다.



### 5. 회고

첫 스테이지였던 만큼 재밌었고 아쉬운점도 굉장히 많았던것 같다. 특히 ""이러면 이러지 않을까?"라는 추측에 의해 모델을 학습시켰고 그 결과 어떤 부분에서 성능 향상이 이루어졌고 어떤 부분때문에 성능이 하락했다는 점을 인지 하지 못한채 스테이지를 마무리 하게 되었던 것 같다. 가설을 세운뒤에 가설이 맞는지 아닌지 차근차근 판단하면서 하나하나 추가해 나가는 전략을 세웠더라면 좀더 높은 성능을 기대 할 수 있었을 것 같다. 하지만 그 가설을 세우기에 이론적으로 많이 부족하다는 점을 깨닫게 되었다.







