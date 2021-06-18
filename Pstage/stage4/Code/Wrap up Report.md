# Wrap up Report Pstage4



## 1. 점수 및 순위 : Private LB (score) : 1.1343



## 2 . 문제 정의 및 접근 방법

1) Overview

<a href="https://ibb.co/NmLvyB9"><img src="https://i.ibb.co/SdKj3Ct/2021-06-18-16-04-44.png" alt="2021-06-18-16-04-44" border="0"></a>

<a href="https://ibb.co/Sdg7NMY"><img src="https://i.ibb.co/0FLVq64/2021-06-18-16-04-54.png" alt="2021-06-18-16-04-54" border="0"></a>

연산량을 최대한 줄이면서 f1 score를 높이는 방법을 찾아야 한다.



2) 데이터 EDA

<a href="https://ibb.co/jVxpYxZ"><img src="https://i.ibb.co/7gT9DTN/2021-06-18-16-06-12.png" alt="2021-06-18-16-06-12" border="0"></a>

<a href="https://ibb.co/88qzQZN"><img src="https://i.ibb.co/cT0bsnk/2021-06-18-16-06-23.png" alt="2021-06-18-16-06-23" border="0"></a>

class별로 데이터 불균형 문제가 발생했다. 데이터 또한 원본이미지를 crop한 이미지이기 때문에 불필요한 padding이 들어가게 된다.



### 3. 해결 방안

1. 세로로 길거나 가로로 긴 이미지들을 전부 가로로 눞혀놓기 위해 세로로 긴 이미지를 90도 rotation후 평균 비율로 resize(1:1->1:0.723)해주어 macs를 줄이는 효과를 얻어 냈다. 연산량이 약 27.7%감소하는 효과를 나타 냈다.

2. 데이터 불균형 문제를 해결하기 위해 WeightRandomsampler를 사용해 class imbalanced 문제를 완화 했다.

   <a href="https://ibb.co/98CDLny"><img src="https://i.ibb.co/HFkSs7G/2021-06-18-16-16-05.png" alt="2021-06-18-16-16-05" border="0"></a>

3. tune.py로 제공되었던 베이스라인 코드를 이용해 가설을 세워 module의 변화와 search space의 변화를 주어 최적의 모델을 AutoML로 찾았다.

   - shufflenetv2를 돌렸을때 96이미지에 macs 7자리에 f1 0.5정도여서 shufflenet의 전체적인 구조를 tune.py에 따라하면 괜찮을까?

     - ```
       ​```python
       def search_model(trial: optuna.trial.Trial) -> List[Any]:
           """Search model structure from user-specified search space."""
           model = []
           # 1, 2,3, 4,5, 6,7, 8,9
           # TODO: remove hard-coded stride
           global_output_channel = 3
           UPPER_STRIDE = 1
           # Module 1
           """
           moduel 1 은 stride = 2 , reapeat = 1
           """
           m1 = trial.suggest_categorical("m1", ["Conv", "DWConv"])
           m1_args = []
           m1_repeat = 1
           m1_out_channel = trial.suggest_int("m1/out_channels", low=16, high=24, step=8)
           m1_stride = 2
           m1_activation = trial.suggest_categorical(
               "m1/activation", ["ReLU", "Hardswish"]
               )
           if m1 == "Conv":
               # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
               m1_args = [m1_out_channel, 3, m1_stride, None, 1, m1_activation]
           elif m1 == "DWConv":
               # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
               m1_args = [m1_out_channel, 3, m1_stride, None, m1_activation]
           model.append([m1_repeat, m1, m1_args])
           global_output_channel = m1_out_channel
       
           # Maxpooling 
           model.append([1, 'MaxPool', [3,2,1]])
       
           # Module 2
           m2 = trial.suggest_categorical(
               "m2",
               ["Conv",
               "InvertedResidualv2",
               "InvertedResidualv3",
               "MBConv",
               "ShuffleNetV2"
               ]
           )
           '''
           stride = 2 & repeat = 1로 고정 -> 초반에 resolution을 줄여주기 위함
           '''
           m2_args = []
           m2_sub_args = []
           m2_stride = 2
           m2_repeat = trial.suggest_int("m2/repeat", 2, 4)
       
           if m2 == "Conv":
               # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
               m2_c = trial.suggest_int("m2/conv_c", low=global_output_channel, high=40, step=8)
               m2_kernel = 3
               m2_activation = trial.suggest_categorical("m3/activation", ["ReLU", "Hardswish"])
               m2_sub_args = [m2_c*2, m2_kernel, 1, None, 1, m2_activation]
               m2_args = [m2_c*2, m2_kernel, m2_stride, None, 1, m2_activation]
           elif m2 == "InvertedResidualv2":
               # m2_c = trial.suggest_int("m2/v2_c", low=16, high=32, step=16)
               m2_c = trial.suggest_int("m2/v2_c", low=global_output_channel, high=40, step=8)
               m2_t = trial.suggest_int("m2/v2_t", low=1, high=3)
               m2_args = [m2_c, m2_t, m2_stride]
               m2_sub_args = [m2_c,m2_t , 1]
           elif m2 == "InvertedResidualv3":
               m2_kernel = 3
               # m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
               m2_t = round(trial.suggest_float("m2/v3_t", low=1, high=3, step = 0.2),1)
               m2_c = trial.suggest_int("m2/v3_c", low=global_output_channel, high=40, step=8)
               m2_se = trial.suggest_categorical("m2/v3_se", [0, 1])
               m2_hs = trial.suggest_categorical("m2/v3_hs", [0, 1])
               # k t c SE HS s
               m2_args = [m2_kernel, m2_t, m2_c, m2_se, m2_hs, m2_stride]
               m2_sub_args = [m2_kernel, m2_t, m2_c, m2_se, m2_hs, 1]
           elif m2 == "MBConv":
               m2_t = trial.suggest_int("m2/MB_t", low=1, high=3)
               m2_c = trial.suggest_int("m2/MB_c", low=global_output_channel, high=40, step=8)
               m2_kernel = 3
               # m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
               m2_args = [m2_t, m2_c, m2_stride, m2_kernel]
               m2_sub_args = [m2_t, m2_c, 1, m2_kernel]
           elif m2 == "ShuffleNetV2":
               m2_c = global_output_channel * 2
               m2_args = [m2_stride]
               m2_sub_args = [1]
           
                                
           model.append([1, m2, m2_args])      # repeat = 1 로 고정 
           global_output_channel = m2_c
       
           # Module2의 부하
           model.append([m2_repeat, m2, m2_sub_args])  ## max ch 48
       
           # Module 3
           m3 = trial.suggest_categorical(
               "m3",
               
               # "DWConv",
               ["Conv",
               "InvertedResidualv2",
               "InvertedResidualv3",
               "MBConv",
               "ShuffleNetV2"
               ]
               )
           '''
           strde = 1 , repeat = 3 ~5 로 열심히 학습해라
           '''
           m3_args = []
           m3_sub_args = []
           m3_stride = 2
           m3_repeat = trial.suggest_int("m3/repeat", 2, 4)
       
           if m3 == "Conv":
               # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
               m3_c = trial.suggest_int("m3/conv_c", low=global_output_channel, high=96, step=8)
               m3_kernel = 3
               m3_activation = trial.suggest_categorical("m3/activation", ["ReLU", "Hardswish"])
               m3_args = [m3_c*2, m3_kernel, m3_stride, None, 1, m3_activation]
               m3_sub_args = [m3_c*2, m3_kernel, 1, None, 1, m3_activation]
           elif m3 == "DWConv":
               # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
               m3_out_channel = trial.suggest_int("m3/out_channels", low=16, high=128, step=16)
               m3_kernel = trial.suggest_int("m3/kernel_size", low=1, high=5, step=2)
               m3_activation = trial.suggest_categorical("m3/activation", ["ReLU", "Hardswish"])
               m3_args = [m3_out_channel, m3_kernel, m3_stride, None, m3_activation]
           elif m3 == "InvertedResidualv2":
               m3_c = trial.suggest_int("m3/v2_c", low=global_output_channel, high=96, step=8)
               m3_t = trial.suggest_int("m3/v2_t", low=1, high=3)
               m3_args = [m3_c, m3_t, m3_stride]
               m3_sub_args = [m3_c, m3_t, 1]
           elif m3 == "InvertedResidualv3":
               # m3_kernel = trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
               m3_kernel = 3
               m3_t = round(trial.suggest_float("m3/v3_t", low=1, high=3, step = 0.2),1)
               m3_c = trial.suggest_int("m3/v3_c", low=global_output_channel, high=96, step=8)
               m3_se = trial.suggest_categorical("m3/v3_se", [0, 1])
               m3_hs = trial.suggest_categorical("m3/v3_hs", [0, 1])
               m3_args = [m3_kernel, m3_t, m3_c, m3_se, m3_hs, m3_stride]
               m3_sub_args = [m3_kernel, m3_t, m3_c, m3_se, m3_hs, 1]
           elif m3 == "MBConv":
               m3_t = trial.suggest_int("m3/MB_t", low=1, high=3)
               m3_c = trial.suggest_int("m3/MB_c", low=global_output_channel, high=96, step=8)
               m3_kernel = 3
               # trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
               m3_args = [m3_t, m3_c, m3_stride, m3_kernel]
               m3_sub_args = [m3_t, m3_c, 1, m3_kernel]
           elif m3 == "ShuffleNetV2":
               m3_c = global_output_channel
               m3_args = [m3_stride]
               m3_sub_args = [1]
           
           model.append([1, m3, m3_args])
           global_output_channel = m3_c
               
           
           # Module3 부하
           model.append([m3_repeat, m3, m3_sub_args])   ## 96
       
           # Module 4
           m4 = trial.suggest_categorical(
               "m4",
               # "DWConv",
               ["Conv",
               "InvertedResidualv2",
               "InvertedResidualv3",
               "MBConv",
               "ShuffleNetV2",
               ]
               )
           m4_args = []
           m4_sub_args = []
           m4_stride = 2
           m4_repeat = trial.suggest_int("m4/repeat", 2, 4)
       
           if m4 == "Conv":
               # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
               m4_c = trial.suggest_int("m4/conv_c", low=global_output_channel, high=128, step=8)
               m4_kernel = 3
               m4_activation = trial.suggest_categorical("m4/activation", ["ReLU", "Hardswish"])
               m4_args = [m4_c*2, m4_kernel, m4_stride, None, 1, m4_activation]
               m4_sub_args = [m4_c*2, m4_kernel, 1, None, 1, m4_activation]
           elif m4 == "DWConv":
               # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
               m4_out_channel = trial.suggest_int("m4/out_channels", low=16, high=256, step=16)
               m4_kernel = trial.suggest_int("m4/kernel_size", low=1, high=5, step=2)
               m4_activation = trial.suggest_categorical("m4/activation", ["ReLU", "Hardswish"])
               m4_args = [m4_out_channel, m4_kernel, m4_stride, None, m4_activation]
               m4_sub_args = [m4_out_channel, m4_kernel, 1, None, m4_activation]
           elif m4 == "InvertedResidualv2":
               m4_c = trial.suggest_int("m4/v2_c", low=global_output_channel, high=128, step=8)
               m4_t = trial.suggest_int("m4/v2_t", low=2, high=3)
               m4_args = [m4_c, m4_t, m4_stride]
               m4_sub_args = [m4_c, m4_t, 1]
           elif m4 == "InvertedResidualv3":
               m4_kernel = 3
               # trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
               m4_t = round(trial.suggest_float("m4/v3_t", low=2, high=3, step = 0.2),1)
               m4_c = trial.suggest_int("m4/v3_c", low=global_output_channel, high=128, step=8)
               m4_se = trial.suggest_categorical("m4/v3_se", [0, 1])
               m4_hs = trial.suggest_categorical("m4/v3_hs", [0, 1])
               m4_args = [m4_kernel, m4_t, m4_c, m4_se, m4_hs, m4_stride]
               m4_sub_args = [m4_kernel, m4_t, m4_c, m4_se, m4_hs, 1]
           elif m4 == "MBConv":
               m4_t = trial.suggest_int("m4/MB_t", low=2, high=3)
               m4_c = trial.suggest_int("m4/MB_c", low=global_output_channel, high=128, step=8)
               m4_kernel = 3
               # trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
               m4_args = [m4_t, m4_c, m4_stride, m4_kernel]
               m4_sub_args = [m4_t, m4_c, 1, m4_kernel]
           elif m4 == "ShuffleNetV2":
               m4_args = [m4_stride]
               m4_sub_args = [1]
               m4_c = global_output_channel * 2
       
           model.append([1, m4, m4_args])
           global_output_channel = m4_c
       
           # Module 4 부하
           model.append([m4_repeat, m4, m4_sub_args])
       
           # Module 5
           m5 = trial.suggest_categorical(
               "m5",
               
               # "DWConv",
               ["Conv",
               "InvertedResidualv2",
               "InvertedResidualv3",
               "MBConv",
               "ShuffleNetV2",
               ]
               )
           m5_args = []
           m5_stride = 1
           # trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
           if m5_stride == 2:
               m5_repeat = 1
           else:
               m5_repeat = trial.suggest_int("m5/repeat", 2, 4)
       
           if m5 == "Conv":
               # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
               m5_c = trial.suggest_int("m5/conv_c", low=global_output_channel, high=256, step=8)
               m5_kernel = 3
               m5_activation = trial.suggest_categorical("m5/activation", ["ReLU", "Hardswish"])
               m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
               m5_args = [m5_c*2, m5_kernel, m5_stride, None, 1, m5_activation]
           elif m5 == "DWConv":
               # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
               m5_out_channel = trial.suggest_int("m5/out_channels", low=16, high=256, step=16)
               m5_kernel = trial.suggest_int("m5/kernel_size", low=1, high=5, step=2)
               m5_activation = trial.suggest_categorical("m5/activation", ["ReLU", "Hardswish"])
               m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
               m5_args = [m5_out_channel, m5_kernel, m5_stride, None, m5_activation]
           elif m5 == "InvertedResidualv2":
               m5_c = trial.suggest_int("m5/v2_c", low=global_output_channel, high=256, step=8)
               m5_t = trial.suggest_int("m5/v2_t", low=2, high=4)
               m5_args = [m5_c, m5_t, m5_stride]
           elif m5 == "InvertedResidualv3":
               m5_kernel = 3
               # trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
               m5_t = round(trial.suggest_float("m5/v3_t", low=2, high=3, step = 0.2),1)
               m5_c = trial.suggest_int("m5/v3_c", low=global_output_channel, high=256, step=8)
               m5_se = trial.suggest_categorical("m5/v3_se", [0, 1])
               m5_hs = trial.suggest_categorical("m5/v3_hs", [0, 1])
               m5_args = [m5_kernel, m5_t, m5_c, m5_se, m5_hs, m5_stride]
           elif m5 == "MBConv":
               m5_t = trial.suggest_int("m5/MB_t", low=2, high=4)
               m5_c = trial.suggest_int("m5/MB_c", low=global_output_channel, high=256, step=8)
               m5_kernel = 3
               # trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
               m5_args = [m5_t, m5_c, m5_stride, m5_kernel]
           elif m5 == "ShuffleNetV2":
               # m5_c = trial.suggest_int("m5/shuffle_c", low=16, high=32, step=8)
               m5_args = [m5_stride]
               m5_c = global_output_channel
       
           model.append([m5_repeat, m5, m5_args])
           global_output_channel = m5_c
       
           # last layer
           last_dim = trial.suggest_int("last_dim", low=512, high=768, step=128)
           # We can setup fixed structure as well
           model.append([1, "GlobalAvgPool", []])
           model.append([1, "Conv", [last_dim, 1, 1]])
           model.append([1, "FixedConv", [9, 1, 1, None, 1, None]])
       
           return model
       ```

       <a href="https://ibb.co/26PxZfT"><img src="https://i.ibb.co/Sx0bf9z/2021-06-18-16-20-41.png" alt="2021-06-18-16-20-41" border="0"></a>

       위의 논문처럼 stage 2,3,4 부분을 stride 2를 주고 그 뒤에 stride 1을 주어 repeat을 2~4로 주게끔 만드는 방식을 선택했다.

   

4. Class 불균형에 의한 F1 score 하락에 대처하기 위해 다양한 loss 실험 진행후에 0.25\*Crossentropy + F1 loss** 의 조합에서 안정적인 Acc score, 가장 높은 F1 score 확보하여 위와 같은 loss를 선택하게 되었다.

   <a href="https://ibb.co/74LXR45"><img src="https://i.ibb.co/Ryd9Bym/2021-06-18-16-32-04.png" alt="2021-06-18-16-32-04" border="0"></a>

   

   

   

### 4. 다른 전략

1. torchvision에 있는 shufflenet v2 pretrained모델을 이용하여 학습을 시킬 수 있었으나 사용하지 않았다. 직접 AutoML로 모델을 찾고 하이퍼파라미터를 서치하면서 AutoML에 대한 공부를 하고 싶었다.

### 5. 회고

이번 스테이지는 정말 재밌었지만 결과에 대해서 아쉬움이 많이 남았던 스테이지였던것 같다. 이번 스테이지 강의의 핵심내용은 AutoML과 Tensor decomposition이었던것 같은데 pretrained모델을 불러와서 이미지 사이즈를 32x32로 줄여서 실험을 하면 왠만한 NAS모델보다 훨씬 성능이 좋았다. NAS모델을 이용해 퍼블릭 3등을 기록했으나 프라이빗에서 꼴등으로 추락하는 상황이 발생해 많은 아쉬움이 남은 스테이지라고 생각한다. 







