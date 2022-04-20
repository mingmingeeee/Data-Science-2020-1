
# 지금까진 3개의 신경망 구성

# NN -> 동물 이미지 구별 = model.fit
# Regression -> 자동차  = dnn_model.fit
# CNN -> 동물 이미지 구별 -> 더 간단 = model.fit

# 데이터를 어떻게 다루느냐?의 측면에서 같은 것을 지니고 있음
# 학습에 사용될 데이터를 통채로 numpy 배열로 읽어옴
# ---> 비효율적임...

# tf.data 로 이 문제 해결 가능.
# 전처리: 이미지 크기 같게 만들어줌 (데이터 회전.. 대칭.. 새로 만들어서 저장 -> Data augmentation, 정규화, 인코딩)
# 배치 만들기 -> 신경망에 데이터를 배치로 묶어서 한번에 공급.
# 랜덤 셔플링 -> 배치를 위해 필요함 (배치:: 전체 데이터의 랜덤 셔플)
# 랜덤하게 섞은 다음 일정한 크기로 쪼개면... 그것이... 배치 !!!
### 단계적으로, 순차적으로 일어났었음... -> "입력 파이프 라인"



###### tf.data ######
# tf.data를 이용해 파이프라인을 간단하고 효율적이며 일관된 방식으로 구성하게 해줌을 위함
# tf.data.Dataset 클래스 제공

# 데이터셋은 두 가지 방식
# 메모리나 파일에 저장된 데이터 소스로부터 Dataset 객체 생성
# 하나 혹은 그 이상의 Dataset 객체 변환 -> 새로운 데이터셋 생성

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)
# 너무 작은 수들 출력할 때 소수점 4자리까지만 출력되도록 한 것

##### 기본 ######
# Dataset 생성하는 가장 간단한 경우: 모든 데이터가 메모리에 로드되어 있는 경우
# tf.data.Dataset.from_tensors() 혹은 tf.data.Dataset.from_tensor_slices()
# 이 두 함수들로 Dataset 객체 생성 가능
# tf.data.TFRecordDataset() 이용 가능 too.

# Dataset.map() 이용하여 데이터 각각에 대한 변환 또한 수행 가능
# Dataset.batch() 이용하여 여러 원소들을 배치로 묶기도 가능
# Dataset --> iterate 가능. => 데이터 자료구조가 있을 때, 그 데이터들을 하나 하나 꺼낼 수 있음을 의미

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
# => (np.array([...])) np.array 생략하더라도 이게 np.array라는 걸 알 수 있음.
print(dataset)
# <TensorSliceDataset shapes: (), types: tf.int32>
# ():: 0차원

# 데이터들 하나 하나 꺼낼 수 있음
# for문으로 iterator
for elem in dataset: # iterate
    print(elem) # tf.Tensor 객체 값
    print(elem.numpy()) # numpy.array .. data 값임!!
# iterator의 끝이 숨어있음


# 명시적으로 iterator
it = iter(dataset) # 직접... 이터레이터를 만들 수도 있음... (iterator)

while True:
    try:
        print(next(it).numpy())
        # iterator의 끝이 숨어있진 않음
        # next()를 사용해주어야함. exception handling 필요요    except Exception as e:
        break

