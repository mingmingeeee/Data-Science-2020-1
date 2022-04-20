# 데이터 셋의 구조
# 데이터 셋을 구성하는 구조나 타입을 보여줌

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)
# 너무 작은 수들 출력할 때 소수점 4자리까지만 출력되도록 한 것

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
# radom number generator 이용해서 4 * 10 크기의 2차원 배열을 만듬. 0~1의 실수값

### tensors와 tensor_slices의 다른 점 ###
# 배열 전체를 한 덩어리로 만든 것.
# 3차원:: TensorSpec(shape=(4, 10, 5), dtype=tf.float32, name=None)
# 로 나옴

# element_spec:: 길이가 10인 1차원 배열
# 2차원 배열로 주어지면, 첫번째 크기: 4.
# 4개의 data (data의 개수), 각 data는 길이가 10인 1차원 배열. -> 요게 4개인 배열이 될 수 있음.
# 3차원:: TensorSpec(shape=(10, 5), dtype=tf.float32, name=None)
# 앞에는 4개... 10*5가 크기.

print(dataset1.element_spec)

for elem in dataset1:
    print(elem) # tf.Tensor 객체인 것을 잊지 말긔 !!!!
    print(elem.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]), # 길이가 4인 1차원 배열
     tf.random.uniform([4, 100], maxval = 100, dtype=tf.int32))
     # 2차원 배열. 최대값: 100,, dtype=tf.int32 :: 0~100사이의 정수
     # 개수가 같아야 하기 때문에 첫번째 요소가 같아야 함
)

print(dataset2.element_spec)
# (TensorSpec(shape=(), dtype=tf.float32, name=None) -> 하나의 스칼라값
# TensorSpec(shape=(100,), dtype=tf.int32, name=None)) -> 하나의 길이가 100인 스칼라값
# 첫번 째 구성요소, 두번 째 구성요소

for elem in dataset2:
    print(elem)

for elem1, elem2 in dataset2:
    print(elem1.numpy())
    print(elem2.numpy())

dataset3 = tf.data.Dataset.zip(dataset1, dataset2) # 두 리스트 묶기
# 원래 python zip 함수: 두 리스트에서 하나씩 뽑아서 투플들의 리스트로 만듬
# 모든 dataset의 개수가 4로 같기 때문에 zip으로 합칠 수 있음.
print(dataset3.element_spec)
for elem in dataset3:
    print(elem)

for a, (b, c) in dataset3:
    print('shapes: {x.shape}, {y.shape}, {z.shape}'.format(x=a, y=b, z=c))