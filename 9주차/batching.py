# Batching

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np


# 배칭: dataset을 배치에 따라 꺼내는 것

# range(100): 0~99까지의 정수로 구성된 dataset 생성
inc_dataset = tf.data.Dataset.range(100)
# range(100): 0~-99까지 -1감수시켜가며 정수로 구성된 dataset 생성
dec_dataset = tf.data.Dataset.range(0, -100, -1)
# 두개의 dataset을 zip해서 하나의 dataset 생성
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))

# 각각은 두개의 tf.Tensor
for ele in dataset.take(4):
  print(ele)

# batch(3): 이 데이터셋에서 데이터를 세개씩 묶어 공급
batched_dataset = dataset.batch(3)
print(batched_dataset.element_spec)
# TensorSpec: 하나의 투플...
# 현재 투플: 두개의 구성요소 가짐
# 배치 사이즈를 3이라고 했음에도 Why shape=(None, )일까?
# 이유: 내가 아무리 정해도 실제로는 가변길이가 될 수 있음. (뭐... 균형이 맞지 않을 때라던가?)
# batch(3):: 투플이 3개인 길이가 3인 배열 X, 첫번째 투플 길이 3, 두번째 투플 길이 3 이 만들어져서 출력
# 투플 3개 모으면 투플이 3개가 되는 게 아니라
# 길이가 3인 배열이 두개 모인 투플이 리턴되는 것...

# dataset: feature, label (일반적으로)
# feature들끼리만 모아서 하나의 집합
# label들끼리만 모아서 하나의 집합
# ...이 이유라고 볼 수 있음 (두개 모인 투플의 이유)

# 내가 지정한 크기의 배치 단위로 공급됨
for batch in batched_dataset:
  print(batch)
  # print([arr.numpy() for arr in batch])

print(batched_dataset.element_spec)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([99], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([-99], dtype=int64)>)
# 총 data는 100개인데 batch를 3으로 하니 당연 하나가 남겠쥬??
# 길이는 가변적이다: shape = None

# 마지막 배치를 생략하여 모든 배치들이 동일한 크기를 갖도록 할 때 -> drop_remainder 매개변수를 True로 설정
batched_dataset = dataset.batch(7, drop_remainder=True)
print(batched_dataset.element_spec)
# 마지막꺼 버려버림...


# Batching with padding
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
# lambda를 이용하여 함수 만들어줌 -> 매개변수 x 받으면
# tf.cast(x, tf.int32) -> x를 정수로 캐스팅
# tf.fill(dd, x) dd라는 Tensor를 만들어 값을 x로 채워라

for batch in dataset.take(4):
  print(batch.numpy())

print()

# dataset이 가변적인 길이를 가진다고 했을 때,
# padded_batch 사용하여 임의의 값을 추가하여 각각의 길이를 다 똑같이 만들어줌]
# 가장 긴 길이 맞춰줌
dataset_padded = dataset.padded_batch(batch_size=4)
for batch in dataset_padded.take(2):
  print(batch.numpy())
  print()

# padded_shapes=(10,) -> 길이을 고정 길이로 설정 가능
# padding value따로 지정 가능
dataset_padded_fixed = dataset.padded_batch(4, padded_shapes=(10,))
for batch in dataset_padded_fixed.take(2):
  print(batch.numpy())
  print()