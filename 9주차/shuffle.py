# 데이터셋 반복 사용 Repeat -> ex. epoch

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

# 원래 dataset을 지정된 x만큼 혹은 무한히 이어붙인 dataset을 만들어주는구나~ 하고 생각하면 됨

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

# batch의 길이만 수집해서 차트로 그려줌
def plot_batch_sizes(ds):
  batch_sizes = [batch.shape[0] for batch in ds]
  plt.bar(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('Batch number')
  plt.ylabel('Batch size')

# 3개를 이어붙이고 128으로 batching
titanic_batches = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batches)
# --> 가장 마지막 배치를 제외한 나머지 배치들은 128로 길이가 동일함

# 128 batch로 분할, 그것을 3번 반복
# 위와 순서가 바뀜. batch 먼저, repeat 나중에
# 이전은 repeat 먼저, batch 나중에
titanic_batches = titanic_lines.batch(128).repeat(3)
# 마지막만 자투리가 아닌 그 안에서 자투리가 생기고 그것을 3번을 반복
plot_batch_sizes(titanic_batches)

# 각 epoch가 종료될 때마다 추가적인 작업을 원할 때
# 매 epoch마다 dataset iteration 새로 시작
epochs = 3
dataset = titanic_lines.batch(128)

for epoch in range(epochs):
  for batch in dataset:
    print(batch.shape)
  print("End of epoch: ", epoch)


# bacth: 전체 데이터에 대한 랜덤 샘플
# 보통은... shuffle -> batch 이 순서임
# 전처리, map... -> 개별 데이터 or 원하는 데이터들 -> 모든 데이터가 load될 필요 없음
# 그러나 shuffle은 모든 데이터가 섞여야하는 것이기 때문에 모든 데이터가 load되어야함
# 그래서 paython tensorflow는 편법을 씀
# 고정된 크기의 버퍼를 유지하며 그 버퍼에서 랜덤하게 선택하는 방식으로 데이터 shuffle
# 버퍼로 들여와서 랜덤하게 데이터 불러들여와서 ... shuffle
# 버퍼크기 너무 크면 많은 메모리 사용 효율성 ㅂ 랜덤샘플링 ㅇ
# 작으면 랜덤샘플링 ㅂ 실행속도 ㅇㅋ
lines = tf.data.TextLineDataset(titanic_file)
counter = tf.data.experimental.Counter()

# 120 이하 원소
# buffer_size = 100
# 맨처음 버퍼 1~100까지 셔플.
# 하나씩 뽑을 때마다 하나씩 비니까 새로운 데이터 불러들여옴
# batch 20
# 스무개를 뽑는 동안 버퍼 안에는 1~20.
# 엄밀히 말하면 1~119까지.
# 진짜 랜덤샘플은 또 아님... 모든 것을 load하는 것이 아니니.
dataset = tf.data.Dataset.zip((counter, lines))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(20)
print(dataset)
