# TFRecord - dataset 생성

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

# tfrecord로 데이터들 읽어옴
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
print(fsns_test_file)

dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
print(dataset.element_spec)

# 구조화되어진 데이터들이라 하나 하나 꺼내올 수 없고
# Example이라는 파싱을 거쳐야함

raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

parsed.features.feature['image/text']

# 텍스트 데이터

# 텍스트 파일에 들어있는 개별의 라인들이 하나의 유닛이 되어짐
# tf.data.TextLineDataset: txt파일에서 하나의 line을 하나의 data unit이 됨 (line들을 봅아냄)
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name) for file_name in file_names
]
print(file_paths)

dataset = tf.data.TextLineDataset(file_paths)
print(dataset.element_spec)

# 5개 라인 take -> 파일들이 순서대로 나옴
for line in dataset.take(5):
  print(line.numpy())
  print(line.numpy().decode("utf-8")) # 모든 string은 utf-8로 디코딩함

# 상황에 따라 순서대로 나올 필요가 없겠찌?
# 이 때 쓰는 것
# file_pahts: 세 개의 파일들의 경로명 -> files_ds
# interleave: textLine dataset 만듬 -> 경로명으로부터 interleave. 제공
# cycle_length=3 -> 세개씩 추출
files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

# 첫번째 파일의 첫번째줄, 두번째 파일의 두번째줄, 세번째 파일의 세번째 줄 ... 으로 추출
for i, line in enumerate(lines_ds.take(9)):
  if i % 3 == 0:
    print()
  print(line.numpy())

# 어떤 라인은 건너뛰거나 생략하고 싶을 때
# csv: table. 첫 줄은 헤더라인
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

# 처음 열 줄
for line in titanic_lines.take(10):
    print(line.numpy())

# 모든 행들은 사용하고 싶지 않고 첫번 째는 지우고 survived만 추출하고 싶다면
def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), "0")
    # substr. 0에서 1 까지. 한 바이트만 꺼낸다는 뜻
    # 1이면 true아니면 0

# skip(1): 한 줄은 건너뛴다
# filter(**): 원래 dataset에서 추출된 각각의 데이터들이 리턴되기 전에 survived에 들어가서 필터링해줌
survivors = titanic_lines.skip(1).filter(survived)

for line in survivors.take(10):
    print(line.numpy())

# CSV 파일로부터
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

# pandas data frame
df = pd.read_csv(titanic_file, index_col=None)
df.head()

# from_tensor_slices: 굉장히.. general한 method.
# 매개변수들에 대해 받을 수 있는 형태가 굉장히 다양하게 dataset 생성 가능
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

print(titanic_slices.element_spec)
# element_spec이 {}이라는 것은,,, 이것은 dict이라는 뜻

for feature_batch in titanic_slices.take(1):
    # 여기서 꺼낸 하나의 element는 하나의 dictionary
  for key, value in feature_batch.items():
      # dictonary의 key-value쌍 꺼내기
    print("  {!r:20s}: {}".format(key, value))


# 고수준 api. make_scv_dataset
# 파일, 배치 사이즈, 라벨 네임 지정 (어떤 컬럼이 내가 예측하고 싶은 값인지 : label_name)
titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived")

# 0 1 1 0 -> 네명에 대한 데이터 (배치 사이즈 4였기 때문에)
for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  print("features:")
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

# 어떤 칼럼들 무시하고 싶다면? -> select_colums를 통해 정해줄 수 있음 (내가 feature로 사용할 컬럼들 지정 가능)

titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived", select_columns=['class', 'fare', 'survived'])

for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))