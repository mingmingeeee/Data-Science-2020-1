# Python generator

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

# tf.data.Dataset -> python generator 사용해 만들 수 있음

def count(stop):
    i = 0
    while i < stop:
        yield i # pause -> 호출되면 여기서 pause되고, 다시 호출될 때 다시 시작됨
        # 함수는 return, generator은 yield
        i += 1

it = count(5)
# iterator it (코드는 실행되지 않고 generator에 대한 iterator를 리턴해줌)
print(it)
print(next(it))
print(next(it))
print(next(it))
print(next(it))

for n in count(5):
    print(n)

"""
함수는 한 번 호출되는 순간 정보 기억못하고 한 번 호출하면 끝. 이지만
generator는 자신의 state를 계속 기억하고 있음. (local 정보 계속 기억 가능)
"""

##### 다음 data를 생성해서 주는 경우에 적합한 것이 "generator"임 ####
##### generator를 통해 dataset ####
#### 미리 알고 있는 게 아니라 요청이 올 때마다 만들어줄 경우에 용이함 ###


# args=[25] -> count는 0~24까지 된다는 의미
# output_types = tf.int32 -> int32:: 정수 생성, output_shapes = (): 정수 하나이니 스칼라타입으로 -> ()
# output_shapes 지정해주는 것이 좋음 ... -> rank 정해주는 일
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = ())
print(ds_counter.element_spec)

# 처음 세개의 정수만 꺼내기 --> take(3)
for ele in ds_counter.take(3):
    print(ele.numpy())

# 트레이닝은 batch_size같은 걸 고정해서 할 수 있으나, 실제로 예측할 때는
# 가변이어도 동작하도록 해야하기 때문에 정확한 shape은 고정하지 않는 것이 좋음 -> 길이는 "None"이다. 라고 명시
# 그. 러. 나...!!! rank가 가변인 것은 일반적으로... 허용하지 않음!!! rank(축의 개수)는 fix되어야 함
# 1차원일 때 2차원일 때 3차월일 때 -> fix
# 그럴 때의 길이 -> 가변

def gen_series():
    i = 0
    while True:
        size = np.random.randint(0, 10) # 0~10사이 랜덤으로 정수 선택
        yield i, np.random.normal(size=(size, ))
        # random vector yield. -> 길이: 0~9사이
        i += 1

# 길이 랜덤하게 선택됨
for i, series in gen_series():
    print(i, ":", str(series))
    if i > 5:
        break

ds_series = tf.data.Dataset.from_generator(
    gen_series, # 매개변수 없기 때문에 생략
    output_types=(tf.int32, tf.float32), # output_type: 투플
    # 첫번째 i: 정수, 두번째: normal -> float
    output_shapes=((), (None,))
    # 투플. 스칼라 -> (), 가변길이 벡터 -> (None)
)
# types: int or float 자료형...
# shapes: 길이

#### 신경망에서는 동일한 길이의 데이터들이 필요함 -> padded_batch 변환 이용 ####
ds_series_batch = ds_series.shuffle(20).padded_batch(batch_size=10)
# 방금 만들었던 series -> 데이터들을 랜덤하게 셔플링한 다음 꺼냄
# batch_size = 10 -> 10개씩 꺼냄

# ids: 인덱스 값들의 묶음 sequence_batch: 값들의 묶음
# ids를 batch_size가 정해줌 -> padded_batch가 하는 일
# 제일 긴 애를 기준으로 삼아 얘보다 작은 애들은 0을 강제로 심어줌
# ===> 하.지.만.... 배치의 배치가 안맞을 수 있음.
# 단순히 제일 긴 애를 기준으로 삼는 거기 때문에.
ids, sequence_batch = next(iter(ds_series.batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())
print()


# padded_batch할 때 내가 원하는 size를 지정해줌
# ==> padded_shapes를 지정해줌.
# ==> padding_values = (0, -1, 0) ---> -1로 채워랏~! 라는 뜻.
ds_series_batch_fixed_size = ds_series.shuffle(20).padded_batch(batch_size=5, padded_shapes=((), (10,)), padding_values=(0, -1.0))
ids, sequence_batch = next(iter(ds_series_batch_fixed_size))
print(ids.numpy())
print()
print(sequence_batch.numpy())