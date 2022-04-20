# 파일들의 집합으로부터 dataset 생성

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

# 파일의 유형이 img가 아니더라도 사용 가능
# get_file: 경로명을 string으로 return해줌
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)
print(type(flowers_root))
print(flowers_root)
# <class 'pathlib.WindowsPath'> --> 마냥 string이 아닌 path 객체

for item in flowers_root.glob("*"): # *: 임의의 어떤 패턴 값을 찾을 수 있을 때
    # 경로명들 수집. path의 하위 디렉토리에서 임의의 서브 디렉토리나 파일명 수집... (경로까지 수집)
    print(type(item))
    print(item)
    print(item.name)
    # 파일 안의 각각의 데이터들 수집한다는 뜻

# flowers_root.glob("."):: 파일의 형태 like ~~.jpg, ~~.png 이럴 때 유용

# 디렉토리, 하위 디렉토리에 있는 파일들을 생성하는 method, glob과 비슷하게 일종의 패턴을 주면 됨
list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*')) # 모든 파일들 의미

for f in list_ds.take(5):
    print(type(f))
    print(f)
    print(f.numpy()) # data부분만 뽑아냄
    print(f.numpy().decode("utf-8")) # python-string이 됨


### 전처리 함수 ###
# 파일들로 구성된 dataset X, 파일의 경로명으로 구성된 dataset O
def process_path(file_path): # 경로명을 매개변수로 받음 -> 절 . 대 string 아님. tf.Tensor임
  label = tf.strings.split(file_path, os.sep)[-2] # 토크나이징할 string, 딜리미터
  # 딜리미터: 쪼갤 문자 기준 == os.sep == "/"
  # 이미지 파일에 대한 경로명에서 맨 끝에서 두번째 :: 꽃의 종류 :: 라벨 !!!
  return tf.io.read_file(file_path), label # read_file: 경로의 파일을 읽어서 그 파일 리턴
  # tf.Tensor가 값이어야함... -> 모든 리턴값이 다 tf.Tensor여야함.

################ MAP ################
# list_ds data들 뽑아서 map해놓은 함수들에게 주어지고 그 함수가 리턴해주는 값이 labeled_ds 값이 됨
labeled_ds = list_ds.map(process_path)

# img는 배열이 아니라 그냥 바이시퀀스임... 그래서 decode 해줘야함
for image_raw, label_text in labeled_ds.take(1):
  img = tf.image.decode_jpeg(image_raw) # decode
  print(img.shape)
  plt.imshow(img)
  plt.show()

  print(image_raw.numpy()[:100])
  print()
  print(label_text.numpy())


