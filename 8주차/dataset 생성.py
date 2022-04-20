# 데이터 셋의 생성

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)
# 너무 작은 수들 출력할 때 소수점 4자리까지만 출력되도록 한 것

# Numpy 배열로부터 생성

train, test = tf.keras.datasets.fashion_mnist.load_data() # sample dataset

images, labels = train
images = images/255 # img/255 :: 0~1사이의 수. 정규화 됨

print(type(images))
print(len(labels))

# 두 배열의 투플로 set을 만들면,,, 여기서 하나 저기서 하나 뽑아서 만든 투플 들...
# 이미지 한 장이 28*28
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print(dataset.element_spec)

for img, label in dataset.take(3): # dataset에서 몇개만 뽑아보기... -> dd.take(3)
    plt.imshow(img)
    plt.show()
    print(label.numpy())

# List or ndarray or tf.Tensor객체도 모두 from_tensor_slices에 매개변수로 들어갈 수도 있음.