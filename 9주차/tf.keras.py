# 개별 데이터의 전처리

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)
print(type(flowers_root))
print(flowers_root)

list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
# 전처리 작업
# 하나의 경로명 split, 꽃의 종류에 해당하는 라벨 추출
def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = parts[-2]

  # 바이 시퀀스로 읽기
  image = tf.io.read_file(filename)
  # 이미지로 decode
  image = tf.image.decode_jpeg(image)
  # 데이터의 타입 실수로 바꿈 -> 신경망에 계산되어져야하기 때문
  image = tf.image.convert_image_dtype(image, tf.float32)
  # 크기 동일시 시킴
  image = tf.image.resize(image, [128, 128])

  return image, label
# 매개변수, 리턴값 둘다 Tensor라는 제약조건 존재
# 사실 여기 사용되어진 Library(전처리 함수) Tensor
# 나중에 Tensorflow와 연계, 계산되어야하니까
# 그러므로 tf제공 아닌 다른 라이브러리 사용하면 안됨 (numpy도 X)


# Dataset.map(f)
# 매개변수, 리턴 값들 타입이 Tensor라는 것 잊지 말긔


file_path = next(iter(list_ds))
image, label = parse_image(file_path)

# 매개변수: image, label -> image 출력하는 함수
def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')

show(image, label)

# 전처리 작업하도록
images_ds = list_ds.map(parse_image)

for image, label in images_ds.take(2):
  show(image, label)


# 이미지 회전 or resize~~...나만의 변환...
# 임의의 파이썬 로직 정의하고 싶을 때
#tf.image augmentation: 오직 90도회전만 제공
import scipy.ndimage as ndimage

def random_rotate_image(image):
    # 회전할 각도 랜덤하게 정의함
  image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
  return image
    # return type, dtype은 np.array

# 정상적으로 동작하는지 확인
image, label = next(iter(images_ds))
image = random_rotate_image(image)
show(image, label)

# 함수 사용위해 shapes와 types 명시
def tf_random_rotate_image(img, label):
  im_shape = img.shape
  # 어떤 걸 리턴해줄 지 모르니까 [imgae,]라는 배열에 리턴해주는 값들을 넣어놓음
  # shape, tf.Tensor들에게 적용시켜주고픈 함수들,,, img, type
  # 매개변수 타입, 리턴 타입을 adaptor처럼 맞춰줌
  [image,] = tf.py_function(random_rotate_image, [img], [tf.float32])
  # shape 세팅
  image.set_shape(im_shape)
  return image, label

# dataset에 불러옴
rot_ds = images_ds.map(tf_random_rotate_image)

# 그렇게 제대로 실행 가능
for image, label in rot_ds.take(2):
  show(image, label)

### 개별 데이터의 전처리 ###
# 매우 매우 중요한 파트임 !!!!

# tf.keras -> tf.data
# 고수준 api. 그래서 간결함

train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255.0
labels = labels.astype(np.int32)

fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(fmnist_train_ds, epochs=2)

model.fit(fmnist_train_ds.repeat(), epochs=2, steps_per_epoch=20)
# 무한대라 steps_per_epoch=20이면 20 돌면 한 에포크가 끝났구나~ 생각하게 함

loss, accuracy = model.evaluate(fmnist_train_ds)
print("Loss :", loss)
print("Accuracy :", accuracy)
# repeat X : 유한

loss, accuracy = model.evaluate(fmnist_train_ds.repeat(), steps=10)
print("Loss :", loss)
print("Accuracy :", accuracy)
# repeat O : 무한 -> steps 지정

predict_ds = tf.data.Dataset.from_tensor_slices(images).batch(32)
result = model.predict(predict_ds, steps = 10)
print(result.shape)
# predict: label은 필요 X

result = model.predict(fmnist_train_ds, steps = 10)
print(result.shape)
# label을 포함하는 데이터셋을 넘겨주더라도 label들은 그냥 무시되므로 별 문제가 없다.
