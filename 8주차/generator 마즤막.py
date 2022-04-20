# Python generator

import tensorflow as tf

import pathlib # 파일 경로
import os # 파일 시스템에 접근할 수 있도록
import matplotlib.pyplot as plt # plotting 위함
import pandas as pd
import numpy as np

# generator: numpy배열로부터 이미지 생성

# data들은 여러 file로 저장되어있음.
# file개수가 너무 많아서 한 번에 메모리에 불러올 수 없을 때
# file의 경로만 파악해서
# data를 요청하면 경로명을 하나 꺼내서 그 경로에 해당하는 파일을 읽어서 리턴해줌

# get_file:: url에 따라 파일을 읽어 저장함
flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

print(type(flowers), flowers) # 값: string, 경로명

flowers_root = os.path.join(os.path.dirname(flowers), 'flower_photos')
# os.path:: 경로명을 다루는 것을 제공해줌
# dirname:: 경로명에서 맨마지막 항목만 제거한 걸 리턴해줌
# join: 갖다붙인 거.. -> 똑같은 경로명 출력됨 어차피
# flower_photos뺐다가 join으로 붙이니까;;;
print(flowers_root)
# 하위 디렉터리 목록 -> "os.listdir"
class_names = os.listdir(flowers_root)
print(class_names)

# 불필요한 파일: LICENSE.txt

### 한 번 삭제 실행해주면 끝 ###
#os.remove(os.path.join(flowers_root, 'LICENSE.txt'))
# class_names = os.listdir(flowers_root)

# glob.glob() => 수집할 파일들 목록
# 특정한 디렉터리, 그 하위 디렉터리 모든 파일명 수집하는 역할을 함
# generator -> 경로명을 읽어서 실제로 거기 해당하는 것을 리턴해줌

# tf.keras.preprocessing.image.ImageDataGenerator::  이미지 프로세싱 자동으로 해주는 패키지
# data augmentation을 위한 다양한 이미지 변환 기능 또한 제공


# GENERATOR
# 어떤 타입의 augmentation을 원하는지 적어줌.
# rescale=1./255 :: 모든 pixel 값에 255를 곱해준다. -> 0~1사이에 정규화를 하겠다.
# rotation_range :: 20도 범위 안에서 랜덤하게 로테이션해서 데이터 증가시키겠다.
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

print(img_gen)

# flow_from_directory:: 실제 저장된 경로명... iterator 리턴해줌
it = img_gen.flow_from_directory(flowers_root)
# Found 3670 images belonging to 5 classes.
# class가 다섯개라 알아서 class가 다섯개다. 라는 것을 판단해줌
print(it)
images, labels = next(it)

# images, labels 찍어줌

# <class 'numpy.ndarray'> float32 (32, 256, 256, 3)
# 32: batch 단위. -> 낱장이 아닌 batch 단위로 제공되는구나..~~! 나머지는, 이미지 크기
# 실제로는... 이미지 크기가 각각인데.. ImageDataGenerator 가 resize도 해줌.
# 이게... 왜... 근데... batch size, image size가 맘대로??? -> 우리가 지정해주지 않았기 때문에 default값으로 그냥 나옴
# label의 개수와 image의 개수가 같아야 함, 길이가 5 -> 클래스도 5개 -> one-hot encoded 되어있구나! 짐작 가능
print(type(images), images.dtype, images.shape)
print(type(labels), labels.dtype, labels.shape)

def gen_it():
    return img_gen.flow_from_directory(flowers_root)

ds = tf.data.Dataset.from_generator(
    gen_it,
    # lambda: 이름없는 함수 -> 딱 한 번 사용하고 마는 함수를 위한... !!!
    # lambda: img_gen.flow_from_directory(flowers_root),
    # 얘를 리턴해주는 함수...랑 똑같음?! 위의 gen_it() 쓰는 거랑 같음
    output_types=(tf.float32, tf.float32),# Tuple -> img들의 output type, label들의 output type
    output_shapes=([32, 256, 256, 3], [32, 5]) # shape... :D
)

print(ds.element_spec)

