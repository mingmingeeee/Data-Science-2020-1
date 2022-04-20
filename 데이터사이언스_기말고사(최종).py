### Image Classfication ###
### 총 3680개의 train image, 3669개의 test image가 들어있지만 손상된 image들을 제거하고 나면
### 총 3676개의 train image, 총 3668개의 test image가 남게 된다.
### 그 말은 즉 4(train image쪽)+1(test image쪽)=5(총)개의 손상된 이미지가 존재하였다는 뜻이 된다.

# species
# cat: 1
# dog: 2
# category
import tensorflow as tf
from tensorflow import keras
import glob
from pathlib import Path
import numpy as np
import cv2

## cpu 메모리 문제 때문에..
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_WIDTH=128
IMG_HEIGHT=128
IMG_SIZE=(IMG_WIDTH, IMG_HEIGHT)
NUM_CHANNEL=3
NUM_CLASS=5

### Cat:0, Dog:1
class_names = ['Cat', 'Dog']

### images 디렉토리와 annotations 디렉토리에 저장된 파일들의 경로명 수집
dataset_dir = 'Oxford-IIIT_Pet_Dataset'
image_paths = glob.glob('{}/images/*.jpg'.format(dataset_dir))

### 알파벳순으로 정렬
image_paths.sort()

# print(image_paths)

### 수집한 경로명들 trainval.txt / test.txt 파일 내용에 따라 분류 ###

### trainval.txt: train할 data들
trainval_list = open('{}/annotations/trainval.txt'.format(dataset_dir), 'r')
Lines = trainval_list.readlines() # 한 줄에 한 개씩
train_data_file_name = [line.split()[0] for line in Lines]
trainval_image_paths = [p for p in image_paths if Path(p).stem in train_data_file_name]

### test.txt: test할 data들
test_list = open('{}/annotations/test.txt'.format(dataset_dir), 'r')
Lines2 = test_list.readlines() # 한 줄에 한 개씩
test_data_file_name = [line.split()[0] for line in Lines2]
test_image_paths = [p for p in image_paths if Path(p).stem in test_data_file_name]

### 손상된 jpeg 파일 제거위한 함수
def test_if_valid_jpeg(path):
   img = tf.io.read_file(path)
   image = bytearray(img.numpy())
   if image[0] == 255 and image[1] == 216 and image[-2] == 255 and image[-1] == 217:
     return True
   else:
     return False

### img를 실수형으로 읽어서 다시 return해주는 함수
def load_image(addr):
    img = cv2.imread(addr) # 경로를 읽음
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - np.mean(img))/np.std(img)# 정규화
    return img

### dataset cat, dog로 나누기 ###
### 고양이인 지 개인 지 두번째 (species)를 보면 알 수 있음
### 1이면 cat, 2이면 dog
### load_file_train, load_file_test에서 그 역할을 해줌

### 훈련용 데이터들 로드
def load_file_train(Lines):
    train_data_file_images = []
    train_data_file_labels = []
    for line in range(0, len(Lines)):
        if int((Lines[line].split()[-2])) == 1:
            if test_if_valid_jpeg(trainval_image_paths[line]): # 손상된 파일이 있다면 images, label 리스트에서 제거
                image = load_image(trainval_image_paths[line])
                train_data_file_images.append(image)
                train_data_file_labels.append(0) # labels:1 -> cat이지만 class_names를 위해 0으로 바꿔 저장

        elif int((Lines[line].split()[-2])) == 2:
            if test_if_valid_jpeg(trainval_image_paths[line]): # 손상된 파일이 있다면 images, label 리스트에서 제거
                image = load_image(trainval_image_paths[line])
                train_data_file_images.append(image)
                train_data_file_labels.append(1) # labels:2 -> dog지만 class_names를 위해 1로 바꿔 저장

    idxs = np.arange(0, len(train_data_file_images)) # index
    np.random.shuffle(idxs)
    train_data_file_images = np.array(train_data_file_images) # list를 np배열로
    train_data_file_labels = np.array(train_data_file_labels) # list를 np배열로
    shuf_files = train_data_file_images[idxs]
    shuf_labels = train_data_file_labels[idxs] # 랜덤으로 셔플하되 똑같은 index로 셔플

    return shuf_files, shuf_labels

### 테스트용 데이터들 로드
def load_file_test(Lines):
    test_data_file_images = []
    test_data_file_labels = []
    for line in range(0, len(Lines)):
        if int((Lines[line].split()[-2])) == 1:
            if test_if_valid_jpeg(test_image_paths[line]): # 손상된 파일이 있다면 images, label 리스트에 저장하지 않는다
                image = load_image(test_image_paths[line])
                test_data_file_images.append(image)
                test_data_file_labels.append(0) # labels:1 -> cat이지만 class_names를 위해 0으로 바꿔 저장

        elif int((Lines[line].split()[-2])) == 2:
            if test_if_valid_jpeg(test_image_paths[line]): # 손상된 파일이 있다면 images, label 리스트에 저장하지 않는다
                image = load_image(test_image_paths[line])
                test_data_file_images.append(image)
                test_data_file_labels.append(1) # labels:2 -> dog지만 class_names를 위해 1로 바꿔 저장

    idxs = np.arange(0, len(test_data_file_images)) # index
    np.random.shuffle(idxs)
    test_data_file_images = np.array(test_data_file_images) # list를 np배열로
    test_data_file_labels = np.array(test_data_file_labels) # list를 np배열로
    shuf_files = test_data_file_images[idxs]
    shuf_labels = test_data_file_labels[idxs] # 랜덤으로 셔플하되 똑같은 index로 셔플

    return shuf_files, shuf_labels

### 모든 데이터 로드
def load_all_data():
    train_data_file_images, train_data_file_labels = load_file_train(Lines)
    test_data_file_images, test_data_file_labels = load_file_test(Lines2)
    return train_data_file_images, train_data_file_labels, test_data_file_images, test_data_file_labels

### CNN 신경망 구성 ###
def create_model():
    model = keras.Sequential([keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                                  padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),
                              keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
                              keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
                              keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
                              keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
                              keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
                              keras.layers.Flatten(),
                              keras.layers.Dense(512, activation=tf.nn.relu),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(NUM_CLASS, activation=tf.nn.softmax)
                              ])
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

### 신경망에 train ###
def train(model, train_features, train_labels, val_features, val_labels):
    checkpoint_path = "training_cnn/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=5)

    model.fit(train_features, train_labels, epochs=50, batch_size=300,
              validation_data = (val_features, val_labels),
              callbacks=[cp_callback])

### 훈련을 위한 작업 && 정확도 출력 ###
def train_from_scratch():
    train_features, train_labels, test_features, test_labels = load_all_data()

    model = create_model()

    train(model, train_features, train_labels, test_features, test_labels)

    t_loss, test_acc = model.evaluate(test_features, test_labels)
    print('Test accuracy:', test_acc) # 평균 정확도 출력

if __name__ == '__main__': ### 메인 함수
    train_from_scratch() ### 훈련 시작
