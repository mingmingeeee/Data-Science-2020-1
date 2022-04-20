# train 훈련을 위한 데이터
# validate 애매모호한 요소들의 테스트...
# train하던 중 검사해서 계속 할 지 그만해도 되는 지 그 심판...?을 해주는 것

# 준비해둔 데이터를 프로그램안으로 읽어올 것

import numpy as np
import os
import cv2

# 크기는 40*60
IMG_HEIGHT = 40
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5
# channel 3개 class 5개
IMAGE_DIR_BASE = 'C:\\Users\\KANGMINJEONG\\Desktop\\animal_images'

# img를 실수형 읽어 다시 return해주는 함수
def load_image(addr):
    img = cv2.imread(addr) # 경로를 읽음
    # img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE img = # if 흑백 img라면
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)# 똑같은 사이즈로 읽을 것이기 때문에.. 너비 * 높이 순인데
    # numpy로 정의될 땐 높이 * 너비
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# numpy가 BGR color폼에서 사용하게 때문
    img = img.astype(np.float32)# 신경망 안에서는 실수. 그래서 모두 실수형으로 변환해줌
    img = (img - np.mean(img))/np.std(img)# 정규화까지 해줌 편차를 맞추기 위해!
    return img

# train 혹은 test 데이터들을 통째로 읽어들이는 일
# 그냥 똑같이 해서 한번 train을 한번은 test를 읽어들임

def load_data_set (set_name):
    data_set_dir = os.path.join(IMAGE_DIR_BASE, set_name)

    image_dir_list = os.listdir(data_set_dir)
    image_dir_list.sort()
    features = [] # 이미지
    # empty Python list
    labels = [] # 0 cat, 1 cow, 2 dog, 3 pig, 4 sheep
    for cls_index, dir_name in enumerate(image_dir_list):
        # enumerate: 2개의 값 반환해줌
        # enumerate() 안에 있는 두가지 값 혹은 여러가지 값을 하나씩 가지게 됨.

        image_list = os.listdir(os.path.join(data_set_dir, dir_name))
        for file_name in image_list:
            if 'png' in file_name or 'jpg' in file_name or 'jpeg' in file_name:
                image = load_image(os.path.join(data_set_dir, dir_name, file_name))
                features.append(image)
                labels.append(cls_index)

    # 전체 데이터를 shuffle 할 것.
    # 미니 배치로 쪼개어버림
    # 랜덤하게 shuffle하지 않으면? --> 규칙 가지고 있을 때에 문제 생김... 편향될 수 있음 !!!

    # 독립적인 요소를 어떻게 같이 셔플할거냐? (feature - label 쌍이기 때문)
    # np.array...!!! 로 만들어진 수열 필요 list XXX
    idxs = np.arange(0, len(features)) # index
    np.random.shuffle(idxs)
    features = np.array(features) # list를 np배열로
    labels = np.array(labels) # list를 np배열로
    shuf_features = features[idxs]
    shuf_labels = labels[idxs]
    # 랜덤으로 셔플하되 똑같은 index로 셔플 :-)
    return shuf_features, shuf_labels
"""
def lead_data_set(set_name): # set_name: 'train' or 'test'
    # 경로: 'IMG_DIR_BASE' + 'train' or 'test'
    data_set_dir = os.path.join(IMAGE_DIR_BASE, 'test')
    # 합쳐서 경로를 만듬
    image_dir_list = os.listdir(data_set_dir)
    # os.listdir은 지정된 디렉터리에 저장된 모든 파일
    # 혹은 서브 디렉토리명의 리스트를 반환함
    # 예: cat, pig, dog, sheep, cow....
    image_dir_list.sort()  # 알파벳 순으로 정렬

    features = [] # img
    labels = [] # 고양이인지? 개인지? ... cat, cow, dog, pig, sheep -> 0, 1, 2, 3, 4

    # ragne(5) = [0, 1, 2, 3, 4]
    for cls_index in range(5):
        image_list = os.listdir(os.path.join(data_set_dir, image_dir_list))
        for file_name in image_dir_list:
            image = load_image(os.path.join(data_set_dir, image_dir_list(cls_index)))
            features.append(image)
            labels.append(cls_index)
"""
"""
    for dir_name in image_dir_list:
        image_list = os.listdir(os.path.join(data_set_dir, dir_name))
        for file_name in image_list:
            image = load_image(os.path.join(data_set_dir, dir_name, file_name))
            # data_set_dir 경로 밑에 dir_name 밑에 file_name
            # 예:: train 밑에 고양이 폴더 밑에 사진들
            features.append(image)
            if 'cat' in dir_name:
                labels.append(0)
            if 'cow' in dir_name:
                labels.append(1)
            if 'dog' in dir_name:
                labels.append(2)
            if 'pig' in dir_name:
                labels.append(3)
            if 'sheep' in dir_name:
                labels.append(4)
            else:
                print("something wrong")
"""

def load_all_data():
    train_images, train_labels = load_data_set('train')
    test_images, test_labels = load_data_set('test')
    return train_images, train_labels, test_images, test_labels
    # 두 개 데이터를 한꺼번에 리턴해주는 함수

# if __name__ == '__main__': 의 역할
# 다른 파일에서 import를 해버리면 test 목적으로 적어놓은 코드도 실행되어버림


train_images, train_labels, test_images, test_labels = load_all_data()
print(train_images.shape)
