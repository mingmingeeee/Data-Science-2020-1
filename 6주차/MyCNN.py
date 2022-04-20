import tensorflow as tf
import tensorflow as keras
from HandleInput import *

import numpy as np

from matplotlib import pyplot as plt
print(tf.__version__)

class_names = ['cat', 'cow', 'dog', 'pig', 'sheep']
train_features, train_labels, test_features, test_labels = load_all_data()

plt.figure()
plt.imshow(train_features[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
# 크기 큰 이미지
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_features[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

########## 이 위까지가 이미지 읽어서 불러들여오는 작업 ################

## One-Hot Encoding
## classification에서 사용.
# [0,0,0,1] 과 [1,0,0,0]
# 의 오류값이 같음
# 0 1 2 3 4 하면 같이 그냥 틀린건데도
# 오류값이 커져서 unfair함


# 맨 마지막 출력층에서 쓰는 함수
# softmax
# 증가 함수...
# 항상 모든 함수를 0<ㅇ<1
# 그리고 합은 1에 가까워짐 ㅎㅎ
# -> y1+y2+y3+y4+y5 = 1
# 이런 식으루 ^3^~~~

def create_model():
    # 신경망 구축
    model = tf.keras.Sequential([
        # 층대로 써주면 됨
        tf.keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),  # 첫번째 층
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        # 노드의 개수, activation 은 relu 모드
        # Relu 함수
        # sigmid보다 더 자주 슴
        # relu(x) = max(0,x)
        # 학습이 느려지는...그런 문제를 효과적으로 해결
        tf.keras.layers.Dense(NUM_CLASS, activation=tf.nn.softmax)
        # soft max 모드
    ]) # 네트워크 구축, 지정

    model.compile(optimizer=tf.optimizers.Adam(),
                   # 최소값을 찾아냄... 찾아내기 위해 계속 옆으로 가서 다시 보는데
                   # 얼마나 갈 지 정해줌 -> learning rate
                   loss='sparse_categorical_crossentropy', # 오류...오차: loss
                   # sparse:: 정수로 되어있는 것들을
                   # one hot encodng 한 다음 crossentropy 계산하는 것

                   # cross_entropy:: one-hot-encoded일 때
                   # 이 네트워크가 정답일 때의 확률
                   # 정답일 확률이 커지면 함수값은 작아짐 === 오류 표현
                   # 정답에 가까워지면 작아지고 커지면 커지는
                   metrics = ['accuracy'])
                   # 트레이닝하는 것을 모니터링하기 위해 써주는 것
                   # 이건 뭐다! 가 아니라, 출력 값은
                   # 이럴 확률이 ㅇㅇ%다. 라고 해줌
                   # 그렇기 때문에,,,
                   # 제일 큰 확률이 ㅇㅇ가 되는데
                   # 그게 accurate 하면 accurate...
                   # 이걸 계산함
                   # 원래 라벨하고 일치하는지!!!! -> keras가 다 계산해줌
    return model

model = create_model()
model.summary()

# 04d: 4자리 정수로 저장하라
# checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_path = 'training/mychkp.ckpt'
# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint (checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  period=5)
# 5 epoch마다 저장

# 함수 안에
# 트레이닝은 100 epch 동안 진행 -> 모든 학습 데이터를 쓰면 1 epoch 지나간 것
# one hot encoding 되지 않은 label(
# fit: 학습 수행 AdamOptimizer 이용
model.fit(train_features, train_labels, epochs = 20,
          validation_data = (test_features, test_labels),
          callbacks=[cp_callback])
# 매 epoch 가 종료될 때마다 validation data로 테스트 수행.
# batch size = OO 하면서 지정할 수도 있지만 하지 않으면 그냥... 알아서 됨


# model.save_weights('training/final_weight.ckpt')
# ckpt: check point
# weight들 저장해줌
# model.load_weights('./training/final_weight.ckpt')
# model.load_weights('./training/cp-0005.ckpt')

# test_loss: 얼마나 틀렸는가
# test_acc: 몇퍼센트나 적중했는가?
test_loss, test_acc = model.evaluate(test_features, test_labels)
print("Test accuracy: ", test_acc)

# 얼마나 맞추련가???~ 예상
# 테스트 이미지를 주던가 아예 다른 이미지를 주고 예측해봄
predictions = model.predict(test_features)

# 길이가 5인 배열 -> 2차원 배열을 그냥 1차원 배열인 것처럼 계산
print(predictions[0])
# 최대값
print(np.argmax(predictions[0]))
# 정답
print(test_labels[0])

# acc는 높지만 val_acc 가 작다는 의미
# 우리가 준 이미지는 거의 100%
# 하지만 다른 이미지를 주면 40%....
# 낮아진다는 의미
# 내가 제공한 이미지들의 특성들만 학습했다는 의미... T^T..;;;

# 원래는 한 번에 여러장
my_test_img = load_image('cat000.jpg')
my_test_img = np.reshape(my_test_img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
# 3차원 배열을 4차원으로 바꿔줌 -> 1 * h * w * 3

# 그래서 여러장의 이미지를 받는 곳에
# 한 장의 이미지... 를 넣으면 shape이 안맞아서 넣을 수가 없음
# 위에서 shape바꿔줘야함
my_prediction = model.predict(my_test_img)
print(my_prediction[0])
print(np.argmax(my_prediction[0]))

# Classification(분류): 문제가 정해진 클래스들의 리스트로부터 하나를 선택하는 것이 목적
# Regrssion(회귀): 가격이나 확률같은 연속적인 값을 예측하는 것이 목적
# 자동차의 연비를 예측하는 문제를 다룰 것


# 신경망: 구조 or weight <- 최적의 weight값을 찾아나가는 과정
# weight값들만 알아놓으면... 신경망 만들기 더 편함