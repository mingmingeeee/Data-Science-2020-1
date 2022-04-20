# Classification(분류): 문제가 정해진 클래스들의 리스트로부터 하나를 선택하는 것이 목적
# Regrssion(회귀): 가격이나 확률같은 연속적인 값을 예측하는 것이 목적
# 자동차의 연비를 예측하는 문제를 다룰 것

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model', 'Year', 'Origin']


######### GET THE DATA #########
# pd.read_scv 함수::: na: not available: 데이터 존재 X 값은 물음표로 채우기, sep: delimeter
# , kipinitialspace=True: 공백 제거, header: 각 컬럼은... 배열에선 제거하지만 헤더라는 것은 인식함! -> 헤더라는 것을 인식
raw_dataset = pd.read_csv('auto-mpg.data.csv',
                          na_values='?', sep=',', skipinitialspace=True, header=0)

raw_dataset.pop('Comment') # 'Comment' 컬럼 pop 하기...
dataset = raw_dataset.copy()
print(dataset.tail()) # 마지막 몇줄만 출력하기 위한 tail함수
print(dataset.shape)


######### CLEAN THE DATA #########

# 값이 not available이면 True 리턴해줌
print(dataset.isna())

# dataset.isna().sum()
# not available 값이 칼럼당 몇개인지 계산해줌

# not available data들을 알아서 삭제해줌
dataset = dataset.dropna()
print(dataset.shape)

######### One-hot encoding #########

# column 'Origin'만 뽑아내서 mapping 시켜줌
dataset['Origin'] = dataset['Origin'].map({1:'USA', 2:'Europe', 3:'Japan'})
# pd.get_dummies 함수 ::
# 각각의 값을 하나의 변수로 만들어줌
# USA, Europe, Japan 세개의 변수를 만들어서 이 값이면 0 아니면 1으로 만들어줌!
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())

# shuffle 하여 random sampling 함
# 80% :D
train_dataset = dataset.sample(frac=0.8, random_state=0)
# dataset으로부터 train_dataset을 drop하면 나머지 dataset이 test_dataset이 됨
test_dataset = dataset.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

# 저 features로 부터는 'XX' 칼럼 삭제하고
# label은 삭제된 칼럼값을 새 칼럼값으로 가지게 됨
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')



######### Normalization (정규화) #########
# 일정하지 않은 값이니
# 일정한 값으로 바꿔줌 :D... ( 비슷한 값들로 만들어주는 것 )

# 세로로 평균
train_mean = train_features.mean(axis=0)
# 세로로 표준편차
train_std = train_features.std(axis=0)

print(train_mean)
print(train_std)


######### The Normalization layer #########
######### Keras 함수가 정규화해주는 기능을 제공 #########

# 전체 데이터에 대한 평균과 표준편차를 가지고 있어야 함 :)
# 그래서 미리 알려줌!
# Normalization 층을 우리가 알고 있는 데이터로 adapt시켜줌
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
# 평균 0, 표준편차 1인 정규화 분포 :D~~~

print(normalizer.mean.numpy())


######### 신경망 구성 #########

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    # error의 절대값을 loss로 쓰겠다는 뜻
    # 절대값이나 제곱을 하거나...상관은 없지만...
    # 내가 결정해야하는 것 !!! 더 좋은게 무엇인지? 생각해서
    # optimizer: 트레이닝 되어가는 과정을 모니터링하면서
    # learning rate 지정해나가는 함수

    return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# validation: train 중에 20%를 떼어내고
# epoch가 지날 때마다 따로 떼어놨던
# 그 20% 를 뗴어내 성능 평가
history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0,epochs=100
)

# val: 어떤 지점 이후에는 자꾸 over...
# ---> 어느지점에서 멈출 건지를
# fit함수가 자동으로 알아서 해줌
# verbose: 수다스럽기???


# 차트로 그리는 역할을 함
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()

# 예측된 연비의 평균 오차가 test_results이다.
test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(test_results)


######### 예측 #########

# scatter 함수
# 대각선: 100% 정확
# 우리의 예측력을 알 수 있음
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
# test_labels: 정답, test_predictions: 예측
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

# error에 대한 막대그래프
# 25개로 나누어 빈도수를 그림
# 0~0.3에 속하는 애가 11개였고...6보다 큰 애가 한개 있었다...라는 뜻
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()


dnn_model.save('dnn_model')

reloaded = tf.keras.models.load_model('dnn_model')

test_results = reloaded.evaluate(test_features, test_labels, verbose=0)
print(test_results)
