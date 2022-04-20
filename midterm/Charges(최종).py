
# labels: charges
# one hot code: sex, smoker, region
# Numer: age, bmi, children

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


######### GET THE DATA #########

dataset = pd.read_csv('insurance.csv',
                          na_values='?', sep=',', skipinitialspace=True, header=0)


######### CLEAN THE DATA #########

dataset = dataset.dropna()

######### One-hot encoding ######### :: sex, smoker, region
# sex - female, male
# smoker - yes, no
# region - southwest, northwest, southeast, northeast
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

# shuffle 하여 random sampling 함
# test_dataset:: 20% -> train_dataset:: 80%
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

# 답
train_labels = train_features.pop('charges')
test_labels = test_features.pop('charges')

######### The Normalization layer #########

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

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

    return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

print(normalizer.mean.numpy())

######### train 반복 -> data가 많으므로 epochs 수도 올려줌 #########
history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0,epochs=540
)

# 차트 그리기
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10000])
  plt.xlabel('Epoch')
  plt.ylabel('Error [charges]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()

print("\n###########################################################\n")
# 예측된 연비의 평균 오차:: test_results
test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
print("evaluate 값: {}".format(test_results))

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)

# test_labels: 정답, test_predictions: 예측
plt.xlabel('True Values [charges]')
plt.ylabel('Predictions [charges]')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [charges]')
_ = plt.ylabel('Count')
plt.show()