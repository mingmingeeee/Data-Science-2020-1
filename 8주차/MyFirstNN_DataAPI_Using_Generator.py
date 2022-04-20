import tensorflow as tf
from tensorflow import keras


IMG_HEIGHT = 40
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5
IMAGE_DIR_BASE = './animal_images'


img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=10)
train_ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory("{}/train".format(IMAGE_DIR_BASE),
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=10,
                    class_mode='sparse'),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL], [None, ])
)
# print(train_ds.element_spec)
#
# for img, label in train_ds.take(1):
#     print(img, label)

test_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_ds = tf.data.Dataset.from_generator(
    lambda: test_img_gen.flow_from_directory("{}/test".format(IMAGE_DIR_BASE),
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=10,
                    class_mode='sparse'),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL], [None, ])
)
# print(test_ds.element_spec)
#
# train_ds = train_ds.shuffle(100)


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),
        keras.layers.Dense(512, activation=tf.keras.activations.relu),
        keras.layers.Dense(NUM_CLASS, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

    return model


model = create_model()
model.summary()
model.fit(train_ds,
            epochs=20,
            steps_per_epoch=40,
            validation_data=test_ds,
            validation_steps=10,
          )


model.save_weights('training/final_weight.ckpt')
# model.load_weights('./training/cp-0005.ckpt')
print('Training finished...')

test_loss, test_acc = model.evaluate(test_ds, steps=10)
print('Test accuracy:', test_acc)
