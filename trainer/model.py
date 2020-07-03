import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.lib.io import file_io
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import gcsfs


"""
"""
# try:
#     # TPU detection. No parameters necessary if TPU_NAME environment variable is
#     # set.
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     print('Running on TPU ', tpu.master())
# except ValueError:
#     tpu = None

# if tpu:
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     STRATEGY = tf.distribute.experimental.TPUStrategy(tpu)
#     print("Num replicas: ", STRATEGY.num_replicas_in_sync)
# else:
#     # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
#     STRATEGY = tf.distribute.get_strategy()


GCS_PATH = "gs://shopee-product-detection-data/data"
CLASSES = 42
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 15

VAL_SPLIT = 0.1
TRAIN_DF = pd.read_csv(GCS_PATH + "/train.csv")
TRAIN_LEN = int(TRAIN_DF["filename"].shape[0] * (1 - VAL_SPLIT))
VAL_LEN = int(TRAIN_DF["filename"].shape[0] * VAL_SPLIT)

IMG_SIZE = (224, 224) # Resnet

OPTIMIZER = "sgd"
LOSS_FN = "categorical_crossentropy"

MODEL_NAME = "rn152.h5"
SAVE_PATH = "gs://shopee-product-detection-data/models/" + MODEL_NAME

"""
"""
def get_dataset():

    TRAIN_DF["filename"] = GCS_PATH + "/train/train/" + TRAIN_DF["category"].apply(lambda x: "{:02d}".format(x)) + "/" + TRAIN_DF["filename"]

    X_train, X_test, y_train, y_test = train_test_split(TRAIN_DF["filename"], TRAIN_DF["category"], train_size = 1 - VAL_SPLIT, random_state = 42)

    train_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X_train), tf.one_hot(y_train, CLASSES))
    )

    test_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X_test), tf.one_hot(y_test, CLASSES))
    )

    return train_data, test_data


"""
"""
def encode(path, label):

    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = image / 255.0
    
    label = tf.cast(label, tf.int8)

    return image, label


"""
"""
def encode_dataset(dataset):

    dataset = dataset.map(encode, num_parallel_calls=AUTOTUNE)

    return dataset


"""
"""
def augment(image,label):

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_brightness(image, max_delta=0.5)

    return image,label


"""
"""
def augment_dataset(dataset):

    dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    
    return dataset


"""
"""
def prepare_dataset(dataset):

    dataset = dataset.shuffle(buffer_size = 128)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.repeat()

    return dataset


"""
"""
def get_model():
        
    fv = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4", input_shape=(*IMG_SIZE, 3))
    
    model = tf.keras.models.Sequential([
            fv,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(CLASSES, activation='softmax')
        ])
    model.build([None, *IMG_SIZE, 3])
    model.compile(loss=LOSS_FN, optimizer=OPTIMIZER, metrics=['accuracy'])

    return model

"""
"""
def train():

    train_data, val_data = get_dataset()

    train_data, val_data = encode_dataset(train_data), encode_dataset(val_data)
    train_data = augment_dataset(train_data)
    train_data, val_data = prepare_dataset(train_data), prepare_dataset(val_data)

    model = get_model()

    hist = model.fit(
        train_data,
        steps_per_epoch = TRAIN_LEN // BATCH_SIZE,
        validation_data = val_data, 
        validation_steps = VAL_LEN // BATCH_SIZE,
        epochs = EPOCHS
    )

    model.save(MODEL_NAME, save_format='h5')

    with file_io.FileIO(MODEL_NAME, mode='rb') as in_file:
        with file_io.FileIO(SAVE_PATH, mode='wb+') as out_file:
            out_file.write(in_file.read())


if __name__ == "__main__":
    train()