import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.lib.io import file_io
import efficientnet.tfkeras as efn

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import gcsfs


"""
"""
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    STRATEGY = tf.distribute.experimental.TPUStrategy(tpu)
    print("Num replicas: ", STRATEGY.num_replicas_in_sync)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    STRATEGY = tf.distribute.get_strategy()



GCS_PATH = "gs://shopee-product-detection-data/data"
CLASSES = 42
BATCH_SIZE = 16 * STRATEGY.num_replicas_in_sync
AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 25

VAL_SPLIT = 0.1
TRAIN_DF = pd.read_csv(GCS_PATH + "/train.csv")
TRAIN_LEN = int(TRAIN_DF["filename"].shape[0] * (1 - VAL_SPLIT))
VAL_LEN = int(TRAIN_DF["filename"].shape[0] * VAL_SPLIT)

IMG_SIZE = (240, 240)

OPTIMIZER = "sgd"
LOSS_FN = "categorical_crossentropy"

MODEL_NAME = "modelb1.h5"
SAVE_PATH = "gs://shopee-product-detection-data/models" + MODEL_NAME


"""
"""
def get_model():

    with STRATEGY.scope():
        
        efnm = efn.EfficientNetB1(weights='noisy-student', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        efnm.trainable = True
        
        model = tf.keras.models.Sequential([
                efnm,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(CLASSES, activation='softmax')
            ])

        model.compile(loss=LOSS_FN, optimizer=OPTIMIZER, metrics=['accuracy'])

    return model

"""
"""
def train():

    model = get_model()

    model.save(MODEL_NAME, save_format='h5')

    with file_io.FileIO(MODEL_NAME, mode='rb') as in_file:
        with file_io.FileIO(SAVE_PATH, mode='wb+') as out_file:
            out_file.write(in_file.read())


if __name__ == "__main__":
    train()