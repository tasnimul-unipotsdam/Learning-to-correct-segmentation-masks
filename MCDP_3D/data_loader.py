import time
import os
import numpy as np
from scipy import ndimage

import nibabel as nib
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocessing(image, label):
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    return image, label


class DataLoader3D(object):
    def __init__(self, inputs, labels, training=True):
        self.inputs = inputs
        self.labels = labels
        self.training = "train" if training else "validation"
        self.seed = 100
        if self.training == "train":
            self.batch_size = 1
            self.buffer = 155
        else:
            self.batch_size = 1
            self.buffer = 50

    def train_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        data = data.map(preprocessing, num_parallel_calls=AUTOTUNE)
        data = data.shuffle(self.buffer, seed=self.seed)
        data = data.batch(self.batch_size)
        data = data.repeat()
        data = data.prefetch(AUTOTUNE)
        return data

    def validation_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        data = data.map(preprocessing, num_parallel_calls=AUTOTUNE)
        data = data.shuffle(self.buffer, seed=self.seed)
        data = data.batch(self.batch_size)
        # data = data.repeat()
        # data = data.prefetch(AUTOTUNE)
        return data


if __name__ == "__main__":

    train_input_image = np.load("D:/PROJECTS/internship/3D_data/train_input_image.npy")
    train_label_image = np.load("D:/PROJECTS/internship/3D_data/train_label_image.npy")
    val_input_image = np.load("D:/PROJECTS/internship/3D_data/val_input_image.npy")
    val_label_image = np.load("D:/PROJECTS/internship/3D_data/val_label_image.npy")

    train_dataset = DataLoader3D(inputs=train_input_image, labels=train_label_image,
                                 training=True).train_data()

    validation_dataset = DataLoader3D(inputs=val_input_image, labels=val_label_image,
                                      training=False).validation_data()

    print(train_dataset)
    print(validation_dataset)

    for i, batch in enumerate(train_dataset):
        images, labels = batch
        print('images size: {}, labels size: {}'.format(images.shape, labels.shape))
        break
