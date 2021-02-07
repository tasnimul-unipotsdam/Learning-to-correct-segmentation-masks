import numpy as np
import tensorflow as tf

from pipeline.augmentation import train_augmentation, validation_augmentation
from pipeline.path import train_mask_path, train_corrupt_mask_path


class DataLoader(object):
    def __init__(self, corrupt_mask_path, mask_path, training=True):
        self.corrupt_mask_path = corrupt_mask_path
        self.mask_path = mask_path
        self.training = "train" if training else "validation"
        self.seed = 1
        if self.training == "train":
            self.batch_size = 4
            self.buffer = 1000
        else:
            self.batch_size = 4
            self.buffer = 100

    @staticmethod
    def _parse_image_mask(corrupt_mask_path, mask_path):
        corrupt_mask_file = tf.io.read_file(corrupt_mask_path)
        corrupt_mask = tf.image.decode_jpeg(corrupt_mask_file, channels=1)
        corrupt_mask = tf.image.convert_image_dtype(corrupt_mask, tf.uint8)

        mask_file = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask_file, channels=1)
        return corrupt_mask, mask


    def load_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.corrupt_mask_path, self.mask_path))
        data = data.map(self._parse_image_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.training == "train":
            data = data.map(train_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            data = data.shuffle(self.buffer, seed=self.seed)
            data = data.repeat()
            data = data.batch(self.batch_size)
            data = data.prefetch(1)
        else:
            data = data.map(validation_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            data = data.repeat(1)
            data = data.batch(self.batch_size)
        return data


if __name__ == "__main__":
    train_dataset = DataLoader(train_corrupt_mask_path, train_mask_path, training=True).load_data()
    # validation_dataset = DataLoader(validation_image_path, validation_mask_path, training=False).load_data()
    print(train_dataset)
    # print(validation_dataset)
