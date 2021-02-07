import numpy as np
import tensorflow as tf

from pipeline.augmentation import train_augmentation, validation_augmentation
from pipeline.path import mask_paths, corrupt_mask_path


class DataLoader(object):
    def __init__(self, image_path, mask_path, training=True):
        self.image_path = image_path
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
    def _parse_image_mask(image_path, mask_path):
        image_file = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_file, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)


        mask_file = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask_file, channels=1)
        mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

        return image, mask

    def load_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.image_path, self.mask_path))
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
    train_dataset = DataLoader(corrupt_mask_path, mask_paths, training=True).load_data()
    # validation_dataset = DataLoader(validation_image_path, validation_mask_path, training=False).load_data()
    print(train_dataset)
    # print(validation_dataset)
