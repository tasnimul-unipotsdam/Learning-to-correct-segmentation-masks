import os
import glob
import cv2

import tensorflow as tf

from pipeline.augmentation import _normalize_, train_augmentation



class TFRecordReader(object):

    def __init__(self, record_path, is_training=True):
        self.record_path = record_path
        self.seed = 10
        self.batch_size = 4
        self.buffer = 100
        self.is_shuffle = True
        self.is_training = 'train' if is_training else 'validation'

        if self.is_training:
            self.data_type = 'train'
            self.buffer = 1000
        else:
            self.data_type = 'validation'
            self.buffer = 100

    @classmethod
    def parse_record(cls, record):
        features = {
            'crr_image': tf.io.FixedLenFeature([], dtype=tf.string),
            'crr_img_height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'crr_img_width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'crr_img_depth': tf.io.FixedLenFeature([], dtype=tf.int64),

            'org_image': tf.io.FixedLenFeature([], dtype=tf.string),
            'org_img_height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'org_img_width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'org_img_depth': tf.io.FixedLenFeature([], dtype=tf.int64),
        }

        record = tf.io.parse_single_example(record, features)

        crr_image = tf.io.decode_raw(record['crr_image'], tf.float32)
        crr_image = tf.reshape(crr_image, [record['crr_img_height'], record['crr_img_width'], 1])

        org_image = tf.io.decode_raw(record['org_image'], tf.float32)
        org_image = tf.reshape(org_image, [record['org_img_height'], record['org_img_width'], 1])

        return crr_image, org_image


    def train_dataset(self):
        files = os.path.join(self.record_path,f'{self.data_type}_femur.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=self.is_shuffle,
                                             seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn),
                                     cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames),
                                                            tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(train_augmentation,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(_normalize_,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(self.buffer, seed=self.seed)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def validation_dataset(self):
        files = os.path.join(self.record_path,f'{self.data_type}_femur.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=self.is_shuffle,
                                             seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn),
                                     cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames),
                                                            tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(_normalize_,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    train_dataset = TFRecordReader('../records', is_training=True).train_dataset()
    validation_dataset = TFRecordReader('../records', is_training=False).validation_dataset()
    print(train_dataset)

    for i, batch in enumerate(train_dataset):
        images, labels = batch
        print('images size: {}, labels size: {}'.format(images.shape,
                                                        labels.shape))
        break



    image, label = next(iter(train_dataset))

    for i in range(1):
        print(label[i])
        cv2.imshow('label', label[0].numpy())

    cv2.waitKey()

