import os
import glob
import cv2

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _normalize_(corrupt_image, original_image):
    corrupt_image = corrupt_image / 255
    original_image = original_image / 255
    return corrupt_image, original_image


class TFRecordReader(object):

    def __init__(self, record_path):
        self.record_path = record_path
        self.seed = 10
        self.batch_size = 1265
        self.is_shuffle = True
        self.data_type = 'test'
        self.buffer = 1000

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
        crr_image = tf.reshape(crr_image, [record['crr_img_height'], record['crr_img_width'],
                                           record['crr_img_depth']])

        org_image = tf.io.decode_raw(record['org_image'], tf.float32)
        org_image = tf.reshape(org_image, [record['org_img_height'], record['org_img_width'],
                                           record['org_img_depth']])

        return crr_image, org_image

    def test_dataset(self):
        files = os.path.join(self.record_path, f'{self.data_type}_femur.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=self.is_shuffle, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn),
                                     cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(_normalize_, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(self.buffer, seed=self.seed)
        # dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    test_dataset = TFRecordReader("D:/PROJECTS/internship/test_records").test_dataset()

    for i, batch in enumerate(test_dataset):
        images, labels = batch
        print('images size: {}, labels size: {}'.format(images.shape, labels.shape))
        break

    image, label = next(iter(test_dataset))

    for i in range(1):
        # print(label[i])
        cv2.imshow('images', image[0].numpy())

    cv2.waitKey()
