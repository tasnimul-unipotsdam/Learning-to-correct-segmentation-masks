import os
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

class TFRecordReader:

    def __init__(self):
        self.record_path = '../records'
        self.seed = 10
        self.batch_size = 4
        self.buffer = 100

    @classmethod
    def parse_record(cls, record):
        """This method is used for parsing the records e.g extracting features,
        labels.

        Args:
            record: tfrecord

        Returns:
            image, label

        """
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
        print(org_image)
        print(org_image.shape)
        org_image = tf.reshape(org_image, [record['org_img_height'], record['org_img_width'], 1])

        return crr_image, org_image


    def generate_dataset(self, is_training=True):
        """Read tfrecord, parse the images and labels, do all the processing
        like different types of augmentation.

        Args:
            is_training: a boolean value

        Returns:
            processed dataset
        """
        is_shuffle = True
        if is_training:
            data_type = 'train'
            buffer = 1000
        else:
            #         is_shuffle = False
            data_type = 'validation'
            buffer = 100

        files = os.path.join(self.record_path,f'{data_type}_femur.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=is_shuffle,
                                             seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn),
                                     cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames),
                                                            tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_training:
            # dataset = dataset.map(random_jitter,
            #                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(buffer, seed=self.seed)
            dataset = dataset.repeat(1)
            dataset = dataset.prefetch(1)
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size)
        return dataset

def show(crop, title=None):
    plt.imshow(crop)
    if title:
        plt.title(title)
    plt.show()

if __name__ == '__main__':
    record_reader = TFRecordReader()
    train_dataset = record_reader.generate_dataset(is_training=True)
    print(train_dataset)

    for i, batch in enumerate(train_dataset):
        images, labels = batch
        print('images size: {}, labels size: {}'.format(images.shape,
                                                        labels.shape))
        break



    image, label = next(iter(train_dataset))

    for i in range(1):
        print(label[i])
        cv2.imshow('label', image[0].numpy())

    cv2.waitKey()

