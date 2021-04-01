import glob
import os
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_examples(example_data, filename: str, channels):
    with tf.io.TFRecordWriter(filename) as writer:
        for i, ex in enumerate(example_data):
            corrupt_image = ex['image'].tostring()
            original_image = ex['label'].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'crr_img_height': _int64_feature(ex['image'].shape[0]),
                'crr_img_width': _int64_feature(ex['image'].shape[1]),
                'crr_img_depth': _int64_feature(channels),
                'crr_image': _bytes_feature(corrupt_image),

                'org_img_height': _int64_feature(ex['label'].shape[0]),
                'org_img_width': _int64_feature(ex['label'].shape[1]),
                'org_img_depth': _int64_feature(channels),
                'org_image': _bytes_feature(original_image),
            }))
            writer.write(example.SerializeToString())
    return None


class WritingRecord:

    def __init__(self):
        self.original_img_dir = '../Test_dataset_253_subjects/*'
        self.corrupt_images = glob.glob('../test_images_corrupted/*')
        self.record_path = '../test_records'

    def create_dataset(self, image_list, data_type):
        data = []
        for org_img_path in tqdm(image_list):
            img_name = os.path.basename(org_img_path).split('.')[0]
            crr_img_path_list = [k for k in self.corrupt_images if img_name in k]
            img_label = cv2.imread(org_img_path, cv2.IMREAD_UNCHANGED)
            img_label = img_label.astype(np.float32)
            for crr_img_path in crr_img_path_list:
                shape_name = os.path.basename(crr_img_path).split('.')[0].split('_')[1]
                img = cv2.imread(crr_img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img = img.astype(np.float32)
                    meta = {
                        'image': img,
                        'label': img_label,
                        'shape_name': shape_name,
                        'data_type': data_type
                    }
                    data.append(meta)

        file_name = f'{data_type}_femur.tfrecord'
        file_name = os.path.join(self.record_path, file_name)
        _write_examples(example_data=data, filename=file_name, channels=1)

    def pre_process_data(self):
        # get all data from directory
        images = glob.glob(self.original_img_dir)
        shuffle(images)

        test_image_list = images[:]

        print(len(test_image_list))

        data_types = {'test': test_image_list}

        for d in data_types:
            self.create_dataset(image_list=data_types[d], data_type=d)


if __name__ == '__main__':
    tf_record = WritingRecord()
    tf_record.pre_process_data()
