import glob
import os
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _int64_feature(value):
    """This functions can be used to convert a value to a type compatible
    with tf.train.Example

    Returns:
        an int64_list from int
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """This functions can be used to convert a value to a type compatible
    with tf.train.Example

    Returns:
        a bytes_list from a string / byte
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_examples(example_data, filename: str, channels):
    """ This method is used for writing the examples as a TFRecord
    which contains features and label of the image.

    Args:
        example_data:
        filename: name of the file
        channels: value of the channels

    Returns:

    """
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
        self.original_img_dir = '../femur_2d_segmentations/middle_slices/*'
        self.corrupt_images = glob.glob('../images/*')
        self.record_path = '../records'
        self.train_ratio = 0.9
        self.valid_ratio = 0.05

    def create_dataset(self, image_list, data_type):
        data = []
        for org_img_path in tqdm(image_list):
            img_name = os.path.basename(org_img_path).split('.')[0]
            crr_img_path_list = [k for k in self.corrupt_images if
                                 img_name in k]
            img_label = cv2.imread(org_img_path, cv2.IMREAD_UNCHANGED)
            img_label = img_label.astype(np.float32)
            for crr_img_path in crr_img_path_list:
                shape_name = os.path.basename(
                    crr_img_path).split('.')[0].split('_')[1]
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

        # get the image list based on ration
        train_image_no = int(len(images) * self.train_ratio)
        valid_image_no = train_image_no + int(len(images) * self.valid_ratio)

        # separating images with train test validation
        train_image_list = images[:train_image_no]
        valid_image_list = images[train_image_no:valid_image_no]
        test_image_list = images[valid_image_no:]

        data_types = {'validation': valid_image_list,
                      'train': train_image_list, 'test': test_image_list
                      }

        for d in data_types:
            self.create_dataset(image_list=data_types[d], data_type=d)


if __name__ == '__main__':
    tf_record = WritingRecord()
    tf_record.pre_process_data()
