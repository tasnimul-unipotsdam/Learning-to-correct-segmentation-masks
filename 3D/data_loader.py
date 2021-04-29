import os
import numpy as np

import nibabel as nib
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

original_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur"
corrupt_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur_corrupt"

input_image_path = [os.path.join(corrupt_image_dir, x) for x in os.listdir(corrupt_image_dir)]
label_image_path = [os.path.join(original_image_dir, x) for x in os.listdir(original_image_dir)]


def read_nifti_file(filepath):
    file = nib.load(filepath)
    file = file.get_fdata()
    return file


def process_file(path):
    volume = read_nifti_file(path)
    return volume


def nifti_numpy():
    input_path = np.array([process_file(path) for path in input_image_path],
                          dtype=np.int8)
    label_path = np.array([process_file(path) for path in label_image_path],
                          dtype=np.int8)

    inputs_image = np.expand_dims(input_path, axis=4)
    label_image = np.expand_dims(label_path, axis=4)

    train_ratio = 0.8
    train_image_no = int(len(inputs_image) * train_ratio)

    train_input_image = inputs_image[:train_image_no]
    train_label_image = label_image[:train_image_no]

    val_input_image = inputs_image[train_image_no:]
    val_label_image = label_image[train_image_no:]

    print(len(inputs_image))
    print(len(label_image))

    print(inputs_image.shape)
    print(label_image.shape)

    print(len(train_input_image))
    print(len(val_input_image))

    return train_input_image, train_label_image, val_input_image, val_label_image


class DataLoader(object):
    def __init__(self, inputs, labels, training=True):
        self.inputs = inputs
        self.labels = labels
        self.training = "train" if training else "validation"
        self.seed = 100
        if self.training == "train":
            self.batch_size = 4
            self.buffer = 300
        else:
            self.batch_size = 4
            self.buffer = 50

    def train_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        data = data.shuffle(self.buffer, seed=self.seed)
        data = data.batch(self.batch_size)
        data = data.repeat()
        data = data.prefetch(AUTOTUNE)
        return data

    def validation_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        data = data.shuffle(self.buffer, seed=self.seed)
        data = data.batch(self.batch_size)
        data = data.repeat()
        data = data.prefetch(AUTOTUNE)
        return data


if __name__ == "__main__":
    train_input_image, train_label_image, val_input_image, val_label_image = nifti_numpy()

    train_dataset = DataLoader(inputs=train_input_image, labels=train_label_image,
                               training=True).train_data()

    validation_dataset = DataLoader(inputs=val_input_image, labels=val_label_image,
                                    training=False).validation_data()

    print(train_dataset)
    print(validation_dataset)

    for i, batch in enumerate(train_dataset):
        images, labels = batch
        print('images size: {}, labels size: {}'.format(images.shape, labels.shape))
        break
