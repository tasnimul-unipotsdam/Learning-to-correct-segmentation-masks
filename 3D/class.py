import os

import nibabel as nib
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

original_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur"
corrupt_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur_corrupt"

input_image_path = [os.path.join(corrupt_image_dir, x) for x in os.listdir(corrupt_image_dir)]
label_image_path = [os.path.join(original_image_dir, x) for x in os.listdir(original_image_dir)]


def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan


def process_scan(path):
    volume = read_nifti_file(path)
    return volume


def nifti_numpy():
    inputs = np.array([process_scan(path) for path in input_image_path],
                      dtype=np.int8)
    labels = np.array([process_scan(path) for path in label_image_path],
                      dtype=np.int8)

    inputs = np.expand_dims(inputs, axis=4)
    labels = np.expand_dims(labels, axis=4)

    train_ratio = 0.8
    train_image_no = int(len(inputs) * train_ratio)

    train_inputs = inputs[:train_image_no]
    train_labels = labels[:train_image_no]

    val_inputs = inputs[train_image_no:]
    val_labels = labels[train_image_no:]

    print(len(inputs))
    print(len(labels))

    print(inputs.shape)
    print(labels.shape)

    print(len(train_inputs))
    print(len(val_inputs))

    return train_inputs, train_labels, val_inputs, val_labels


def train_data(train_inputs, train_labels):
    data = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    data = data.shuffle(1000, seed=10)
    data = data.batch(4)
    data = data.repeat()
    data = data.prefetch(AUTOTUNE)
    return data


if __name__ == "__main__":
    train_inputs, train_labels, val_inputs, val_labels = nifti_numpy()
    training = train_data(train_inputs=train_inputs, train_labels=train_labels)

    print(training)

    for i, batch in enumerate(training):
        images, labels = batch
        print('images size: {}, labels size: {}'.format(images.shape, labels.shape))
        break
