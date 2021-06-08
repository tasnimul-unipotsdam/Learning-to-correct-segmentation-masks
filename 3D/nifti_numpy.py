import os

import nibabel as nib
import numpy as np

from scipy import ndimage


def read_nifti_file(filepath):
    file = nib.load(filepath)
    file = file.get_fdata()
    return file


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth

    desired_width = 128
    desired_height = 224
    desired_depth = 224
    # Get current depth

    current_width = img.shape[0]
    current_height = img.shape[1]
    current_depth = img.shape[-1]
    # Compute depth factor

    width = current_width / desired_width
    height = current_height / desired_height
    depth = current_depth / desired_depth

    width_factor = 1 / width
    height_factor = 1 / height
    depth_factor = 1 / depth

    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_file(path):
    volume = read_nifti_file(path)

    volume = resize_volume(volume)
    return volume


def nifti_numpy():
    corrupt_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur_corrupt"
    original_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur"

    input_image_path = [os.path.join(corrupt_image_dir, x) for x in os.listdir(corrupt_image_dir)]
    label_image_path = [os.path.join(original_image_dir, x) for x in os.listdir(original_image_dir)]

    input_path = np.array([process_file(path) for path in input_image_path]).astype('float32')
    label_path = np.array([process_file(path) for path in label_image_path]).astype('float32')
    image_name_list = [os.path.basename(p) for p in label_image_path]

    inputs_image = np.expand_dims(input_path, axis=4)
    label_image = np.expand_dims(label_path, axis=4)

    train_input_image = inputs_image[:int(len(inputs_image) * 0.6)].astype('float32')
    train_label_image = label_image[:int(len(inputs_image) * 0.6)].astype('float32')

    val_input_image = inputs_image[
                      int(len(inputs_image) * 0.6):int(len(inputs_image) * 0.8)].astype('float32')
    val_label_image = label_image[int(len(inputs_image) * 0.6):int(len(inputs_image) * 0.8)].astype(
        'float32')

    test_input_image = inputs_image[int(len(inputs_image) * 0.8):].astype('float32')
    test_label_image = label_image[int(len(inputs_image) * 0.8):].astype('float32')

    test_image_name = image_name_list[int(len(inputs_image) * 0.8):]

    np.save("D:/PROJECTS/internship/3D_data/train_input_image", train_input_image)
    np.save("D:/PROJECTS/internship/3D_data/train_label_image", train_label_image)

    np.save("D:/PROJECTS/internship/3D_data/val_input_image", val_input_image)
    np.save("D:/PROJECTS/internship/3D_data/val_label_image", val_label_image)

    np.save("D:/PROJECTS/internship/3D_data/test_input_image", test_input_image)
    np.save("D:/PROJECTS/internship/3D_data/test_label_image", test_label_image)

    with open('D:/PROJECTS/internship/3D_data/test_image_name.txt', 'w+') as f:
        for list_item in test_image_name:
            f.write('%s\n' % list_item)

    print(train_input_image.shape)
    print(val_input_image.shape)
    print(test_input_image.shape)

    return train_input_image, train_label_image, val_input_image, val_label_image


if __name__ == '__main__':
    nifti_numpy()
