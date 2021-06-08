import glob
import os

import nibabel as nib
import numpy as np

number_of_rect = 7


def rect(image, image_np):
    x = np.random.randint(0, 160, number_of_rect)
    y = np.random.randint(0, 384, number_of_rect)
    z = np.random.randint(0, 384, number_of_rect)

    a = 40
    b = 40
    c = 40

    x_start = np.maximum(x - a, 0)
    y_start = np.maximum(y - b, 0)
    z_start = np.maximum(z - c, 0)

    x_end = np.minimum(x + a, image.shape[0])
    y_end = np.minimum(y + b, image.shape[1])
    z_end = np.minimum(z + c, image.shape[2])

    for i in range(0, number_of_rect):
        image_np[x_start[i]:x_end[i], y_start[i]:y_end[i], z_start[i]:z_end[i]] = 0

    image_nifti = nib.Nifti1Image(image_np, image.affine)
    return image_nifti


def write_images(file_path, image_nifti):
    names = os.path.basename(file_path).split('.')
    dest_file_name = names[0] + '_' + "corrupt" + '.' + names[1] + "." + names[2]
    nib.save(image_nifti, os.path.join(dest_path, dest_file_name))


if __name__ == '__main__':
    data_path = "D:/PROJECTS/internship/3D_data/labels_nifti_femur"
    dest_path = "D:/PROJECTS/internship/3D_data/labels_nifti_femur_corrupt"

    files = glob.glob(os.path.join(data_path, '*'))

    for file in files:
        image = nib.load(file)
        image_np = nib.load(file, mmap=False).get_fdata(caching='unchanged')

        img = rect(image, image_np)
        write_images(file_path=file, image_nifti=img)
