import os
import glob

import nibabel as nib
import numpy as np

original_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur"
corrupt_image_dir = "D:/PROJECTS/internship/3D_data/labels_nifti_femur_corrupt"

original_image_path = glob.glob(os.path.join(original_image_dir, '*'))
corrupt_image_path = glob.glob(os.path.join(corrupt_image_dir, '*'))

print(len(original_image_path))


def nifti_numpy():
    data = []
    for file in original_image_path:
        image_np = nib.load(file, mmap=False).get_fdata(caching='unchanged')
        data.append(image_np)
        np.save("D:/PROJECTS/internship/3D_data/original_image_y.npy", data)


nifti_numpy()

# X_train = np.load("X_valid.npy")
# print(X_train.shape)
# img = X_train[1]
# print(img)
# print(len(X_train))
# print(type(X_train))
