import time
import os
import random
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K

from distance import compute_surface_distances, compute_average_surface_distance, \
    compute_robust_hausdorff

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seeds = 1000
seed_everything(seeds)

start = time.time()

test_input_image = np.load("D:/PROJECTS/internship/3D_data/test_input_image.npy")
test_label_image = np.load("D:/PROJECTS/internship/3D_data/test_label_image.npy")

Ntest = len(test_input_image)


def prediction():
    model = tf.keras.models.load_model(
        "D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_weights.h5", compile=False)
    mask_prediction = model.predict(test_input_image, batch_size=1)
    pred_mask = np.save("D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_pred_mask.npy",
                        mask_prediction)
    return pred_mask


def numpy_nifti():
    pred_mask = np.load("D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_pred_mask.npy")
    threshold = 0.5
    pred_mask = np.select([pred_mask <= threshold, pred_mask > threshold],
                          [np.zeros_like(pred_mask), np.ones_like(pred_mask)])

    for i in range(pred_mask.shape[0]):
        img = nib.Nifti1Image(pred_mask[i], np.eye(4))
        img.to_filename(os.path.join("D:/PROJECTS/internship/3D_data/unet_3D_02/predicted_images_unet_3D_02",
                                     f'{i}_prediction.nii.gz'))


def robust_hausdorff(m_gt, m_pr):
    m_gt = m_gt > 0.5
    m_gt = np.resize(m_gt, (m_gt.shape[0], m_gt.shape[1], m_gt.shape[2]))
    # print(m_gt.shape)

    m_pr = m_pr > 0.5
    m_pr = np.resize(m_pr, (m_pr.shape[0], m_pr.shape[1], m_pr.shape[2]))

    surface_distance = compute_surface_distances(m_gt, m_pr, spacing_mm=(1, 1, 1))
    hausdorff_distance = compute_robust_hausdorff(surface_distance, 100)

    return hausdorff_distance


def compute_hausdorff():
    pred_mask = np.load("D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_pred_mask.npy")
    hausdorff = []
    for i in range(Ntest):
        hausdorff.append(robust_hausdorff(test_label_image[i], pred_mask[i]))
    hausdorff = np.array(hausdorff)
    np.save('D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_hausdorff_distance.npy',
            hausdorff)


def dice_coefficient(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def compute_dice():
    pred_mask = np.load("D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_pred_mask.npy")
    dice = []
    for i in range(Ntest):
        dice.append(dice_coefficient(test_label_image[i], pred_mask[i]))
    dice = np.array(dice)
    np.save('D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_dice_score.npy', dice)


if __name__ == '__main__':
    # prediction()
    # numpy_nifti()
    # compute_hausdorff()
    compute_dice()
