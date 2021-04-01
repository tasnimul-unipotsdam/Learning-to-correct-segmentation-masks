import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from distance import compute_surface_distances, compute_average_surface_distance, \
    compute_robust_hausdorff
from test_data.test_tfrecord_reader import TFRecordReader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

test_dataset = TFRecordReader("D:/PROJECTS/internship/test_records").test_dataset()

model = tf.keras.models.load_model(
    "D:/PROJECTS/internship/saved model/unet_20_model.h5", compile=False)


def display(display_list):
    fig = plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    fig.savefig("unet_20_bad_pred_dice")


smooth = 1


def show_predictions(dataset=None, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        #  pred_mask *= 255.0
        print(pred_mask.min())
        print(pred_mask.max())
        print(image[0])
        print(np.unique(image, return_counts=True))
        print(np.unique(pred_mask, return_counts=True))
        display([image[0], mask[0], pred_mask[0]])


def compute_best_worst_dist(dataset, num=1265):
    distance_dict = {}
    for image, base_mask_gt in dataset.take(num):
        base_mask_pred = model.predict(image)
        for image_num in range(0, image.shape[0]):
            mask_gt = base_mask_gt.numpy().copy()
            mask_pred = base_mask_pred.copy()

            'Dice Coefficient'
            y_true_f = K.flatten(mask_gt[image_num])
            y_pred_f = K.flatten(mask_pred[image_num])
            intersection = K.sum(y_true_f * y_pred_f)
            dice_score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

            "Hausdroff Distance and Average Surface Distance"
            mask_gt = np.asarray(mask_gt[image_num], np.float32)
            mask_gt = mask_gt >= 0.5
            mask_gt = np.resize(mask_gt, (mask_gt.shape[1], mask_gt.shape[2]))

            mask_pred = np.asarray(mask_pred[image_num], np.float32)
            mask_pred = mask_pred >= 0.5
            mask_pred = np.resize(mask_pred, (mask_pred.shape[1], mask_pred.shape[2]))

            surface_dict = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1, 1))

            hausdorff_distance100 = compute_robust_hausdorff(surface_dict, 100)
            average_distance = compute_average_surface_distance(surface_dict)

            distance_dict[image_num] = [hausdorff_distance100, average_distance,
                                        dice_score.numpy()]
    idx = 2
    distance_list = []
    for key in distance_dict:
        distance_list.append(distance_dict[key][idx])
    key_list = list(distance_dict.keys())
    dist_arr = np.array(distance_list)

    arg_min = dist_arr.argmin()
    arg_max = dist_arr.argmax()

    print(
        'Image Index with lowest distance: {}, Image Index with highest distance: {}, '
        'lowest distance: {}, highest distance: {}'.format(
            key_list[arg_min],
            key_list[arg_max],
            dist_arr[arg_min],
            dist_arr[arg_max]))

    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(dist_arr)
    plt.show()
    fig.savefig("unet_20_dice-coefficient_boxplot")

    return distance_dict


def compute_metrics_score(dataset, num, image_num):
    for image, mask_gt in dataset.take(num):
        mask_pred = model.predict(image)
        # mask_pred *= 255.0
        display([image[image_num], mask_gt[image_num], mask_pred[image_num]])

        'Dice Coefficient'
        y_true_f = K.flatten(mask_gt[image_num])
        y_pred_f = K.flatten(mask_pred[image_num])
        intersection = K.sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        "Hausdroff Distance and Average Surface Distance"
        mask_gt = np.asarray(mask_gt[image_num], np.float32)
        mask_gt = mask_gt >= 0.5
        mask_gt = np.resize(mask_gt, (mask_gt.shape[1], mask_gt.shape[2]))

        mask_pred = np.asarray(mask_pred[image_num], np.float32)
        mask_pred = mask_pred >= 0.5
        mask_pred = np.resize(mask_pred, (mask_pred.shape[1], mask_pred.shape[2]))

        surface_dict = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1, 1))

        hausdorff_distance100 = compute_robust_hausdorff(surface_dict, 100)
        average_distance = compute_average_surface_distance(surface_dict)

        print(
            'Robust Hausdroff Distance100: {}, Average Surface Distance: {}, Dice Coefficient: {}'.format(
                hausdorff_distance100,
                average_distance,
                dice_score))

        return hausdorff_distance100, average_distance, dice_score


if __name__ == '__main__':
    compute_metrics_score(test_dataset, num=1, image_num=645)
    # show_predictions(test_dataset, 20)
    # distance_dict = compute_best_worst_dist(test_dataset)
    # print(len(list(distance_dict.keys())))
