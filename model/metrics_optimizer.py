import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


from distance import compute_surface_distances, compute_average_surface_distance, \
    compute_robust_hausdorff

smooth = 1


def surface_distance(mask_gt, mask_pred):
    mask_gt_cp = mask_gt.numpy().copy()
    mask_pred_cp = mask_pred.numpy().copy()

    sf_list = []
    for i in range(mask_gt.shape[0]):
        m_gt = mask_gt_cp[i]
        p_gt = mask_pred_cp[i]

        m_gt = np.asarray(m_gt, np.float32)
        m_gt = m_gt >= 0.5
        m_gt = np.resize(m_gt, (mask_gt.shape[1], mask_gt.shape[2]))

        p_gt = np.asarray(p_gt, np.float32)
        p_gt = p_gt >= 0.5
        p_gt = np.resize(p_gt, (mask_pred.shape[1], mask_pred.shape[2]))

        surface_dict = compute_surface_distances(m_gt, p_gt, spacing_mm=(1, 1))
        sf = compute_average_surface_distance(surface_dict)
        sf_list.append([sf[0], sf[1]])

    sum_arr = np.array(sf_list).sum(axis=0)
    avg = sum_arr / 4
    return tuple(avg)


# def surface_distance(mask_gt, mask_pred):
#     mask_gt = np.asarray(mask_gt, np.float32)
#     mask_gt = mask_gt >= 0.5
#     mask_gt = np.resize(mask_gt, (mask_gt.shape[1], mask_gt.shape[2]))
#
#     mask_pred = np.array(mask_pred, np.float32)
#     mask_pred = mask_pred >= 0.5
#     mask_pred = np.resize(mask_pred, (mask_pred.shape[1], mask_pred.shape[2]))
#
#     surface_dict = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1, 1))
#     return compute_average_surface_distance(surface_dict)


def robust_hausdorff(mask_gt, mask_pred):
    mask_gt_cp = mask_gt.numpy().copy()
    mask_pred_cp = mask_pred.numpy().copy()

    haus_dorff_dist = []
    for i in range(mask_gt.shape[0]):
        m_gt = mask_gt_cp[i]
        p_gt = mask_pred_cp[i]

        m_gt = np.asarray(m_gt, np.float32)
        m_gt = m_gt >= 0.5
        m_gt = np.resize(m_gt, (mask_gt.shape[1], mask_gt.shape[2]))

        p_gt = np.asarray(p_gt, np.float32)
        p_gt = p_gt >= 0.5
        p_gt = np.resize(p_gt, (mask_pred.shape[1], mask_pred.shape[2]))

        surface_dict = compute_surface_distances(m_gt, p_gt, spacing_mm=(1, 1))
        haus_dorff_dist.append(compute_robust_hausdorff(surface_dict, 100))

    avg = sum(haus_dorff_dist) / 4
    return avg


# def robust_hausdorff(mask_gt, mask_pred):
#
#     mask_gt = np.asarray(mask_gt, np.float32)
#     mask_gt = mask_gt >= 0.5
#     mask_gt = np.resize(mask_gt, (mask_gt.shape[1], mask_gt.shape[2]))
#
#     mask_pred = np.array(mask_pred, np.float32)
#     mask_pred = mask_pred >= 0.5
#     mask_pred = np.resize(mask_pred, (mask_pred.shape[1], mask_pred.shape[2]))
#
#     surface_dict = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1, 1))
#     haus_dorff_dist = compute_robust_hausdorff(surface_dict, 100)
#     return haus_dorff_dist


def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss


binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

adam = tf.keras.optimizers.Adam(learning_rate=0.0001)


def plot_crossentropy_loss(history, name):
    fig = plt.figure()

    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('crossentropy_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


def plot_dice_loss(history, name):
    fig = plt.figure()

    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('dice loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


def plot_metrics(history, name):
    fig = plt.figure(num=None, figsize=(20, 6), dpi=150, facecolor='w', edgecolor='k',
                     tight_layout=True)

    plt.subplot(1, 3, 1)
    plt.plot(history.history['dice_coefficient'], color='b', label="train_coefficient")
    plt.plot(history.history['val_dice_coefficient'], color='r', label="validation_coefficient")
    plt.title('dice coefficient')
    plt.ylabel('coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 3, 2)
    plt.plot(history.history['surface_distance'], color='b', label="train_surface_distance")
    plt.plot(history.history['val_surface_distance'], color='r',
             label="validation_surface_distance")
    plt.title('surface distance')
    plt.ylabel('surface_distance')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 3, 3)
    plt.plot(history.history['robust_hausdorff'], color='b', label="train_robust_hausdorff")
    plt.plot(history.history['val_robust_hausdorff'], color='r',
             label="validation_robust_hausdorff")
    plt.title('hausdorff distance')
    plt.ylabel('hausdorff_distance')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)
