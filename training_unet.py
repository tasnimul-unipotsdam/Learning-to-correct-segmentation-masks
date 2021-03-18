import os
import numpy as np
import tensorflow as tf

from distance import compute_surface_distances, compute_average_surface_distance, \
    compute_robust_hausdorff

from pipeline.tfrecord_reader import TFRecordReader
from model.unet import unet
from model.metrics_optimizer import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

tf_train_recorder = TFRecordReader("D:/PROJECTS/internship/records", is_training=True)
train_dataset = tf_train_recorder.train_dataset()
tf_valid_recorder = TFRecordReader("D:/PROJECTS/internship/records",  is_training=False)
validation_dataset = tf_valid_recorder.validation_dataset()


def surface_distance(mask_gt, mask_pred):
    sf_list = []
    for i in range(mask_gt.shape[0]):
        m_gt = mask_gt.numpy()[i]
        p_gt = mask_pred.numpy()[i]

        m_gt = np.asarray(m_gt, np.float32)
        m_gt = m_gt >= 0.5
        m_gt = np.resize(m_gt, (mask_gt.shape[1], mask_gt.shape[2]))

        p_gt = np.array(p_gt, np.float32)
        p_gt = p_gt >= 0.5
        p_gt = np.resize(p_gt, (mask_pred.shape[1], mask_pred.shape[2]))

        surface_dict = compute_surface_distances(m_gt, p_gt, spacing_mm=(1, 1))
        sf = compute_average_surface_distance(surface_dict)
        sf_list.append([sf[0], sf[1]])

    sum_arr = np.array(sf_list).sum(axis=0)
    avg = sum_arr / tf_train_recorder.batch_size
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
    haus_dorff_dist = []
    for i in range(mask_gt.shape[0]):
        m_gt = mask_gt.numpy()[i]
        p_gt = mask_pred.numpy()[i]

        m_gt = np.asarray(m_gt, np.float32)
        m_gt = m_gt >= 0.5
        m_gt = np.resize(m_gt, (mask_gt.shape[1], mask_gt.shape[2]))

        p_gt = np.array(p_gt, np.float32)
        p_gt = p_gt >= 0.5
        p_gt = np.resize(p_gt, (mask_pred.shape[1], mask_pred.shape[2]))

        surface_dict = compute_surface_distances(m_gt, p_gt, spacing_mm=(1, 1))
        haus_dorff_dist.append(compute_robust_hausdorff(surface_dict, 100))

    avg = sum(haus_dorff_dist) / tf_train_recorder.batch_size
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


def compile_model():
    model = unet()
    model.summary()
    model.compile(optimizer=adam,
                  loss=dice_loss,
                  metrics=[dice_coefficient, surface_distance, robust_hausdorff], run_eagerly=True)
    return model


def train_model(model):
    BATCH_SIZE = 8
    STEPS_PER_EPOCH = 1016 // BATCH_SIZE
    VALIDATION_STEPS = 254 // BATCH_SIZE
    history = model.fit(train_dataset,
                        verbose=2,
                        epochs=100,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=validation_dataset,
                        workers=12)

    model.save("unet_13_model.h5")
    plot_dice_loss_coefficient(history, "unet_13_dice_loss_coefficient.jpg")
    plot_surface_distance(history, "unet_13_surface_distance.jpg")


if __name__ == '__main__':
    compile_model = compile_model()
    train_model(compile_model)
    pass
