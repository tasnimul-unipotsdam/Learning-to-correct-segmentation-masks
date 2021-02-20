import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K

smooth = 1


def generalized_dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - generalized_dice_coefficient(y_true, y_pred)
    return loss


def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


adam = tf.keras.optimizers.Adam(learning_rate=0.0001)


def plot_dice_loss_coefficient(history, name):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('dice loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['generalized_dice_coefficient'], color='b', label="train_coefficient")
    plt.plot(history.history['val_generalized_dice_coefficient'], color='r',
             label="validation_coefficient")
    plt.title('dice coefficient')
    plt.ylabel('coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    fig.savefig(name)



