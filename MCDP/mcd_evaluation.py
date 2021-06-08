import os
import sys

import numpy as np

from model.metrics_optimizer import *
from model.plot_metrics_loss import *

from test_data.test_tfrecord_reader import TFRecordReader
from models import unet_mcd

epsilon = sys.float_info.min
criterion = get_criterion('cross_entropy')
optimizer = get_optimizers('adam')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)


def unet_model():
    model = unet_mcd(0.5, dropout=True)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=criterion,
                  metrics=[dice_coefficient, robust_hausdorff], run_eagerly=True)

    model.load_weights('test.h5')
    print("load weight complete")
    return model


def image_label_pred(model):
    test_dataset = TFRecordReader("D:/PROJECTS/internship/test_records", shuffle=False,
                                  batch_size=4).test_dataset()
    Y_ts = np.concatenate([y for x, y in test_dataset], axis=0)
    X_ts = np.concatenate([x for x, y in test_dataset], axis=0)
    np.save("Y_ts.npy", Y_ts)
    np.save("X_ts", X_ts)
    Y_ts_hat = model.predict(X_ts, batch_size=1)
    print("prediction complete")
    return Y_ts, X_ts, Y_ts_hat


def estimate_uncertainty(Y_ts_hat, model, X_ts):
    T = 20
    for t in range(T - 1):
        print('model', t + 1, 'of', T - 1)
        Y_ts_hat = Y_ts_hat + model.predict(X_ts, batch_size=1)

    Y_ts_hat = Y_ts_hat / T

    np.save('Y_ts_hat.npy', Y_ts_hat)

    P_foreground = Y_ts_hat
    P_background = 1 - P_foreground
    P_background = np.where(P_background == 0, 0.0001, P_background)

    U_ts = -(P_foreground * np.log(P_foreground) + P_background * np.log(P_background))
    np.save('U_ts.npy', U_ts)

    U_ts_foreground = -(P_foreground * np.log(P_foreground))
    np.save("U_ts_foreground.npy", U_ts_foreground)
    U_ts_background = -(P_background * np.log(P_background))
    np.save("U_ts_background.npy", U_ts_background)

    return Y_ts_hat, U_ts, U_ts_foreground, U_ts_background


def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def compute_dice(X_ts, Y_ts, Y_ts_hat):
    dice = []
    Ntest = len(X_ts)
    for i in range(Ntest):
        dice.append(dice_coefficient(Y_ts[i, :, :, 0], Y_ts_hat[i, :, :, 0]))
    dice = np.array(dice)
    np.save("dice.npy", dice)
    return dice


if __name__ == '__main__':
    model = unet_model()
    Y_ts, X_ts, Y_ts_hat = image_label_pred(model=model)
    Y_ts_hat, U_ts, U_ts_foreground, U_ts_background = estimate_uncertainty(Y_ts_hat=Y_ts_hat,
                                                                            model=model, X_ts=X_ts)
    dice = compute_dice(X_ts=X_ts, Y_ts=Y_ts, Y_ts_hat=Y_ts_hat)

    Y_ts_hat = np.load("D:/PROJECTS/internship/MCDP/Y_ts_hat.npy")
    print(Y_ts_hat.shape)
    threshold = 0.5
    Y_ts_hat[Y_ts_hat < threshold] = 0
    Y_ts_hat[Y_ts_hat >= threshold] = 1
    plt.imshow(Y_ts_hat[25, :, :, 0])
    plt.colorbar()
    plt.show()
