import os
import numpy as np

from model.metrics_optimizer import *

from model_mcdp_3D import unet_mcdp_3D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

threshold = 0.5


def unet_model():
    model = unet_mcdp_3D()
    model.summary()

    model.load_weights('unet_mcdp_3D_weights.h5')
    print("load weight complete")
    return model


def image_label():
    X_ts = np.load("D:/PROJECTS/internship/3D_data/test_input_image.npy")
    Y_ts = np.load("D:/PROJECTS/internship/3D_data/test_label_image.npy")
    return X_ts, Y_ts


def estimate_uncertainty(model, X_ts):

    X_ts[X_ts < threshold] = 0
    X_ts[X_ts >= threshold] = 1

    Y_ts_hat = model.predict(X_ts, batch_size=1)
    print("prediction complete")
    T = 20
    for t in range(T - 1):
        print('model', t + 1, 'of', T - 1)
        Y_ts_hat = Y_ts_hat + model.predict(X_ts, batch_size=1)

    Y_ts_hat = Y_ts_hat / T

    np.save('Y_ts_hat_3D.npy', Y_ts_hat)

    P_foreground = Y_ts_hat
    P_background = 1 - P_foreground
    P_background = np.where(P_background == 0, 0.0001, P_background)

    U_ts = -(P_foreground * np.log(P_foreground) + P_background * np.log(P_background))
    np.save('U_ts_3D.npy', U_ts)

    return Y_ts_hat, U_ts


def dice_coefficient(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def compute_dice(Y_ts, Y_ts_hat):

    Y_ts[Y_ts < threshold] = 0
    Y_ts[Y_ts >= threshold] = 1

    Y_ts_hat[Y_ts_hat < threshold] = 0
    Y_ts_hat[Y_ts_hat >= threshold] = 1

    dice = []
    Ntest = len(Y_ts)
    for i in range(Ntest):
        dice.append(dice_coefficient(Y_ts[i], Y_ts_hat[i]))
    dice = np.array(dice)
    np.save("dice_mcdp_3D.npy", dice)
    return dice


if __name__ == '__main__':
    model = unet_model()
    X_ts, Y_ts = image_label()
    Y_ts_hat, U_ts = estimate_uncertainty(model=model, X_ts=X_ts)
    dice = compute_dice(Y_ts=Y_ts, Y_ts_hat=Y_ts_hat)
    print(dice.mean())
