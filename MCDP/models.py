import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from tensorflow.keras.regularizers import l2


def conv_layer(x, num_filters, kernel_size, padding='same'):
    x = tf.keras.layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def unet_mcd(rate=0.5, dropout=True):
    inputs = tf.keras.Input(shape=(384, 160, 1))
    x = inputs
    conv1 = conv_layer(x, num_filters=64, kernel_size=3)
    conv1 = conv_layer(conv1, num_filters=64, kernel_size=3)
    drop1 = tf.keras.layers.Dropout(rate)(conv1, training=dropout)
    pool1 = tf.keras.layers.MaxPool2D()(drop1)

    conv2 = conv_layer(pool1, num_filters=128, kernel_size=3)
    conv2 = conv_layer(conv2, num_filters=128, kernel_size=3)
    drop2 = tf.keras.layers.Dropout(rate)(conv2, training=dropout)
    pool2 = tf.keras.layers.MaxPool2D()(drop2)

    conv3 = conv_layer(pool2, num_filters=256, kernel_size=3)
    conv3 = conv_layer(conv3, num_filters=256, kernel_size=3)
    drop3 = tf.keras.layers.Dropout(rate)(conv3, training=dropout)
    pool3 = tf.keras.layers.MaxPool2D()(drop3)

    conv4 = conv_layer(pool3, num_filters=512, kernel_size=3)
    conv4 = conv_layer(conv4, num_filters=512, kernel_size=3)

    drop4 = tf.keras.layers.Dropout(rate)(conv4, training=dropout)

    conc1 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same")(drop4), conv3], axis=3)

    conv5 = conv_layer(conc1, num_filters=256, kernel_size=3)
    conv5 = conv_layer(conv5, num_filters=256, kernel_size=3)

    drop5 = tf.keras.layers.Dropout(rate)(conv5, training=dropout)

    conc2 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same")(drop5), conv2], axis=3)

    conv6 = conv_layer(conc2, num_filters=128, kernel_size=3)
    conv6 = conv_layer(conv6, num_filters=128, kernel_size=3)

    drop6 = tf.keras.layers.Dropout(rate)(conv6, training=dropout)

    conc3 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(drop6), conv1], axis=3)

    conv7 = conv_layer(conc3, num_filters=64, kernel_size=3)
    conv7 = conv_layer(conv7, num_filters=64, kernel_size=3)

    output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(conv7)

    unet_model = tf.keras.Model(inputs, output, name='U-Net_model')

    return unet_model


if __name__ == '__main__':
    model = unet_mcd()
    model.summary()

    # dot_img_file = 'model_1.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
