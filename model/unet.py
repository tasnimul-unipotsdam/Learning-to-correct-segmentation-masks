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

    # x = tf.keras.layers.LeakyReLU()(x)

    return x


def unet():
    inputs = tf.keras.Input(shape=(384, 160, 1))
    x = inputs
    conv1 = conv_layer(x, num_filters=64, kernel_size=3)
    conv1 = conv_layer(conv1, num_filters=64, kernel_size=3)
    pool1 = tf.keras.layers.MaxPool2D()(conv1)

    conv2 = conv_layer(pool1, num_filters=128, kernel_size=3)
    conv2 = conv_layer(conv2, num_filters=128, kernel_size=3)
    pool2 = tf.keras.layers.MaxPool2D()(conv2)

    conv3 = conv_layer(pool2, num_filters=256, kernel_size=3)
    conv3 = conv_layer(conv3, num_filters=256, kernel_size=3)
    pool3 = tf.keras.layers.MaxPool2D()(conv3)

    conv4 = conv_layer(pool3, num_filters=512, kernel_size=3)
    conv4 = conv_layer(conv4, num_filters=512, kernel_size=3)

    conc1 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same")
         (conv4), conv3], axis=3)

    conv5 = conv_layer(conc1, num_filters=256, kernel_size=3)
    conv5 = conv_layer(conv5, num_filters=256, kernel_size=3)

    conc2 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same")
         (conv5), conv2], axis=3)

    conv6 = conv_layer(conc2, num_filters=128, kernel_size=3)
    conv6 = conv_layer(conv6, num_filters=128, kernel_size=3)

    conc3 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")
         (conv6), conv1], axis=3)

    conv7 = conv_layer(conc3, num_filters=64, kernel_size=3)
    conv7 = conv_layer(conv7, num_filters=64, kernel_size=3)

    output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(conv7)

    unet_model = tf.keras.Model(inputs, output, name='U-Net_model')

    return unet_model


def get_unet():
    inputs = Input(shape=(384, 160, 1))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                      axis=3)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                      axis=3)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                      axis=3)

    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1],
                      axis=3)

    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=l2(0.01))(conv9)

    new_model = Model(inputs=inputs, outputs=conv10)

    return new_model


if __name__ == '__main__':
    model = get_unet()
    model.summary()

    # dot_img_file = 'model_1.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
