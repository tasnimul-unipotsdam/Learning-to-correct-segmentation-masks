import tensorflow as tf
from tensorflow.keras.models import Model


def conv_layer_3D(x, num_filters, kernel_size, padding='same'):
    x = tf.keras.layers.Conv3D(num_filters,
                               kernel_size=kernel_size,
                               padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    return x


def unet_3D():
    inputs = tf.keras.Input(shape=(128, 224, 224, 1))
    x = inputs
    conv1 = conv_layer_3D(x, num_filters=4, kernel_size=3)
    conv1 = conv_layer_3D(conv1, num_filters=4, kernel_size=3)
    pool1 = tf.keras.layers.MaxPool3D()(conv1)

    conv2 = conv_layer_3D(pool1, num_filters=8, kernel_size=3)
    conv2 = conv_layer_3D(conv2, num_filters=8, kernel_size=3)
    pool2 = tf.keras.layers.MaxPool3D()(conv2)

    conv3 = conv_layer_3D(pool2, num_filters=16, kernel_size=3)
    conv3 = conv_layer_3D(conv3, num_filters=16, kernel_size=3)
    pool3 = tf.keras.layers.MaxPool3D()(conv3)

    conv4 = conv_layer_3D(pool3, num_filters=32, kernel_size=3)
    conv4 = conv_layer_3D(conv4, num_filters=32, kernel_size=3)

    concat1 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv3DTranspose(16, 2, strides=2, padding="same")(conv4), conv3])

    conv5 = conv_layer_3D(concat1, num_filters=16, kernel_size=3)
    conv5 = conv_layer_3D(conv5, num_filters=16, kernel_size=3)

    concat2 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv3DTranspose(8, 2, strides=2, padding="same")(conv5), conv2])

    conv6 = conv_layer_3D(concat2, num_filters=8, kernel_size=3)
    conv6 = conv_layer_3D(conv6, num_filters=8, kernel_size=3)

    concat3 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv3DTranspose(4, 2, strides=2, padding="same")(conv6), conv1])

    conv7 = conv_layer_3D(concat3, num_filters=4, kernel_size=3)
    conv7 = conv_layer_3D(conv7, num_filters=4, kernel_size=3)

    output = tf.keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), activation="sigmoid")(conv7)

    unet_model_3D = tf.keras.Model(inputs, output, name='U-Net_model_3D')

    return unet_model_3D


if __name__ == '__main__':
    model = unet_3D()
    model.summary()

    # dot_img_file = 'model_1.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
