import os

import tensorflow as tf
from pipeline.tfrecord_reader import TFRecordReader
from model.unet import unet
from model.metrics_optimizer import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

train_dataset = TFRecordReader("D:/PROJECTS/internship/records", is_training=True).train_dataset()
validation_dataset = TFRecordReader("D:/PROJECTS/internship/records",
                                    is_training=False).validation_dataset()


def compile_model():
    model = unet()
    model.summary()
    model.compile(optimizer=adam,
                  loss=dice_loss,
                  metrics=[generalized_dice_coefficient])
    return model


def train_model(model):
    history = model.fit(train_dataset,
                        verbose=2,
                        epochs=10,
                        validation_data=validation_dataset,
                        workers=12)

    model.save("unet_4_model.h5")
    plot_dice_loss_coefficient(history, "unet_4_dice_loss_coefficient.jpg")


if __name__ == '__main__':
    compile_model = compile_model()
    train_model(compile_model)
    pass