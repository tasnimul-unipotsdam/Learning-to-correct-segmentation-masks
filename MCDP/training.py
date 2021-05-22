import os
import random
import numpy as np
import tensorflow as tf

from pipeline.tfrecord_reader import TFRecordReader
from MCDP import models
from model.metrics_optimizer import *
from model.plot_metrics_loss import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

seeds = 1000

criterion = get_criterion('cross_entropy')
optimizer = get_optimizers('adam')

tf_train_recorder = TFRecordReader("D:/PROJECTS/internship/records", is_training=True)
train_dataset = tf_train_recorder.train_dataset()
tf_valid_recorder = TFRecordReader("D:/PROJECTS/internship/records", is_training=False)
validation_dataset = tf_valid_recorder.validation_dataset()


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(seeds)


def compile_model():
    model = models.unet_mcd()
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=criterion,
                  metrics=[dice_coefficient, robust_hausdorff], run_eagerly=True)
    return model


def train_model(model):
    BATCH_SIZE = 4
    STEPS_PER_EPOCH = 1016 // BATCH_SIZE
    VALIDATION_STEPS = 254 // BATCH_SIZE
    history = model.fit(train_dataset,
                        verbose=2,
                        epochs=20,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=validation_dataset,
                        workers=12)

    model.save("test.h5")

    # plot_dice_loss(history, "unet_20_dice_loss_.jpg")
    plot_crossentropy_loss(history, "test_crossentropy_loss.jpg")

    # plot_metrics(history, "test_metrics.jpg")
    plot_hausdorff(history, "test_metrics.jpg")


if __name__ == '__main__':
    compile_model = compile_model()
    train_model(compile_model)
    pass
