import os
import random
import numpy as np
import tensorflow as tf
import pickle

from data_loader import DataLoader3D
from model_mcdp_3D import unet_mcdp_3D
from metrics_loss_optimizer_3D import dice_coefficient
from plot_metrics_loss_3D import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seeds = 1000
seed_everything(seeds)

train_input_image = np.load("D:/PROJECTS/internship/3D_data/train_input_image.npy")
train_label_image = np.load("D:/PROJECTS/internship/3D_data/train_label_image.npy")
val_input_image = np.load("D:/PROJECTS/internship/3D_data/val_input_image.npy")
val_label_image = np.load("D:/PROJECTS/internship/3D_data/val_label_image.npy")

train_dataset = DataLoader3D(inputs=train_input_image, labels=train_label_image,
                             training=True).train_data()

validation_dataset = DataLoader3D(inputs=val_input_image, labels=val_label_image,
                                  training=False).validation_data()


def compile_model():
    model = unet_mcdp_3D(rate=0.5, dropout=False)
    model.summary()
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=adam,
                  loss=binary_crossentropy,
                  metrics=[dice_coefficient], run_eagerly=True)
    return model


def train_model(model):
    BATCH_SIZE = 1
    STEPS_PER_EPOCH = len(train_input_image) // BATCH_SIZE
    VALIDATION_STEPS = len(val_input_image) // BATCH_SIZE
    history = model.fit(train_dataset,
                        verbose=1,
                        epochs=150,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=validation_dataset,
                        workers=12)

    model.save("unet_mcdp_3D_weights.h5")

    plot_crossentropy_loss_dice_coefficient_3D(
        history,
        "unet_mcdp_3D_plot.jpg")

    with open("unet_mcdp_3D_history.pickle",
              'wb') as hist_file:
        pickle.dump(history.history, hist_file)


if __name__ == '__main__':
    compile_model = compile_model()
    train_model(compile_model)
    pass
