import tensorflow as tf
import tensorflow_addons as tfa
from random import randint


def rotate(corrupt_image, original_image):
    angle = randint(-15, 15)
    concat = tf.concat([corrupt_image, original_image], axis=2)
    concat = tfa.image.rotate(concat, angle)
    corrupt_image, original_image = tf.split(concat, num_or_size_splits=2, axis=2)
    return corrupt_image, original_image


def _random_crop_(corrupt_image, original_image):
    concat = tf.concat([corrupt_image, original_image], axis=2)
    concat = tf.image.random_crop(concat, size=[128, 128, 1 + 1])
    corrupt_image, original_image = tf.split(concat, [1, 1], axis=2)
    return corrupt_image, original_image


def _normalize_(corrupt_image, original_image):
    corrupt_image = corrupt_image / 255
    original_image = original_image / 255
    return corrupt_image, original_image



def train_augmentation(corrupt_image, original_image):
    corrupt_image, original_image = rotate(corrupt_image, original_image)
    corrupt_image, original_image = _random_crop_(corrupt_image, original_image)
    return corrupt_image, original_image


