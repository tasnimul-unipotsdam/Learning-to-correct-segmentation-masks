import tensorflow as tf
import tensorflow_addons as tfa
from random import randint


def rotate(corrupt_image, original_image):
    angle = randint(-5, 5)
    radians = (angle / 180) * 3.14

    concat = tf.concat([corrupt_image, original_image], axis=2)
    concat = tfa.image.rotate(concat, radians)
    corrupt_image, original_image = tf.split(concat, num_or_size_splits=2, axis=2)
    return corrupt_image, original_image


def _normalize_(corrupt_image, original_image):
    corrupt_image = corrupt_image / 255
    original_image = original_image / 255
    return corrupt_image, original_image
