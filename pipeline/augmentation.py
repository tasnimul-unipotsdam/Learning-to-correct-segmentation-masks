import tensorflow as tf


def _flip_left_right_(corrupt_image, original_image):
    corrupt_image = tf.image.random_flip_left_right(corrupt_image)
    original_image = tf.image.random_flip_left_right(original_image)
    return corrupt_image, original_image


def _flip_up_down_(corrupt_image, original_image):
    corrupt_image = tf.image.random_flip_up_down(corrupt_image)
    original_image = tf.image.random_flip_up_down(original_image)
    return corrupt_image, original_image


def _normalize_(corrupt_image, original_image):
    corrupt_image = corrupt_image / 255
    original_image = original_image / 255
    return corrupt_image, original_image


def train_augmentation(corrupt_image, original_image):
    corrupt_image, original_image = _flip_left_right_(corrupt_image, original_image)
    corrupt_image, original_image = _flip_up_down_(corrupt_image, original_image)
    return corrupt_image, original_image


