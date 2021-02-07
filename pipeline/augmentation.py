
import tensorflow as tf


def _flip_left_right_(corrupt_mask, mask):
    corrupt_mask = tf.image.random_flip_left_right(corrupt_mask)
    mask = tf.image.random_flip_left_right(mask)
    return corrupt_mask, mask


def _flip_up_down_(corrupt_mask, mask):
    corrupt_mask = tf.image.random_flip_up_down(corrupt_mask)
    mask = tf.image.random_flip_up_down(mask)
    return corrupt_mask, mask


def _normalize_(corrupt_mask, mask):
    corrupt_mask = corrupt_mask / 255
    mask = mask / 255
    return corrupt_mask, mask


def train_augmentation(corrupt_mask, mask):
    corrupt_mask, mask = _flip_left_right_(corrupt_mask, mask)
    corrupt_mask, mask = _flip_up_down_(corrupt_mask, mask)
    corrupt_mask, mask = _normalize_(corrupt_mask, mask)
    return corrupt_mask, mask


def validation_augmentation(corrupt_mask, mask):
    image, mask = _normalize_(corrupt_mask, mask)
    return image, mask
