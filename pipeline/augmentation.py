import tensorflow as tf
import imgaug.augmenters as iaa
import tensorflow_addons as tfa
import numpy as np


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


def _augmentations_corrupt_image(corrupt_image):
    corrupt_image = corrupt_image.numpy()
    seq = iaa.Sequential([(
        iaa.Affine(scale=(1.0, 1.1),
                   rotate=(-25, 25),
                   mode='constant'))])
    corrupt_image = seq.augment_image(corrupt_image)
    return corrupt_image


def _training_augmentation_corrupt_image(corrupt_image):
    shape = corrupt_image.get_shape()
    corrupt_image = tf.py_function(_augmentations_corrupt_image, inp=[corrupt_image],
                                   Tout=tf.float32)
    corrupt_image.set_shape(shape)
    return corrupt_image


def _augmentations_original_image(original_image):
    original_image = original_image.numpy()
    seq = iaa.Sequential([(
        iaa.Affine(scale=(1.0, 1.1),
                   rotate=(-25, 25),
                   mode='constant'))])
    original_image = seq.augment_image(original_image)
    return original_image


def _training_augmentation_original_image(original_image):
    shape = original_image.get_shape()
    original_image = tf.py_function(_augmentations_original_image, inp=[original_image],
                                    Tout=tf.float32)
    original_image.set_shape(shape)
    return original_image


def training_aug(corrupt_image, original_image):
    corrupt_image = _training_augmentation_corrupt_image(corrupt_image)
    original_image = _training_augmentation_original_image(original_image)
    return corrupt_image, original_image
