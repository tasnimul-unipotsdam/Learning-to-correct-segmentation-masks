import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from test_data.test_tfrecord_reader import TFRecordReader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

test_dataset = TFRecordReader("D:/PROJECTS/internship/test_records").test_dataset()

model = tf.keras.models.load_model(
    "D:/PROJECTS/internship/saved model/unet_2_model.h5", compile=False)


def display(display_list):
    fig = plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    fig.savefig("prediction")


def show_predictions(dataset=None, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        pred_mask *= 255.0
        print(pred_mask.min())
        print(pred_mask.max())
        print(np.unique(pred_mask, return_counts=True))
        display([image[0], mask[0], pred_mask[0]])


show_predictions(test_dataset, 3)
