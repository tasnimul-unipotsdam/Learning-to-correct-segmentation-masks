import glob
import os

import cv2
import numpy as np


def corruption_image_comparison():
    data_dir = '../femur_2d_segmentations/middle_slices/*'
    image_dir = '../images/'
    files = glob.glob(data_dir)

    image_ix = 0

    while True:
        file = files[image_ix]
        file_name_ext = os.path.basename(file)
        file_names = file_name_ext.split('.')

        original_image = cv2.imread(file)
        cv2.putText(original_image, 'Original', (20, original_image.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

        circle_img = cv2.imread(
            image_dir + file_names[0] + '_circle.' + file_names[1])
        cv2.putText(circle_img, 'Circle', (30, circle_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

        rect_img = cv2.imread(
            image_dir + file_names[0] + '_rect.' + file_names[1])
        cv2.putText(rect_img, 'Rectangle', (0, rect_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

        triangle_img = cv2.imread(
            image_dir + file_names[0] + '_triangle.' + file_names[1])
        cv2.putText(triangle_img, 'Triangle', (20, triangle_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

        ellipse_img = cv2.imread(
            image_dir + file_names[0] + '_ellipse.' + file_names[1])
        cv2.putText(ellipse_img, 'Ellipse', (30, ellipse_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))


        line_img = cv2.imread(
            image_dir + file_names[0] + '_line.' + file_names[1])
        cv2.putText(line_img, 'Line', (50, line_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

        erosion_img = cv2.imread(
            image_dir + file_names[0] + '_erosion.' + file_names[1])
        cv2.putText(erosion_img, 'Erosion', (20, erosion_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

        dilation_img = cv2.imread(
            image_dir + file_names[0] + '_dilation.' + file_names[1])
        cv2.putText(dilation_img, 'Dilation', (20, dilation_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

        img = np.concatenate((
            original_image, circle_img, rect_img, triangle_img,
            ellipse_img,
            line_img, erosion_img, dilation_img), axis=1)
        cv2.imshow('Shapes Comparison', img)
        k = cv2.waitKey(0)

        if k == 97:
            'up'
            if image_ix > 0:
                image_ix = image_ix - 1
            else:
                image_ix = 0
        elif k == 115:
            'down'
            if image_ix == len(files):
                image_ix = 0
            else:
                image_ix = image_ix + 1
        else:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    corruption_image_comparison()
