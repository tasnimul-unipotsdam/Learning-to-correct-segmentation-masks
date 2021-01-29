import glob
import os
import skimage.io
from skimage.morphology import convex_hull_image
import shutil
import cv2
import numpy as np


def create_random_circle(img, x, y, w, h, radius=20):
    new_x = np.random.randint(x, (x + w), 1)
    new_y = np.random.randint(y, (y + h), 1)
    color = (0, 0, 0)
    for xx, yy in zip(new_x, new_y):
        center_coordinates = (int(xx), int(yy))
        img = cv2.circle(img, center_coordinates, radius, color, -1)

    return img


def create_random_rect(img, x, y, w, h):
    new_x = np.random.randint(x, (x + w), 1)
    new_y = np.random.randint(y, (y + h), 1)
    color = (0, 0, 0)
    for xx, yy in zip(new_x, new_y):
        val = np.random.randint(-5, 5)
        if val < 0:
            img = cv2.rectangle(img, (xx, yy), (xx - 30, yy + 30), color, -1)
        else:
            img = cv2.rectangle(img, (xx, yy), (xx + 30, yy + 30), color, -1)

    return img


def create_random_triangle(img, x, y, w, h):
    new_x = np.random.randint(x, (x + w), 1)
    new_y = np.random.randint(y, (y + h), 1)
    color = (0, 0, 0)
    for xx, yy in zip(new_x, new_y):
        val = np.random.randint(-5, 5)
        if val < 0:
            img = cv2.drawContours(img, [np.array([(xx + 10, yy + 20), (xx + 25, yy + 30), (xx - 30, yy + 35)])],  -1, color, -1)
        else:
             img = cv2.drawContours(img, [np.array([(xx - 10, yy - 20), (xx + 25, yy + 30), (xx + 30, yy - 35)])], 0, color, -1)


    return img


def create_random_ellipse(img, x, y, w, h):
    new_x = np.random.randint(x, (x + w), 1)
    new_y = np.random.randint(y, (y + h), 1)
    color = (0, 0, 0)
    for xx, yy in zip(new_x, new_y):
        img = cv2.ellipse(img, (xx, yy), (30, 10), 0, 0, 360, color, -1)

    return img


def create_random_line(img, x, y, w, h):
    new_x = np.random.randint(x, (x + w), 1)
    new_y = np.random.randint(y, (y + h), 1)
    color = (0, 0, 0)
    for xx, yy in zip(new_x, new_y):
        # end_x = xx + np.random.randint(10, 20, 1)
        if yy < 20:
            end_y = yy
        else:
            end_y = np.random.randint(yy - 20, yy + 20, 1)

        val = np.random.randint(-5, 5)
        if val < 0:
            img = cv2.line(img, (xx, yy), (xx - 20, end_y), color, 5)
        else:
            img = cv2.line(img, (xx, yy), (xx + 20, end_y), color, 5)

    return img


def erosion(img):
    kernel_size = (np.random.randint(8, 20), np.random.randint(8, 20))
    kernel = np.ones(kernel_size, np.uint8)
    iteration = np.random.randint(5, 15)
    img = cv2.erode(img, kernel, iteration)
    return img


def dilation(img):
    kernel_size = (np.random.randint(8, 20), np.random.randint(8, 20))
    kernel = np.ones(kernel_size, np.uint8)
    iteration = np.random.randint(5, 15)
    img = cv2.dilate(img, kernel, iteration)
    return img



def convex(img):
    img = convex_hull_image(img)
    img = img.astype(np.float32)
    return img



def write_images(shape_type, file_path, img):
    names = os.path.basename(file_path).split('.')
    dest_file_name = names[0] + '_' + shape_type + '.' + names[1]
    if shape_type == 'convex':
        skimage.io.imsave(os.path.join(dest_dir, dest_file_name), img)
    else:
        cv2.imwrite(os.path.join(dest_dir, dest_file_name), img)


if __name__ == '__main__':
    data_dir = '../femur_2d_segmentations/middle_slices/'
    dest_dir = '../images'

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    os.mkdir(dest_dir)

    files = glob.glob(os.path.join(data_dir, '*'))


    for file in files:
        image = cv2.imread(file, 0)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        x, y, w, h = cv2.boundingRect(thresh)

        # img = cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),3)

        # # Circle
        img = create_random_circle(image.copy(), x, y, w, h)
        write_images(shape_type='circle', file_path=file, img=img)

        # # Rectangle
        img = create_random_rect(image.copy(), x, y, w, h)
        write_images(shape_type='rect', file_path=file, img=img)

        # # Ellipse
        img = create_random_ellipse(image.copy(), x, y, w, h)
        write_images(shape_type='ellipse', file_path=file, img=img)

        # # line
        img = create_random_line(image.copy(), x, y, w, h)
        write_images(shape_type='line', file_path=file, img=img)

        ## erosion
        img = erosion(image.copy())
        write_images(shape_type='erosion', file_path=file, img=img)

        ## dilation
        img = dilation(image.copy())
        write_images(shape_type='dilation', file_path=file, img=img)

        ## triangle
        img = create_random_triangle(image.copy(), x, y, w, h)
        write_images(shape_type='triangle', file_path=file, img=img)


        img = convex(image.copy())
        write_images(shape_type='convex', file_path=file, img=img)

