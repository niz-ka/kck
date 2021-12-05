import cv2
from scipy.ndimage import interpolation as inter
import numpy as np
import sys
import config


# Shorthand for image saving and showing
def save_and_show(filename, img):
    path = config.OUTPUT_DIR
    cv2.imwrite(path + filename, img)
    cv2.imshow(filename, img)
    cv2.waitKey(0)


def import_image(size):
    path = config.INPUT_DIR

    if (len(sys.argv) != 2):
        print("Usage: python3 main.py <filename>")
        print("where <filename> must be in {} directory".format(path))
        exit(0)

    img = cv2.imread(path + sys.argv[1])
    if (img is None):
        print("Error. Cannot read '{}'".format(path + sys.argv[1]))
        exit(-1)

    aspect_ratio = img.shape[1] / img.shape[0]
    width = size[0]
    height = int(width / aspect_ratio)

    return cv2.resize(img, (width, height))