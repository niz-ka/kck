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

def straighten_helper(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return score

def import_image(size):
    path = config.INPUT_DIR
    
    if(len(sys.argv) != 2):
        print("Usage: python3 main.py <filename>")
        print("where <filename> must be in {} directory".format(path))
        exit(0)

    img = cv2.imread(path + sys.argv[1])
    if(img is None):
        print("Error. Cannot read '{}'".format(path + sys.argv[1]))
        exit(-1)

    return cv2.resize(img, size)