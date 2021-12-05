import cv2 as cv
import numpy as np


def detect_rotation_angle(img: np.ndarray):
    from score.classes import HoughLine

    # Detect lines longer than ~ (0.6 * image.width)
    lines = cv.HoughLines(np.invert(img), 1, np.pi / 360.0, int(img.shape[0] * 0.6))
    lines: list = [HoughLine(line[0][0], line[0][1]) for line in lines]

    # Choose median of detected angles as most probable rotation angle.
    angles = [line.angle for line in lines]
    median = np.median(angles)

    # Due to numerical errors angles are rounded to two decimal points.
    return round(median * 180 / np.pi - 90.0, 2)


def rotate(img, angle, cval = 255, reshape = False):
    from scipy.ndimage.interpolation import rotate

    return rotate(img, angle, reshape = reshape, order = 0, cval = cval)
