import cv2 as cv
import numpy as np
from scipy.ndimage.interpolation import rotate


def detect_rotation_angle(img: np.ndarray):
    from score.classes import HoughLine

    # Detect lines longer than ~ (0.6 * image.width)
    lines = cv.HoughLines(np.invert(img), 1, np.pi / 360.0, int(img.shape[0] * 0.6))
    lines: list = [HoughLine(line[0][0], line[0][1]) for line in lines]

    # Choose median of detected angles as most probable rotation angle.
    angles = [line.angle for line in lines]
    median = np.median(angles)

    # Due to numerical errors angles are rounded to two decimal points.
    return round(median * 180 / np.pi - 90.0, 5)


def rotate_img(img, angle, cval = 255, reshape = False):
    return rotate(img, angle, reshape = reshape, order = 0, cval = cval)


def straighten_by_weight(bin_img, delta, limit):
    def straighten_helper(arr, angle):
        data = rotate(arr, angle, reshape = False, order = 0)
        hist = np.sum(data, axis = 1, dtype = float)
        score = np.sum((hist[1:] - hist[:-1])**2, dtype = float)
        return score

    bin_img = np.invert(bin_img)
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        score = straighten_helper(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle:', best_angle)
    data = rotate(bin_img, best_angle, reshape = False, order = 0)
    img = np.invert(np.array(data).astype(np.uint8))

    return img