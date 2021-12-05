import cv2 as cv
import numpy as np
import logging

_log = logging.getLogger('applogger')


def desaturate(img: np.ndarray):
    if len(img.shape) == 2:
        return img
    elif img.shape[2] == 3:
        _log.info("Assuming RGB color layout.")
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return cv.fastNlMeansDenoising(gray)
    elif img.shape[2] == 4:
        _log.info("Assuming RGBA color layout.")
        gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
        return cv.fastNlMeansDenoising(gray)
    else:
        _log.error("Incorrect image format. Cannot desaturate.")
        return False


def binarize(img, block_size = 51, offset = 10, filter = (9, 75, 75)):
    if len(img.shape) != 2:
        _log.error("Cannot binarize non-grayscale image.")
        return False

    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, offset)
    return cv.bilateralFilter(thresh, *filter)


def find_staff_bounds(img):
    from pipe import where, select
    from score.classes import HoughLine, Staff
    import itertools
    import more_itertools as iter

    def find_approx_staff_height(lines):
        if len(lines) < 2:
            _log.error("Cannot infer staff height from less than 2 lines.")
            return False

        gradient = [lines[i].y - lines[i - 1].y for i in range(1, len(lines))]
        threshold = np.median(gradient) + np.std(gradient)
        spaces = list(enumerate(gradient) | where(lambda val: val[1] > threshold) | select(lambda val: val[0]))
        indices = sorted(list(itertools.chain(*[[i, i + 1] for i in spaces])) + [0, len(lines) - 1])
        heights = [lines[j].distance - lines[i].distance for i, j in iter.grouper(indices, 2, indices[-1])]

        return np.median(heights), indices

    # Find long horizontal lines (most probable staff lines).
    PI_HALF = np.pi / 2
    lines = cv.HoughLines(img, 1, np.pi / 360, img.shape[1] // 5)
    lines = list(
        sorted([HoughLine(line[0][0], line[0][1]) for line in lines], key = lambda line: line.y)
        | where(lambda line: abs(line.angle - PI_HALF) < 0.0001))

    height, indices = find_approx_staff_height(lines)
    _log.debug(f"Found staff height: {height}")

    return [Staff(lines[index].y, height, i) for i, index in enumerate(indices[::2])]


def detect_horizontal_lines(img):
    img = np.invert(img)

    horizontal_size = img.shape[1] // 30
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    detected_lines = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations = 1)

    return detected_lines


def detect_vertical_lines(img):
    img = np.invert(img)

    vertical_size = img.shape[0] // 30
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    detected_lines = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations = 1)

    return detected_lines


# Remove horizontal lines and repair through vertical lines
def remove_lines(img, horizontal_lines, vertical_lines):
    img = cv.bitwise_or(img, horizontal_lines)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 2))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    img = np.invert(cv.bitwise_or(np.invert(img), vertical_lines))

    return img


# Find rectangles of objects that have area between min_area and max_area
def find_bounding_rectangles(img, min_area, max_area):
    count, labels, stats = cv.connectedComponentsWithStats(np.invert(img))[:3]
    areas = stats[:, 4]

    for label in range(1, count):
        if areas[label] > max_area or areas[label] < min_area:
            labels[labels == label] = 0
    labels[labels > 0] = 255

    stats = cv.connectedComponentsWithStats(labels.astype(np.uint8))[2]

    return [[x, y, w, h] for x, y, w, h, *_ in stats[1:]]