import cv2 as cv
import numpy as np
import logging

_log = logging.getLogger("applogger")


def import_image(filepath: str, size: tuple[int]):
    img = cv.imread(filepath)
    if img is None:
        _log.critical(f"Cannot load image '{filepath}'")
        return False
    resized = cv.resize(img, size)
    return cv.cvtColor(resized, cv.COLOR_BGR2RGB)


def save_image(filepath: str, img: np.ndarray):
    status = cv.imwrite(filepath, img)
    if status == 0:
        _log.error(f"Cannot save image '{filepath}'.")
        return False
    return True


def show_image(title: str, img: np.ndarray):
    cv.imshow(title, img)
    cv.waitKey(0)


def combine(overlay, background):
    if len(background.shape) == 2:
        background = cv.cvtColor(background, cv.COLOR_GRAY2RGBA)

    if overlay.shape[2] != background.shape[2]:
        _log.info("Assuming RGB color layout.")
        background = cv.cvtColor(background, cv.COLOR_RGB2RGBA)

    if overlay.shape[:2] != background.shape[:2]:
        _log.error(f"Cannot combine images of diffent shapes!\nx"
                   f"Overlay shape: {overlay.shape}\n"
                   f"Background shape: {background.shape}")

    out = np.zeros(background.shape)

    alpha = overlay[:, :, 3] / 255.0
    out[:, :, 0] = (1. - alpha) * background[:, :, 0] + alpha * overlay[:, :, 0]
    out[:, :, 1] = (1. - alpha) * background[:, :, 1] + alpha * overlay[:, :, 1]
    out[:, :, 2] = (1. - alpha) * background[:, :, 2] + alpha * overlay[:, :, 2]
    out[:, :, 3] = (1. - alpha) * background[:, :, 3] + alpha * overlay[:, :, 3]

    return out


def overlay(img, notes, rotation):
    import score.transform as transform

    img = cv.resize(img, None, fx = 3, fy = 3)
    box_canvas = np.zeros((img.shape[0], img.shape[1], 4))
    text_canvas = np.zeros((img.shape[0], img.shape[1], 4))

    for note in notes:
        # Text label example "wiolin 2 (3)" means "g-clef is placed on second staff and third line of this staff"
        text_label = note.description()

        cv.rectangle(box_canvas, (note.x * 3, note.y * 3), (note.x * 3 + note.width * 3, note.y * 3 + note.height * 3), (0, 255, 0, 255), 1)
        cv.putText(text_canvas,
                   text_label, (note.x * 3, note.y * 3),
                   cv.FONT_HERSHEY_COMPLEX,
                   0.5, (255, 0, 0, 255),
                   thickness = 1,
                   lineType = cv.LINE_AA)

    canvas = combine(text_canvas, box_canvas)

    canvas = transform.rotate(canvas, rotation, cval = 0)
    out = combine(canvas, img)

    return out