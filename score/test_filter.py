import score.filter as filter
import score.io as io
import os
from pathlib import Path

MODULE_PATH = Path(__file__).parent.resolve()
os.chdir(MODULE_PATH)

def test_desaturate_rgb_image():
    IMG_PATH = 'res/test/io.png'
    IMG_SIZE = (320, 200)
    img = io.import_image(IMG_PATH, IMG_SIZE)
    gray = filter.desaturate(img)

    assert gray.shape == (200, 320), "Image 'res/test/io.png' should exist and be correctly desaturated."