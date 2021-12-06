import score.io as io

import os
from pathlib import Path

MODULE_PATH = Path(__file__).parent.resolve()
os.chdir(MODULE_PATH)

def test_import_existing_image():
    IMG_PATH = 'res/test/io.png'
    IMG_SIZE = (320, 200)
    img = io.import_image(IMG_PATH, IMG_SIZE)

    assert img.shape == (200, 320, 3), "Image 'res/test/io.png' should exist and be correctly imported."

def test_import_nonexisting_image():
    IMG_PATH = 'res/test/nonexsisting.png'
    IMG_SIZE = (320, 200)
    img = io.import_image(IMG_PATH, IMG_SIZE)

    assert not img, "Image 'res/test/nonexisting.png' should not exist."