
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from skimage.transform import rotate


def RandomResize(img, scale=0.12):
    rn = np.random.uniform(1 - scale, 1 + scale)
    h = np.shape(img)[0]
    w = np.shape(img)[1]
    h = int(h * rn)
    w = int(w * rn)
    return img.resize((h, w))

####################################################
def transformations(img, choice):
    # Can try 3 flips (no flip, vertical and horizontal); and 4 rotations (0, 90, 180, 270)


    if choice == 0:
        # Rotate 90
        new_img = img.rotate(270)

    if choice == 1:
        # Rotate 90 and flip horizontally
        new_img = img.rotate(270)
        new_img = new_img.transpose(Image.FLIP_TOP_BOTTOM)

    if choice == 2:
        # Rotate 180
        new_img = img.rotate(180)

    if choice == 3:
        # Rotate 180 and flip horizontally
        new_img = img.rotate(180)
        new_img = new_img.transpose(Image.FLIP_TOP_BOTTOM)

    if choice == 4:
        # Rotate 90 counter-clockwise
        new_img = img.rotate(90)

    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        new_img = img.rotate(90)
        new_img = new_img.transpose(Image.FLIP_TOP_BOTTOM)

    if choice == 6:
        # no transformation
        new_img = img

    return new_img