import cv2
import glob
import os
import numpy as np
from random import randrange


def depthLabelToRgb(pred):
    colors = {}
    count = 0
    rows,cols = pred.shape
    new_image = np.zeros((rows,cols,3), np.uint8)
    new_image[:] = (255, 255, 255)
    for i in range(rows):
        for j in range(cols):
            k = pred[i,j]

            if not k in colors:
                colors.update({k: (randrange(256), randrange(256), randrange(256))})

            new_image[i, j] = colors.get(k)
    return new_image

