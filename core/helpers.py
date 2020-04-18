import numpy as np


def depthLabelToRgb(pred, colors):
    rows,cols = pred.shape
    new_image = np.zeros((rows,cols,3), np.uint8)
    new_image[:] = (255, 255, 255)
    for i in range(rows):
        for j in range(cols):
            k = pred[i,j]
            new_image[i, j] = colors.get(k)
    return new_image

