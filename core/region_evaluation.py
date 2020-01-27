from core import metrics
from collections import Counter
import numpy as np
import cv2
import math
from random import randrange
CLASS_VALUE = 255
OUT_CLASS_VALUE = 0
MAX_DISTANCE = 1

def evaluate(pred, gt):
    isPredRgb = len(pred.shape) == 3
    if isPredRgb:
        unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 3)})
    else:
        unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 1)})[:, 0]

    totalAccuracy = 0
    totalPrec = 0
    totalRec = 0
    totalF1 = 0
    totalIou = 0
    numRegions = len(unique_pixels)

    for i, region in enumerate(unique_pixels):
        if isPredRgb:
            indices = np.where(np.all(pred == region, axis=2 if isPredRgb else 1))
        else:
            indices = np.where(pred == region)

        x1 = np.min(indices[0])
        x2 = np.max(indices[0])
        y1 = np.min(indices[1])
        y2 = np.max(indices[1])

        croppedPred = pred[x1:x2+1, y1:y2+1].copy()
        croppedGt = gt[x1:x2+1, y1:y2+1].copy()

        #cv2.imwrite('tests/'+str(i)+'croppedPred.png', croppedPred)
        #cv2.imwrite('tests/'+str(i)+'croppedGt.png', croppedGt)

        unique_pixels_gt, unique_pixels_count_gt = np.unique(gt[indices], return_counts=True)
        most_frequent_color = unique_pixels_gt[unique_pixels_count_gt == np.max(unique_pixels_count_gt)]

        most_frequent_color = np.max(most_frequent_color) #in case of draw

        np.putmask(croppedGt, croppedGt != most_frequent_color, OUT_CLASS_VALUE)
        np.putmask(croppedGt, croppedGt == most_frequent_color, CLASS_VALUE)

        if isPredRgb:
            croppedPred[np.where(np.all(croppedPred == region, axis=2))] = CLASS_VALUE#most_frequent_color
        else:
            croppedPred[np.where(croppedPred == region)] = CLASS_VALUE  # most_frequent_color
        #for index in indices:
        #    newIndexX = index[0] - x1
        #    newIndexY = index[1] - y1
        #    croppedPred[newIndexX][newIndexY] = 255#most_frequent_color

        np.putmask(croppedPred, croppedPred != CLASS_VALUE, OUT_CLASS_VALUE)#most_frequent_color, 0)
        if isPredRgb:
            croppedPred = croppedPred[:, :, 0]

        #cv2.imwrite('tests/'+str(i)+'_croppedPred_normalized.png', croppedPred)
        #cv2.imwrite('tests/'+str(i)+'_croppedGt_normalized.png', croppedGt)

        #accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(croppedPred, croppedGt, 2, score_averaging="weighted")
        accuracy, class_accuracies, prec, rec, f1, iou = 0,0,0,0,0,0
        totalAccuracy += accuracy
        totalPrec += prec
        totalRec += rec
        totalF1 += f1
        totalIou += iou

    totalAccuracy = totalAccuracy / numRegions
    totalPrec = totalPrec / numRegions
    totalRec = totalRec / numRegions
    totalF1 = totalF1 / numRegions
    totalIou = totalIou / numRegions

    return totalAccuracy, totalPrec, totalRec, totalF1, totalIou


def check_pixel_close_region(pred, region, x1, y1, x2, y2):
    if x2 > - 1 and x2 < len(pred) and y2 > -1 and y2 < len(pred[0]) and pred[x2][y2] != -1:
        dist = math.hypot(x1 - x2, y1 - y2)
        if abs(dist) <= MAX_DISTANCE:
            return True

    return False


def split_classes_to_regions(pred):
    pred = pred.copy().astype(int)
    predWithRegions = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=int)
    exist_unregion_pixels = True
    countColor = 0
    while exist_unregion_pixels:
        exist_unregion_pixels = False
        region = None
        startColor = (randrange(256), randrange(256), randrange(256))
        pointsToVisit = []
        countColor += 1

        for i in range(len(pred[0])):
            for j in range(len(pred[:,0])):
                if pred[i][j] != -1:
                    pointsToVisit.append([i, j])
                    region = pred[i][j]
                    exist_unregion_pixels = True
                    break
            if exist_unregion_pixels:
                break

        while len(pointsToVisit) > 0:
            point = pointsToVisit.pop()
            x = point[0]
            y = point[1]

            if pred[x][y] == region:
                predWithRegions[x][y] = startColor
                pred[x][y] = -1

                for i in range(-MAX_DISTANCE, MAX_DISTANCE + 1, 1):
                    for j in range(-MAX_DISTANCE, MAX_DISTANCE + 1, 1):
                        if i != 0 or j != 0:
                            if check_pixel_close_region(pred, region, x, y, x + i, y + j):
                                pointsToVisit.append([x + i, y + j])
                    """
                    if check_pixel_close_region(pred, region, x, y, x - i, y - i):
                        pointsToVisit.append([x - i, y - i])
                    if check_pixel_close_region(pred, region, x, y, x + i, y + i):
                        pointsToVisit.append([x + i, y + i])                        
                    if check_pixel_close_region(pred, region, x, y, x - 1, y - 1):
                        pointsToVisit.append([x - 1, y - 1])
                    if check_pixel_close_region(pred, region, x, y, x, y - 1):
                        pointsToVisit.append([x, y - 1])
                    if check_pixel_close_region(pred, region, x, y, x+1, y - 1):
                        pointsToVisit.append([x+1, y - 1])
                    if check_pixel_close_region(pred, region, x, y, x - 1, y):
                        pointsToVisit.append([x - 1, y])
                    if check_pixel_close_region(pred, region, x, y, x+1, y):
                        pointsToVisit.append([x+1, y])
                    if check_pixel_close_region(pred, region, x, y, x - 1, y + 1):
                        pointsToVisit.append([x - 1, y + 1])
                    if check_pixel_close_region(pred, region, x, y, x, y + 1):
                        pointsToVisit.append([x, y + 1])
                    if check_pixel_close_region(pred, region, x, y, x + 1, y + 1):
                        pointsToVisit.append([x + 1, y + 1])
"""
    for x in range(len(predWithRegions[0])):
        for y in range(len(predWithRegions[:,0])):
            if np.array_equal(predWithRegions[x][y][:], [0, 0, 0]):
                raise Exception('Exist pixel not visited.')

    print("Finished, colors used: " + str(countColor) + ".")
    return predWithRegions