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


def split_classes_to_regions(pred):
    pred = pred.copy()
    predWithRegions = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=int)

    unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 1)})[:, 0]
    countColor = 0
    for i, region in enumerate(unique_pixels):
        indices = np.where(pred == region)
        print("Processing region "+str(i)+", "+str(len(indices[0]))+" points")
        exist_unregion_pixels = True

        while exist_unregion_pixels:
            groupIndices = [[], []]
            x1 = -1
            startColor = (randrange(256), randrange(256), randrange(256))
            countColor += 1
            tryCount = -1
            newGroupIndices = [[], []]

            #verify three times closely pixels
            #for t in range(6):
            restartOnFoundClosePixel = True
            while restartOnFoundClosePixel:
                restartOnFoundClosePixel = False
                found_pixels = False
                tryCount += 1

                if tryCount > 1:
                    groupIndices = newGroupIndices
                    newGroupIndices = [[], []]
                    tryCount = 1

                indices = [np.delete(indices[0], np.argwhere(indices[0] == -1)),
                           np.delete(indices[1], np.argwhere(indices[1] == -1))]

                for j in range(len(indices[0])):
                    if indices[0][j] == -1:
                        continue

                    found_pixels = True
                    if x1 == -1:
                        x1 = indices[0][j]
                        y1 = indices[1][j]
                        groupIndices[0].append(x1)
                        groupIndices[1].append(y1)

                        if not np.array_equal(predWithRegions[x1][y1][:], [0, 0, 0]):
                            raise Exception('Pixel already processed')
                        predWithRegions[x1][y1] = startColor
                        indices[0][j] = -1
                        indices[1][j] = -1
                    else:
                        x2 = indices[0][j]
                        y2 = indices[1][j]

                        isClosest = False

                        for k in range(len(groupIndices[0])):
                            x1 = groupIndices[0][k]
                            y1 = groupIndices[1][k]

                            #dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            dist = math.hypot(x1 - x2, y1 - y2)
                            if abs(dist) <= MAX_DISTANCE:
                                #print("dist: "+str(dist)+" - ("+str(x1)+","+str(y1)+"),("+str(x2)+","+str(y2)+")")
                                isClosest = True
                                break

                        if isClosest:
                            groupIndices[0].insert(0, x2)
                            groupIndices[1].insert(0, y2)
                            newGroupIndices[0].append(x2)
                            newGroupIndices[1].append(y2)

                            if not np.array_equal(predWithRegions[x2][y2][:], [0, 0, 0]):
                                raise Exception('Pixel already processed')

                            predWithRegions[x2][y2] = startColor
                            indices[0][j] = -1
                            indices[1][j] = -1
                            restartOnFoundClosePixel = True #back to start
                            #break

            exist_unregion_pixels = found_pixels

    for x in range(len(predWithRegions[0])):
        for y in range(len(predWithRegions[:,0])):
            if np.array_equal(predWithRegions[x][y][:], [0, 0, 0]):
                raise Exception('Exist pixel not visited.')

    print("Finished, colors used: " + str(countColor) + ".")
    return predWithRegions


def split_classes_to_regions2(pred):
    pred = pred.copy()
    predWithRegions = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=int)
    exist_unregion_pixels = True
    while exist_unregion_pixels:
        exist_unregion_pixels = False

        for x in range(len(pred[0])):
            for y in range(len(pred[:,0])):


    print("Finished, colors used: " + str(countColor) + ".")
    return predWithRegions



    unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 1)})[:, 0]
    countColor = 0
    for i, region in enumerate(unique_pixels):
        indices = np.where(pred == region)
        print("Processing region "+str(i)+", "+str(len(indices[0]))+" points")
        exist_unregion_pixels = True

        while exist_unregion_pixels:
            groupIndices = [[], []]
            x1 = -1
            startColor = (randrange(256), randrange(256), randrange(256))
            countColor += 1
            tryCount = -1
            newGroupIndices = [[], []]

            #verify three times closely pixels
            #for t in range(6):
            restartOnFoundClosePixel = True
            while restartOnFoundClosePixel:
                restartOnFoundClosePixel = False
                found_pixels = False
                tryCount += 1

                if tryCount > 1:
                    groupIndices = newGroupIndices
                    newGroupIndices = [[], []]
                    tryCount = 1

                indices = [np.delete(indices[0], np.argwhere(indices[0] == -1)),
                           np.delete(indices[1], np.argwhere(indices[1] == -1))]

                for j in range(len(indices[0])):
                    if indices[0][j] == -1:
                        continue

                    found_pixels = True
                    if x1 == -1:
                        x1 = indices[0][j]
                        y1 = indices[1][j]
                        groupIndices[0].append(x1)
                        groupIndices[1].append(y1)

                        if not np.array_equal(predWithRegions[x1][y1][:], [0, 0, 0]):
                            raise Exception('Pixel already processed')
                        predWithRegions[x1][y1] = startColor
                        indices[0][j] = -1
                        indices[1][j] = -1
                    else:
                        x2 = indices[0][j]
                        y2 = indices[1][j]

                        isClosest = False

                        for k in range(len(groupIndices[0])):
                            x1 = groupIndices[0][k]
                            y1 = groupIndices[1][k]

                            #dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            dist = math.hypot(x1 - x2, y1 - y2)
                            if abs(dist) <= MAX_DISTANCE:
                                #print("dist: "+str(dist)+" - ("+str(x1)+","+str(y1)+"),("+str(x2)+","+str(y2)+")")
                                isClosest = True
                                break

                        if isClosest:
                            groupIndices[0].insert(0, x2)
                            groupIndices[1].insert(0, y2)
                            newGroupIndices[0].append(x2)
                            newGroupIndices[1].append(y2)

                            if not np.array_equal(predWithRegions[x2][y2][:], [0, 0, 0]):
                                raise Exception('Pixel already processed')

                            predWithRegions[x2][y2] = startColor
                            indices[0][j] = -1
                            indices[1][j] = -1
                            restartOnFoundClosePixel = True #back to start
                            #break

            exist_unregion_pixels = found_pixels

    for x in range(len(predWithRegions[0])):
        for y in range(len(predWithRegions[:,0])):
            if np.array_equal(predWithRegions[x][y][:], [0, 0, 0]):
                raise Exception('Exist pixel not visited.')

    print("Finished, colors used: " + str(countColor) + ".")
    return predWithRegions