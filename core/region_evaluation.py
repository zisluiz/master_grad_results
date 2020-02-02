from asyncore import dispatcher

from core import metrics
from core import helpers
import numpy as np
import cv2
import math
import os
import glob
from random import randrange

CLASS_VALUE = -1
OUT_CLASS_VALUE = -2
WRITE_REGIONS = False
CLASS_VALUE_SHOW = 255
OUT_CLASS_VALUE_SHOW = 0

MAX_DISTANCE = 1

def evaluate(pred, gt, regionPerClass='only_best_precision', printPerClassMetrics=False, removeTinyRegions=False):
    isPredRgb = len(pred.shape) == 3
    if isPredRgb:
        unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 3)})
    else:
        unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 1)})[:, 0]

    results = {}

    if WRITE_REGIONS:
        for filename in glob.glob("tests/*cropped*"):
            os.remove(filename)

    print('Regions founded: ', str(len(unique_pixels)))
    discartedRegions = 0

    for i, region in enumerate(unique_pixels):
        if isPredRgb:
            indices = np.where(np.all(pred == region, axis=2 if isPredRgb else 1))
        else:
            indices = np.where(pred == region)

        x1 = np.min(indices[0])
        x2 = np.max(indices[0])
        y1 = np.min(indices[1])
        y2 = np.max(indices[1])

        croppedPred = pred[x1:x2+1, y1:y2+1].copy().astype(int)
        croppedGt = gt[x1:x2+1, y1:y2+1].copy().astype(int)

        points = croppedPred.size

        unique_pixels_gt, unique_pixels_count_gt = np.unique(gt[indices], return_counts=True)
        most_frequent_color = unique_pixels_gt[unique_pixels_count_gt == np.max(unique_pixels_count_gt)]

        if removeTinyRegions and points < 5: #len(unique_pixels_gt) == 1 and
            discartedRegions += 1
            #print('Discarted region with shape '+str(croppedPred.shape))
            continue

        if WRITE_REGIONS:
            cv2.imwrite('tests/'+str(i)+'croppedPred.png', croppedPred)
            cv2.imwrite('tests/'+str(i)+'croppedGt.png', helpers.depthLabelToRgb(croppedGt))

        most_frequent_color = np.max(most_frequent_color) #in case of draw

        np.putmask(croppedGt, croppedGt != most_frequent_color, OUT_CLASS_VALUE)
        np.putmask(croppedGt, croppedGt == most_frequent_color, CLASS_VALUE)

        if isPredRgb:
            croppedPred[np.where(np.all(croppedPred == region, axis=2))] = CLASS_VALUE
        else:
            croppedPred[np.where(croppedPred == region)] = CLASS_VALUE

        np.putmask(croppedPred, croppedPred != CLASS_VALUE, OUT_CLASS_VALUE)
        if isPredRgb:
            croppedPred = croppedPred[:, :, 0]

        if WRITE_REGIONS:
            coloredCroppedPred = croppedPred.copy().astype(int)
            coloredCroppedGt = croppedGt.copy().astype(int)
            np.putmask(coloredCroppedPred, coloredCroppedPred == CLASS_VALUE, CLASS_VALUE_SHOW)
            np.putmask(coloredCroppedGt, coloredCroppedGt == CLASS_VALUE, CLASS_VALUE_SHOW)
            np.putmask(coloredCroppedPred, coloredCroppedPred == OUT_CLASS_VALUE, OUT_CLASS_VALUE_SHOW)
            np.putmask(coloredCroppedGt, coloredCroppedGt == OUT_CLASS_VALUE, OUT_CLASS_VALUE_SHOW)
            cv2.imwrite('tests/'+str(i)+'_croppedPred_normalized.png', coloredCroppedPred)
            cv2.imwrite('tests/'+str(i)+'_croppedGt_normalized.png', coloredCroppedGt)

        if len(croppedPred) == 0 or len(croppedPred[0]) == 0 or np.count_nonzero(croppedPred) == 0 or np.all(croppedPred > 1):
            raise Exception('Wrong cropped values')

        np.putmask(croppedPred, croppedPred == CLASS_VALUE, 1)
        np.putmask(croppedPred, croppedPred == OUT_CLASS_VALUE, 0)
        np.putmask(croppedGt, croppedGt == CLASS_VALUE, 1)
        np.putmask(croppedGt, croppedGt == OUT_CLASS_VALUE, 0)

        accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(croppedPred, croppedGt, 2, score_averaging="binary")

        if not results.get(most_frequent_color):
            results[most_frequent_color] = []

        results[most_frequent_color].append({
            'id': i,
            'region': most_frequent_color,
            'points': points,
            'accuracy': accuracy,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'iou': iou
        })

    print('Total grouped regions evaluated: '+str(len(results)))
    print('Total discarted tiny regions: ' + str(discartedRegions))

    totalPoints = 0
    totalAccuracy = 0
    totalPrec = 0
    totalRec = 0
    totalF1 = 0
    totalIou = 0
    metricsPerClass = {}

    for resultRegion in results.keys():
        if regionPerClass == 'all': #'only_best_precision'
            for prediction in results[resultRegion]:
                if totalPoints == 0:
                    totalPoints = prediction['points']
                    totalAccuracy = prediction['accuracy']
                    totalPrec = prediction['precision']
                    totalRec = prediction['recall']
                    totalF1 = prediction['f1']
                    totalIou = prediction['iou']
                else:
                    currentTotalPoints = totalPoints + prediction['points']
                    totalAccuracy = ((totalAccuracy * totalPoints) + (prediction['accuracy'] * prediction['points'])) / currentTotalPoints
                    totalPrec = ((totalPrec * totalPoints) + (prediction['precision'] * prediction['points'])) / currentTotalPoints
                    totalRec = ((totalRec * totalPoints) + (prediction['recall'] * prediction['points'])) / currentTotalPoints
                    totalF1 = ((totalF1 * totalPoints) + (prediction['f1'] * prediction['points'])) / currentTotalPoints
                    totalIou = ((totalIou * totalPoints) + (prediction['iou'] * prediction['points'])) / currentTotalPoints
                    totalPoints = currentTotalPoints

                if not metricsPerClass.get(resultRegion):
                    metricsPerClass[resultRegion] = {
                        'region': resultRegion,
                        'points': prediction['points'],
                        'accuracy': prediction['accuracy'],
                        'precision': prediction['precision'],
                        'recall': prediction['recall'],
                        'f1': prediction['f1'],
                        'iou': prediction['iou']
                    }
                else:
                    metricPerClass = metricsPerClass[resultRegion]
                    totalPoints = metricPerClass['points'] + prediction['points']
                    metricsPerClass[resultRegion] = {
                        'region': metricsPerClass[resultRegion]['region'],
                        'accuracy': ((metricPerClass['accuracy'] * metricPerClass['points']) + (prediction['accuracy'] * prediction['points'])) / totalPoints,
                        'precision': ((metricPerClass['precision'] * metricPerClass['points']) + (prediction['precision'] * prediction['points'])) / totalPoints,
                        'recall': ((metricPerClass['recall'] * metricPerClass['points']) + (prediction['recall'] * prediction['points'])) / totalPoints,
                        'f1': ((metricPerClass['f1'] * metricPerClass['points']) + (prediction['f1'] * prediction['points'])) / totalPoints,
                        'iou': ((metricPerClass['iou'] * metricPerClass['points']) + (prediction['iou'] * prediction['points'])) / totalPoints,
                        'points': totalPoints
                    }
        elif regionPerClass == 'only_best_precision':
            idxBestPrecision = -1
            for idx, prediction in enumerate(results[resultRegion]):
                if idxBestPrecision == -1 or results[resultRegion][idx]['iou'] > results[resultRegion][idxBestPrecision]['iou']:
                    idxBestPrecision = idx

            prediction = results[resultRegion][idxBestPrecision]

            if totalPoints == 0:
                totalPoints = prediction['points']
                totalAccuracy = prediction['accuracy']
                totalPrec = prediction['precision']
                totalRec = prediction['recall']
                totalF1 = prediction['f1']
                totalIou = prediction['iou']
            else:
                currentTotalPoints = totalPoints + prediction['points']
                totalAccuracy = ((totalAccuracy * totalPoints) + (
                            prediction['accuracy'] * prediction['points'])) / currentTotalPoints
                totalPrec = ((totalPrec * totalPoints) + (
                            prediction['precision'] * prediction['points'])) / currentTotalPoints
                totalRec = ((totalRec * totalPoints) + (
                            prediction['recall'] * prediction['points'])) / currentTotalPoints
                totalF1 = ((totalF1 * totalPoints) + (
                            prediction['f1'] * prediction['points'])) / currentTotalPoints
                totalIou = ((totalIou * totalPoints) + (
                            prediction['iou'] * prediction['points'])) / currentTotalPoints
                totalPoints = currentTotalPoints

            metricsPerClass[resultRegion] = {
                'region': resultRegion,
                'points': prediction['points'],
                'accuracy': prediction['accuracy'],
                'precision': prediction['precision'],
                'recall': prediction['recall'],
                'f1': prediction['f1'],
                'iou': prediction['iou']
            }

        if printPerClassMetrics:
            for resultRegion in metricsPerClass.keys():
                print("region: ", str(resultRegion))
                print("totalAccuracy: ", str(metricsPerClass[resultRegion]['accuracy']))
                print("totalPrec: ", str(metricsPerClass[resultRegion]['precision']))
                print("totalRec: ", str(metricsPerClass[resultRegion]['recall']))
                print("totalF1: ", str(metricsPerClass[resultRegion]['f1']))
                print("totalIou: ", str(metricsPerClass[resultRegion]['iou']))


    return totalAccuracy, totalPrec, totalRec, totalF1, totalIou, metricsPerClass

"""
    totalPoints = results[0]['points']
    totalAccuracy = results[0]['accuracy']
    totalPrec = results[0]['precision']
    totalRec = results[0]['recall']
    totalF1 = results[0]['f1']
    totalIou = results[0]['iou']
    metricsPerClass = {}

    for index in range(1, len(results)):
        currentTotalPoints = totalPoints + results[index]['points']
        totalAccuracy = ((totalAccuracy * totalPoints) + (results[index]['accuracy'] * results[index]['points'])) / currentTotalPoints
        totalPrec = ((totalPrec * totalPoints) + (results[index]['precision'] * results[index]['points'])) / currentTotalPoints
        totalRec = ((totalRec * totalPoints) + (results[index]['recall'] * results[index]['points'])) / currentTotalPoints
        totalF1 = ((totalF1 * totalPoints) + (results[index]['f1'] * results[index]['points'])) / currentTotalPoints
        totalIou  = ((totalIou * totalPoints) + (results[index]['iou'] * results[index]['points'])) / currentTotalPoints
        totalPoints += results[index]['points']

        if not metricsPerClass.get(results[index]['region']):
            metricsPerClass[results[index]['region']] = {
                'points': results[index]['points'],
                'accuracy': results[index]['accuracy'],
                'precision': results[index]['precision'],
                'recall': results[index]['recall'],
                'f1': results[index]['f1'],
                'iou': results[index]['iou'],
                'accuracy': results[index]['accuracy'],
            }
        else:
            metricPerClass = metricsPerClass[results[index]['region']]
            totalPoints = metricPerClass['points'] + results[index]['points']
            metricsPerClass[results[index]['region']] = {
                'accuracy': ((metricPerClass['accuracy'] * metricPerClass['points']) + (results[index]['accuracy'] * results[index]['points'])) / totalPoints,
                'precision': ((metricPerClass['precision'] * metricPerClass['points']) + (results[index]['precision'] * results[index]['points'])) / totalPoints,
                'recall': ((metricPerClass['recall'] * metricPerClass['points']) + (results[index]['recall'] * results[index]['points'])) / totalPoints,
                'f1': ((metricPerClass['f1'] * metricPerClass['points']) + (results[index]['f1'] * results[index]['points'])) / totalPoints,
                'iou': ((metricPerClass['iou'] * metricPerClass['points']) + (results[index]['iou'] * results[index]['points'])) / totalPoints,
                'points': totalPoints
            }
"""

def check_pixel_close_region(pred, x1, y1, x2, y2):
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

        for i in range(len(pred)):
            for j in range(len(pred[0])):
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
                            if check_pixel_close_region(pred, x, y, x + i, y + j):
                                pointsToVisit.append([x + i, y + j])

    for x in range(len(predWithRegions)):
        for y in range(len(predWithRegions[0])):
            if np.array_equal(predWithRegions[x][y][:], [0, 0, 0]):
                raise Exception('Exist pixel not visited.')

    print("Finished, colors used: " + str(countColor) + ".")
    return predWithRegions