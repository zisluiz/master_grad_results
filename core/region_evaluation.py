#from asyncore import dispatcher
from core import metrics
#from core import helpers
import numpy as np
import cv2
import math
import os
import glob
from random import randrange
from collections import Counter
from core import eval_semantic_segmentation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

CLASS_VALUE = -1
OUT_CLASS_VALUE = -2
WRITE_REGIONS = True
CLASS_VALUE_SHOW = 230
OUT_CLASS_VALUE_SHOW = 0

MAX_DISTANCE = 1

def evaluate(pred, gt, gt_regions, regionPerClass='all', discartTinyRegions=False, ignoreValueZero=False, meanPondered=False):
    unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 3)})
    results = {}

    if WRITE_REGIONS:
        for filename in glob.glob("tests/*cropped*"):
            os.remove(filename)

    print('Regions founded: ', str(len(unique_pixels)))
    discartedRegions = 0

    for i, region_pred in enumerate(unique_pixels):
        regionValueZero = np.all(region_pred == 0)
        if ignoreValueZero and regionValueZero:
            continue

        indices = np.where(np.all(pred == region_pred, axis=2))

        points = len(indices[0])

        if discartTinyRegions and points < 11:
            discartedRegions += 1
            continue

        x1p = np.min(indices[0])
        x2p = np.max(indices[0])
        y1p = np.min(indices[1])
        y2p = np.max(indices[1])

        unique_pixels_gt, unique_pixels_count_gt = np.unique(gt[indices], return_counts=True)
        class_gt = unique_pixels_gt[unique_pixels_count_gt == np.max(unique_pixels_count_gt)]
        class_gt = np.max(class_gt)  # in case of draw

        gt_regions_for_pred = gt_regions[indices]
        unique_pixels_region_gt, unique_pixels_count_region_gt = np.unique(gt_regions_for_pred.reshape(-1, gt_regions_for_pred.shape[-1]), axis=0, return_counts=True)
        region_gt = unique_pixels_region_gt[unique_pixels_count_region_gt == np.max(unique_pixels_count_region_gt)][0]

        #unique_pixels_region_gt = np.vstack({tuple(r) for r in gt_regions_for_pred.reshape(-1, 3)})
        #unique_pixels_count_region_gt = []
        #for unique_pixel_region_gt in unique_pixels_region_gt:
        #    nique_pixels_gt, unique_pixels_count_gt = np.unique(gt[indices], return_counts=True)
        #    unique_pixels_count_region_gt.append(len(np.where(np.all(unique_pixels_region_gt == unique_pixel_region_gt, axis=2))))

        #region_gt = unique_pixels_region_gt[unique_pixels_count_region_gt.index(np.max(unique_pixels_count_region_gt))]

        indices_region_gt = np.where(np.all(gt_regions == region_gt, axis=2))

        x1gt = np.min(indices_region_gt[0])
        x2gt = np.max(indices_region_gt[0])
        y1gt = np.min(indices_region_gt[1])
        y2gt = np.max(indices_region_gt[1])

        x1 = x1p if x1p < x1gt else x1gt
        x2 = x2p if x2p > x2gt else x2gt
        y1 = y1p if y1p < y1gt else y1gt
        y2 = y2p if y2p > y2gt else y2gt

        #crop by the bigger object size
        croppedPred = pred[x1:x2 + 1, y1:y2 + 1].copy().astype(int)
        croppedGt = gt_regions[x1:x2 + 1, y1:y2 + 1].copy().astype(int)

        if WRITE_REGIONS:
            cv2.imwrite('tests/'+str(i)+'croppedPred.png', croppedPred)
            cv2.imwrite('tests/'+str(i)+'croppedGt.png', croppedGt)

        #set regions colors to class values
        croppedGt[np.where(np.all(croppedGt == region_gt, axis=2))] = CLASS_VALUE
        croppedGt[np.where(np.all(croppedGt != CLASS_VALUE, axis=2))] = OUT_CLASS_VALUE

        croppedPred[np.where(np.all(croppedPred == region_pred, axis=2))] = CLASS_VALUE
        croppedPred[np.where(np.all(croppedPred != CLASS_VALUE, axis=2))] = OUT_CLASS_VALUE

        #convert 3d color to 1d
        croppedPred = croppedPred[:, :, 0]
        croppedGt = croppedGt[:, :, 0]

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

        flat_pred = croppedPred.flatten()
        flat_label = croppedGt.flatten()

        class_accuracies = metrics.compute_class_accuracies(flat_pred, flat_label, 2)

        prec = precision_score(flat_label, flat_pred, average="binary")
        rec = recall_score(flat_label, flat_pred, average="binary")
        f1 = f1_score(flat_label, flat_pred, average="binary")

        #accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(croppedPred, croppedGt, 2, score_averaging="binary")

        #eval_semantic_results = eval_semantic_segmentation.eval_semantic_segmentation(
        #    np.reshape(croppedPred, (1, croppedPred.shape[0], croppedPred.shape[1])),
        #    np.reshape(croppedGt, (1, croppedGt.shape[0], croppedGt.shape[1])))

        intersection = np.logical_and(croppedPred, croppedGt)
        union = np.logical_or(croppedPred, croppedGt)
        iou = np.sum(intersection) / np.sum(union)

        if not results.get(class_gt):
            results[class_gt] = {}

        if not results[class_gt].get(str(region_gt)):
            results[class_gt][str(region_gt)] = []

        predPoints = len(croppedPred[croppedPred == 1])
        #gtPoints = len(croppedGt[croppedGt == 1])

        results[class_gt][str(region_gt)].append({
            'id': i,
            'predPoints': predPoints,
            #'gtPoints': gtPoints,
            'accuracy': class_accuracies[1],
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'iou': iou
        })

    print('Total grouped regions evaluated: '+str(len(results)))
    print('Total discarted tiny regions: ' + str(discartedRegions))

    metricsPerClass = {}

    for class_region in results.keys():
        ponderedWeight = 0
        totalPoints = 0
        instancesGt = 0
        instancesPred = 0
        ponderedAccuracy = 0.0
        ponderedPrecision = 0.0
        ponderedRecall = 0.0
        ponderedF1 = 0.0
        ponderedIou = 0.0

        for class_instance in results[class_region]:
            instancesGt += 1

            if regionPerClass == 'only_best_iou': #choice best iou region proposal on a instance class
                idxBestPrecision = -1

                for idx, region_pred in enumerate(results[class_region][class_instance]):
                    if idxBestPrecision == -1 or region_pred['iou'] > results[class_region][class_instance][idxBestPrecision]['iou']:
                        idxBestPrecision = idx

                instancesPred += 1
                region_pred = results[class_region][class_instance][idxBestPrecision]
                totalPoints += region_pred['predPoints']

                if meanPondered:
                    currPoints = region_pred['predPoints']
                    ponderedWeight += region_pred['predPoints']
                else:
                    currPoints = 1
                    ponderedWeight += 1

                ponderedAccuracy += region_pred['accuracy'] * currPoints
                ponderedPrecision += region_pred['precision'] * currPoints
                ponderedRecall += region_pred['recall'] * currPoints
                ponderedF1 += region_pred['f1'] * currPoints
                ponderedIou += region_pred['iou'] * currPoints
            else:
                totalRegionPoints = 0
                ponderedRegionAccuracy = 0.0
                ponderedRegionPrecision = 0.0
                ponderedRegionRecall = 0.0
                ponderedRegionF1 = 0.0
                ponderedRegionIou = 0.0

                for region_pred in results[class_region][class_instance]:
                    instancesPred += 1
                    currPoints = region_pred['predPoints']
                    totalRegionPoints += region_pred['predPoints']

                    ponderedRegionAccuracy += region_pred['accuracy'] * currPoints
                    ponderedRegionPrecision += region_pred['precision'] * currPoints
                    ponderedRegionRecall += region_pred['recall'] * currPoints
                    ponderedRegionF1 += region_pred['f1'] * currPoints
                    ponderedRegionIou += region_pred['iou'] * currPoints

                if meanPondered:
                    currPoints = totalRegionPoints
                    ponderedWeight += totalRegionPoints
                else:
                    currPoints = 1
                    ponderedWeight += 1

                ponderedAccuracy += (ponderedRegionAccuracy / totalRegionPoints) * currPoints
                ponderedPrecision += (ponderedRegionPrecision / totalRegionPoints) * currPoints
                ponderedRecall += (ponderedRegionRecall / totalRegionPoints) * currPoints
                ponderedF1 += (ponderedRegionF1 / totalRegionPoints) * currPoints
                ponderedIou += (ponderedRegionIou / totalRegionPoints) * currPoints

        metricsPerClass[class_region] = {
            'points': totalPoints,
            'instancesGt': instancesGt,
            'instancesPred': instancesPred,
            'accuracy': ponderedAccuracy / ponderedWeight,
            'precision': ponderedPrecision / ponderedWeight,
            'recall': ponderedRecall / ponderedWeight,
            'f1': ponderedF1 / ponderedWeight,
            'iou': ponderedIou / ponderedWeight,
        }

    totalAccuracy = 0.0
    totalPrec = 0.0
    totalRec = 0.0
    totalF1 = 0.0
    totalIou = 0.0
    totalInstancesGt = 0
    totalInstancesPred = 0
    numClasses = len(metricsPerClass)

    for idx, metricLabel in enumerate(metricsPerClass):
        totalAccuracy += metricsPerClass[metricLabel]['accuracy']
        totalPrec += metricsPerClass[metricLabel]['precision']
        totalRec += metricsPerClass[metricLabel]['recall']
        totalF1 += metricsPerClass[metricLabel]['f1']
        totalIou += metricsPerClass[metricLabel]['iou']
        totalInstancesGt += metricsPerClass[metricLabel]['instancesGt']
        totalInstancesPred += metricsPerClass[metricLabel]['instancesPred']

    totalAccuracy = totalAccuracy / numClasses
    totalPrec = totalPrec / numClasses
    totalRec = totalRec / numClasses
    totalF1 = totalF1 / numClasses
    totalIou = totalIou / numClasses

    return totalAccuracy, totalPrec, totalRec, totalF1, totalIou, metricsPerClass, totalInstancesGt, totalInstancesPred

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


def join_regions_into_classes(pred, gt):
    unique_pixels = np.vstack({tuple(r) for r in pred.reshape(-1, 3)})
    result = pred.copy().astype(int)

    for i, region_pred in enumerate(unique_pixels):
        indices = np.where(np.all(pred == region_pred, axis=2))

        unique_pixels_gt, unique_pixels_count_gt = np.unique(gt[indices], return_counts=True)
        class_gt = unique_pixels_gt[unique_pixels_count_gt == np.max(unique_pixels_count_gt)]
        class_gt = np.max(class_gt)  # in case of draw

        result[indices] = class_gt

    return result[:, :, 0]