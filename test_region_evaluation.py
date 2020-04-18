import cv2
from core import region_evaluation
from core import metrics
from core import helpers
import numpy as np

crop = False
isMl = False
removeZeros = False

#imageName = '000210000010101.png'
#dir_pred = 'results/rednet/active_vision/'
#dir_gt = 'results/gt/active_vision/'
#crop = True

#to report
#imageName = '001510001980101.png'
#dir_pred = 'results/rednet/active_vision/'
#dir_gt = 'results/gt/active_vision/'

#imageName = '001510001980101.png'
#dir_pred = 'results/jcsa_rm/active_vision/'
#dir_gt = 'results/gt/active_vision/'
#crop = True
#isMl = False

#imageName = '04a287849657478ea774727e5bff5202_4.png'
#dir_pred = 'results/jcsa_rm/semantics3d_raw/'
#dir_gt = 'results/gt/semantics3d_raw/'
#crop = False
#isMl = False

#imageName = '04a287849657478ea774727e5bff5202_4.png'
#dir_pred = 'results/graph_canny_segm/semantics3d_raw/'
#dir_gt = 'results/gt/semantics3d_raw/'
#crop = False
#isMl = False
#removeZeros = True

#imageName = '4a7bfe0577f74a1a891683cf5b435f93_4.png'
#dir_pred = 'results/fusenet_pytorch/semantics3d_raw/'
#dir_gt = 'results/gt/semantics3d_raw/'
#crop = False

imageName = '04a287849657478ea774727e5bff5202_2.png'
dir_pred = 'results/rgbd_object_propsal/semantics3d_raw/'
dir_gt = 'results/gt/semantics3d_raw/'


pred = cv2.imread(dir_pred+(imageName.replace('jpg','png')), cv2.IMREAD_GRAYSCALE if isMl else cv2.IMREAD_COLOR)
gt_ori = cv2.imread(dir_gt+(imageName.replace('jpg','png')), cv2.IMREAD_GRAYSCALE)

if crop:
    gt_ori = gt_ori[0:1080, 419:1499]

h, w = gt_ori.shape[:2]
pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

h, w = pred.shape[:2]
gt = cv2.resize(gt_ori, (w, h), interpolation=cv2.INTER_NEAREST)

pred_regions = pred
if isMl:
    pred_regions = region_evaluation.split_classes_to_regions(pred)

gt_regions = region_evaluation.split_classes_to_regions(gt)

pred_classed = pred

if not isMl:
    pred_classed = region_evaluation.join_regions_into_classes(pred, gt)

cv2.imwrite('tests/' + '/pred_resized_' + imageName, pred_resized)
cv2.imwrite('tests/' + '/region_' + imageName, pred_regions)
cv2.imwrite('tests/' + '/gt_ori_' + imageName, helpers.depthLabelToRgb(gt_ori))
cv2.imwrite('tests/' + '/gt_' + imageName, helpers.depthLabelToRgb(gt))
cv2.imwrite('tests/' + '/gt_region_' + imageName, gt_regions)
cv2.imwrite('tests/' + '/region_classed_' + imageName, helpers.depthLabelToRgb(pred_classed))

accuracy_reg, prec_reg, rec_reg, f1_reg, iou_reg, metricsPerClass, totalInstancesGt, totalInstancesPred = region_evaluation.evaluate(pred_regions, gt, gt_regions, 'all', True, removeZeros, False)

print("region type all")
print("totalAccuracy region: ", str(accuracy_reg))
print("totalPrec region: ", str(prec_reg))
print("totalRec region: ", str(rec_reg))
print("totalF1 region: ", str(f1_reg))
print("totalIou region: ", str(iou_reg))
print("total instances gt: ", str(totalInstancesGt))
print("total instances pred: ", str(totalInstancesPred))

for metricPerClass in metricsPerClass.keys():
    print("Class label: ", str(metricPerClass))
    print("Class accuracy: ", str(metricsPerClass[metricPerClass]['accuracy']))
    print("Class precision: ", str(metricsPerClass[metricPerClass]['precision']))
    print("Class recall: ", str(metricsPerClass[metricPerClass]['recall']))
    print("Class f1: ", str(metricsPerClass[metricPerClass]['f1']))
    print("Class iou: ", str(metricsPerClass[metricPerClass]['iou']))

accuracy_reg, prec_reg, rec_reg, f1_reg, iou_reg, metricsPerClass, totalInstancesGt, totalInstancesPred = region_evaluation.evaluate(pred_regions, gt, gt_regions, 'only_best_iou', True, removeZeros, False)

print("region type best iou instance")
print("totalAccuracy region: ", str(accuracy_reg))
print("totalPrec region: ", str(prec_reg))
print("totalRec region: ", str(rec_reg))
print("totalF1 region: ", str(f1_reg))
print("totalIou region: ", str(iou_reg))
print("total instances gt: ", str(totalInstancesGt))
print("total instances pred: ", str(totalInstancesPred))

"""
for metricPerClass in metricsPerClass.keys():
    print("Class label: ", str(metricPerClass))
    print("Class accuracy: ", str(metricsPerClass[metricPerClass]['accuracy']))
    print("Class precision: ", str(metricsPerClass[metricPerClass]['precision']))
    print("Class recall: ", str(metricsPerClass[metricPerClass]['recall']))
    print("Class f1: ", str(metricsPerClass[metricPerClass]['f1']))
    print("Class iou: ", str(metricsPerClass[metricPerClass]['iou']))

accuracy_reg, prec_reg, rec_reg, f1_reg, iou_reg, metricsPerClass = region_evaluation.evaluate(pred_regions, gt, 'only_best_precision', False, True)

print("region type only_best_precision")
print("totalAccuracy region: ", str(accuracy_reg))
print("totalPrec region: ", str(prec_reg))
print("totalRec region: ", str(rec_reg))
print("totalF1 region: ", str(f1_reg))
print("totalIou region: ", str(iou_reg))


for metricPerClass in metricsPerClass.keys():
    print("Class label: ", str(metricPerClass))
    print("Class accuracy: ", str(metricsPerClass[metricPerClass]['accuracy']))
    print("Class precision: ", str(metricsPerClass[metricPerClass]['precision']))
    print("Class recall: ", str(metricsPerClass[metricPerClass]['recall']))
    print("Class f1: ", str(metricsPerClass[metricPerClass]['f1']))
    print("Class iou: ", str(metricsPerClass[metricPerClass]['iou']))
"""

#accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred_classed, gt, 39, score_averaging="weighted")

unique_pixels = set(np.unique(pred_classed))
unique_pixels.update(np.unique(gt))

accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred_classed, gt, len(unique_pixels), labels=list(unique_pixels), score_averaging="macro")

#accuracy2, class_accuracies2, prec2, rec2, f12, iou2 = metrics.evaluate_segmentation(pred_classed, gt, 39, score_averaging="weighted")
print("class label metric")
print("totalAccuracy : ", str(accuracy))
print("totalPrec : %s", str(prec))
print("totalRec : %s", str(rec))
print("totalF1 : %s", str(f1))
print("totalIou : %s", str(iou))