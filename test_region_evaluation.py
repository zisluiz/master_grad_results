import cv2
from core import region_evaluation
from core import metrics
from core import helpers

#imageName = '000210000010101.png'
#dir_pred = 'results/rednet/active_vision/'
#dir_gt = 'results/gt/active_vision/'

imageName = '4a7bfe0577f74a1a891683cf5b435f93_4.png'
dir_pred = 'results/fusenet_pytorch/semantics3d_raw/'
dir_gt = 'results/gt/semantics3d_raw/'

crop = False

pred = cv2.imread(dir_pred+(imageName.replace('jpg','png')), cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(dir_gt+(imageName.replace('jpg','png')), cv2.IMREAD_GRAYSCALE)

if crop:
    gt = gt[0:1080, 419:1499]
h, w = pred.shape[:2]
gt = cv2.resize(gt, (w, h))

pred_regions = pred
pred_regions = region_evaluation.split_classes_to_regions(pred)
cv2.imwrite('tests/' + '/region_' + imageName, pred_regions)
cv2.imwrite('tests/' + '/pred_' + imageName, helpers.depthLabelToRgb(pred))
cv2.imwrite('tests/' + '/gt_' + imageName, helpers.depthLabelToRgb(gt))

accuracy_reg, prec_reg, rec_reg, f1_reg, iou_reg, metricsPerClass = region_evaluation.evaluate(pred_regions, gt, 'all', False, True)

print("region type all")
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


accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred, gt, 39,
                                                                               score_averaging="weighted")
print("class label metric")
print("totalAccuracy : ", str(accuracy))
print("totalPrec : %s", str(prec))
print("totalRec : %s", str(rec))
print("totalF1 : %s", str(f1))
print("totalIou : %s", str(iou))