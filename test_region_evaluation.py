import cv2
from core import region_evaluation

pred = cv2.imread('results/rgbd_object_propsal/active_vision/000210000010101.png')
gt = cv2.imread('results/gt/active_vision/000210000010101.png', cv2.IMREAD_GRAYSCALE)

gt = gt[0:1080, 419:1499]
h, w = pred.shape[:2]
gt = cv2.resize(gt, (h, w))

totalAccuracy, totalPrec, totalRec, totalF1, totalIou = region_evaluation.evaluate(pred, gt)

print("totalAccuracy : ", str(totalAccuracy))
print("totalPrec : %d", str(totalPrec))
print("totalRec : %d", str(totalRec))
print("totalF1 : %d", str(totalF1))
print("totalIou : %d", str(totalIou))