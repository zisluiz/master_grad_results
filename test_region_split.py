import cv2
from core import region_evaluation

pred = cv2.imread('results/fusenet_pytorch/active_vision/000210000010101.png', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('tests/region_split_original.png', pred)

pred_region = region_evaluation.split_classes_to_regions(pred)

cv2.imwrite('tests/region_split_proposed.png', pred_region)
