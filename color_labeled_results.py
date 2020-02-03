import glob
import datetime
import os
import cv2
from core import metrics
from core import region_evaluation
from core import helpers
import os.path as osp
from pathlib import Path

files = glob.glob("results/**/*.png", recursive=True)

for imagePath in files:
    if "rednet" in str(imagePath) or "fcn_tensorflow" in str(imagePath) or "fusenet_pytorch" in str(imagePath):
        print('imagePath: ' + imagePath)
        pathRgb = Path(imagePath)
        datasetName = osp.basename(str(pathRgb.parent))
        # print('datasetName: ' + datasetName)
        parentDatasetDir = str(pathRgb.parent)

        coloredDir = str(parentDatasetDir) + 'colored/'
        os.makedirs(coloredDir, exist_ok=True)
        pred = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(coloredDir+os.path.basename(imagePath), helpers.depthLabelToRgb(pred))