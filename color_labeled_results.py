import glob
import datetime
import os
import cv2
from core import metrics
from core import region_evaluation
from core import helpers
import os.path as osp
from pathlib import Path
from random import randrange

#colors = {}
#for index, value in enumerate(range(0,256)):
#  colors.update({index: (randrange(256), randrange(256), randrange(256))})

colors = {
    0: (0, 0, 0), #error
    1: (143,188,143), #parede
    2: (0,255,0), #chão
    3: (255,0,0), #armário
    4: (0,255,255), #cama
    5: (255,255,0), #cadeira
    6: (255,0,255), #sofá
    7: (128,128,128), #mesa
    8: (0,0,128), #porta
    9: (0,128,128), #janela
    10: (0,128,0), #estante de livros
    11: (128,0,128), #quadro
    12: (128,128,0), #balcão
    13: (128,0,0), #persianas
    14: (71,99,255), #escrevaninha
    15: (128,128,240), #prateleiras
    16: (0,69,255), #cortina
    17: (32,165,218), #cômoda
    18: (144,238,144), #travesseiro
    19: (237,149,100), #espelho
    20: (130,0,75), #tapete
    21: (180,105,255), #roupas
    22: (179,222,245), #teto
    23: (19,69,139), #livros
    24: (30,105,210), #geladeira
    25: (96,164,244), #tv
    26: (144,128,112), #papel
    27: (222,196,176), #toalha
    28: (0,100,0), #chuveiro
    29: (79,79,47), #caixa
    30: (255,191,0), #conselho de administração
    31: (143,143,188), #pessoa
    32: (238,130,238), #mesa de cabeceira
    33: (50,205,154), #vaso sanitário
    34: (34,34,178), #pia
    35: (60,20,220), #lâmpada
    36: (122,160,255), #banheira
    37: (60,60,60), #saco
    38: (0,0,255) #meio
}

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
        cv2.imwrite(coloredDir+os.path.basename(imagePath), helpers.depthLabelToRgb(pred, colors))