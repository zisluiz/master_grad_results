import glob
import datetime
import os
import cv2
import numpy as np
#from core import eval_semantic_segmentation
from core import metrics
from core import region_evaluation
#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.neighbors import KNeighborsClassifier

START_TIME_LABEL = "=== Start time: "
END_TIME_LABEL = "=== End time: "
MAX_NUM_THREADS_LABEL = "=== Max num threads: "
SECONDS_PER_IMAGE_LABEL = "=== Seconds per image: "
GPU_USAGE_PERCENT_LABEL = "GPU Usage Percent: "
GPU_MEM_USAGE_LABEL = "GPU Mem Usage (MB)): "
PROCESS_MEMORY_USAGE_LABEL = "Process memory usage: "
PROCESS_CPU_PERCENT_LABEL = "Process CPU Percent: "
TOTAL_IMAGE_PREDICTED_LABEL = "=== Total image predicted: "

classesList = ["__ignore__", "parede", "chao", "armario", "cama", "cadeira", "sofa", "mesa", "porta", "janela",
    "estante_livros", "quadro", "balcao", "persianas", "escrivaninha_carteira", "prateleira", "cortina", "comoda",
    "travesseiro", "espelho", "tapete", "roupa", "teto", "livro", "geladeira", "tv", "papel", "toalha",
    "cortina_chuveiro", "caixa", "quadro_aviso", "pessoa", "mesa_cabeceira_cama", "vaso_sanitario", "pia", "lampada",
   "banheira", "sacola_mochila", "mean"]


def convertClassDic(class_accuracies):
    classes = {}
    for idx, label in enumerate(class_accuracies):
        classes[idx] = {
            'points': 1,
            'accuracy': label,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'iou': 0
        }

    return classes


def insertMetrics(metricsmap, metric_type, accuracy, prec, rec, f1, iou, metricsPerClass):
    metric = metricsmap.get(metric_type)
    if not metric:
        metricsmap[metric_type] = {
            'accuracy': accuracy,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'iou': iou,
            'classes': metricsPerClass
        }
    else:
        classes = metric['classes']

        for classMetric in metricsPerClass:
            classMetricMap = classes.get(classMetric)

            if not classMetricMap:
                classes[classMetric] = {
                    'points': metricsPerClass[classMetric]['points'],
                    'accuracy': metricsPerClass[classMetric]['accuracy'],
                    'precision': metricsPerClass[classMetric]['precision'],
                    'recall': metricsPerClass[classMetric]['recall'],
                    'f1': metricsPerClass[classMetric]['f1'],
                    'iou': metricsPerClass[classMetric]['iou']
                }
            else:
                totalPoints = classMetricMap['points'] + metricsPerClass[classMetric]['points']
                classes[classMetric] = {
                    'points': totalPoints,
                    'accuracy': ((classMetricMap['accuracy'] * classMetricMap['points']) + (metricsPerClass[classMetric]['accuracy'] * metricsPerClass[classMetric]['points'])) / totalPoints,
                    'precision': ((classMetricMap['precision'] * classMetricMap['points']) + (metricsPerClass[classMetric]['precision'] * metricsPerClass[classMetric]['points'])) / totalPoints,
                    'recall': ((classMetricMap['recall'] * classMetricMap['points']) + (metricsPerClass[classMetric]['recall'] * metricsPerClass[classMetric]['points'])) / totalPoints,
                    'f1': ((classMetricMap['f1'] * classMetricMap['points']) + (metricsPerClass[classMetric]['f1'] * metricsPerClass[classMetric]['points'])) / totalPoints,
                    'iou': ((classMetricMap['iou'] * classMetricMap['points']) + (metricsPerClass[classMetric]['iou'] * metricsPerClass[classMetric]['points'])) / totalPoints
                }

        metricsmap[metric_type] = {
            'accuracy': (accuracy + metric['accuracy']) / 2,
            'precision': (prec + metric['precision']) / 2,
            'recall': (rec + metric['recall']) / 2,
            'f1': (f1 + metric['f1']) / 2,
            'iou': (iou + metric['iou']) / 2,
            'classes': classes
        }

methods = glob.glob("results/*", recursive=True)

for method in methods:
    methodName = os.path.basename(method)
    if methodName == "gt" or methodName != "fusenet_pytorch":
        continue

    print("Processing method " + methodName)
    isClassPrediction = methodName == "fcn_tensorflow" or methodName == "fusenet_pytorch" or methodName == "rednet"  # prediction with class labeled

    with open(method+"/summary.txt", "w+") as summary:
        totalElapsedTime = 0
        totalThreads = 0
        numThreads = 0
        secondsPerImage = 0.0
        totalGpuPercent = 0
        numGpuPercent = 0
        totalGpuMemUsage = 0
        numGpuMemUsage = 0
        totalProcessMemUsage = 0
        numProcessMemUsage = 0
        totalProcessedImages = 0
        numProcessedImages = 0

        currentCpuCore = 0
        totalProcessCpuPercent = 0
        numProcessCpuPercent = 0
        totalCpuPercentCore0 = 0
        numCpuPercentCore0 = 0
        totalCpuPercentCore1 = 0
        numCpuPercentCore1 = 0
        totalCpuPercentCore2 = 0
        numCpuPercentCore2 = 0
        totalCpuPercentCore3 = 0
        numCpuPercentCore3 = 0

        log_running_files = glob.glob(method + "/run_*.txt")
        numLogFiles = len(log_running_files)

        for log_run in log_running_files:
            startTime = None
            endTime = None
            currentCpuCore = 0

            with open(log_run) as search:
                for line in search:
                    line = line.rstrip()  # remove '\n' at end of line

                    if line.startswith(START_TIME_LABEL):
                        startTime = line[len(START_TIME_LABEL):len(line)]
                    if line.startswith(END_TIME_LABEL):
                        endTime = line[len(END_TIME_LABEL):len(line)]
                    if line.startswith(TOTAL_IMAGE_PREDICTED_LABEL):
                        totalProcessedImages += int(line[len(TOTAL_IMAGE_PREDICTED_LABEL):len(line)])
                        numProcessedImages += 1
                    if line.startswith(MAX_NUM_THREADS_LABEL):
                        totalThreads += int(line[len(MAX_NUM_THREADS_LABEL):len(line)])
                        numThreads += 1
                    if line.startswith(SECONDS_PER_IMAGE_LABEL):
                        secondsPerImage += float(line[len(SECONDS_PER_IMAGE_LABEL):len(line)])
                    if line.startswith(GPU_USAGE_PERCENT_LABEL):
                        totalGpuPercent += int(line[len(GPU_USAGE_PERCENT_LABEL):len(line)])
                        numGpuPercent += 1
                    if line.startswith(GPU_MEM_USAGE_LABEL):
                        totalGpuMemUsage += float(line[len(GPU_MEM_USAGE_LABEL):len(line)])
                        numGpuMemUsage += 1

                    if line.startswith(PROCESS_CPU_PERCENT_LABEL):
                        remLine = line[len(PROCESS_CPU_PERCENT_LABEL):len(line)]

                        processCpuPercent = float(remLine[0:remLine.index(" ---")])
                        cpuPercent = float(remLine[remLine.index("- CPU Percent: ") + 15:len(remLine)])

                        if currentCpuCore == 0:
                            if processCpuPercent > 0:
                                totalProcessCpuPercent += processCpuPercent
                                numProcessCpuPercent += 1

                            totalCpuPercentCore0 += cpuPercent
                            numCpuPercentCore0 += 1
                            currentCpuCore = 1
                        elif currentCpuCore == 1:
                            totalCpuPercentCore1 += cpuPercent
                            numCpuPercentCore1 += 1
                            currentCpuCore = 2
                        elif currentCpuCore == 2:
                            totalCpuPercentCore2 += cpuPercent
                            numCpuPercentCore2 += 1
                            currentCpuCore = 3
                        elif currentCpuCore == 3:
                            totalCpuPercentCore3 += cpuPercent
                            numCpuPercentCore3 += 1
                            currentCpuCore = 0

                    if line.startswith(PROCESS_MEMORY_USAGE_LABEL):
                        totalProcessMemUsage += float(line[len(PROCESS_MEMORY_USAGE_LABEL):len(line)])
                        numProcessMemUsage += 1

            timeElapsedInSeconds = 0
            try:
                timeElapsedInSeconds = (datetime.datetime.strptime(endTime,  '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(startTime,  '%Y-%m-%d %H:%M:%S.%f')).seconds
            except:
                print("An exception occurred")

            totalElapsedTime += timeElapsedInSeconds

        if numLogFiles > 0:
            summary.write('Mean elapsed time to total processed files: '+str(totalElapsedTime / numLogFiles)+' seconds\n')
            summary.write('Mean total processed files: ' + str(totalProcessedImages / numProcessedImages) + '\n')
            summary.write('Total number of runs: ' + str(numLogFiles) + '\n')

            summary.write('Mean seconds to process per image: ' + str(secondsPerImage / numLogFiles) + '\n')
            summary.write('Mean opened threads during process: ' + str(totalThreads / numThreads) + '\n')

            summary.write('Mean GPU Percent Usage during process: ' + str(totalGpuPercent / numGpuPercent) + '\n')
            summary.write('Mean GPU Memory Usage during process: ' + str(totalGpuMemUsage / numGpuMemUsage) + '\n')

            summary.write('Mean CPU Memory Usage during process: ' + str(totalProcessMemUsage / numProcessMemUsage) + '\n')

            summary.write('Mean Process CPU Percent Usage during process: ' + str(totalProcessCpuPercent / numProcessCpuPercent) + '\n')
            summary.write('Mean CPU(0) Percent Usage during process: ' + str(totalCpuPercentCore0 / numCpuPercentCore0) + '\n')
            summary.write('Mean CPU(1) Percent Usage during process: ' + str(totalCpuPercentCore1 / numCpuPercentCore1) + '\n')
            summary.write('Mean CPU(2) Percent Usage during process: ' + str(totalCpuPercentCore2 / numCpuPercentCore2) + '\n')
            summary.write('Mean CPU(3) Percent Usage during process: ' + str(totalCpuPercentCore3 / numCpuPercentCore3) + '\n')

        datasets = glob.glob(method + "/*" + os.path.sep) #only directories

        for dataset in datasets:
            datasetName = os.path.basename(os.path.dirname(dataset))
            print("Processing dataset " + datasetName)
            predictions = glob.glob(dataset + "/*.png")

            totalClassAccuracy = 0
            totalClassPrecision = 0
            totalClassIou = 0
            totalClassRecall = 0
            totalClassF1 = 0
            totalRegionAccuracy = 0
            totalRegionPrecision = 0
            totalRegionIou = 0
            totalRegionRecall = 0
            totalRegionF1 = 0

            cropImage = datasetName == "active_vision" or datasetName == "putkk"
            mapMetrics = {}
            i = 0
            for prediction in predictions:
                imageName = os.path.basename(prediction)
                print("Processing file "+imageName)
                pred = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE if isClassPrediction else cv2.IMREAD_COLOR)
                gt = cv2.imread('results/gt/'+datasetName+'/'+imageName, cv2.IMREAD_GRAYSCALE)

                if cropImage:
                    gt = gt[0:1080, 419:1499]

                h, w = pred.shape[:2]
                gt = cv2.resize(gt, (w, h))
                pred_regions = pred

                if isClassPrediction:
                    pred_regions = region_evaluation.split_classes_to_regions(pred)

                accuracy, prec, rec, f1, iou, metricsPerClass = region_evaluation.evaluate(pred_regions, gt, 'all', False, True)

                insertMetrics(mapMetrics, 'region_class_all', accuracy, prec, rec, f1, iou, metricsPerClass)

                accuracy, prec, rec, f1, iou, metricsPerClass = region_evaluation.evaluate(pred_regions, gt, 'only_best_precision', False, True)

                insertMetrics(mapMetrics, 'region_class_only_best_precision', accuracy, prec, rec, f1, iou, metricsPerClass)

                if isClassPrediction:
                    accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred, gt, 39,
                                                                                                   score_averaging="weighted")
                    insertMetrics(mapMetrics, 'class_prediction', accuracy, prec, rec, f1, iou,
                                  convertClassDic(class_accuracies))

                i += 1
                break

            summary.write('\n\n\n')
            summary.write('=======================================================================\n')
            summary.write('Dataset: ' + datasetName + '\n')
            for metricType in mapMetrics:
                summary.write('----------------------------------------------------------------------\n')
                summary.write('Metric type: ' + metricType + '\n')
                summary.write('\n')
                summary.write(' - Accuracy: ' + str(mapMetrics[metricType]['accuracy']) + '\n')
                summary.write(' - Precision: ' + str(mapMetrics[metricType]['precision']) + '\n')
                summary.write(' - Recall: ' + str(mapMetrics[metricType]['recall']) + '\n')
                summary.write(' - F1: ' + str(mapMetrics[metricType]['f1']) + '\n')
                summary.write(' - Iou: ' + str(mapMetrics[metricType]['iou']) + '\n')
                summary.write('\n')
                summary.write('Classes: \n')
                summary.write('\n')
                classes = mapMetrics[metricType]['classes']
                for classMetric in classes:
                    summary.write(' Class: ' + str(classesList[classMetric]) + '\n')
                    summary.write(' - Accuracy: ' + str(classes[classMetric]['accuracy']) + '\n')
                    summary.write(' - Precision: ' + str(classes[classMetric]['precision']) + '\n')
                    summary.write(' - Recall: ' + str(classes[classMetric]['recall']) + '\n')
                    summary.write(' - F1: ' + str(classes[classMetric]['f1']) + '\n')
                    summary.write(' - Iou: ' + str(classes[classMetric]['iou']) + '\n')
                summary.write('\n')
                summary.write('----------------------------------------------------------------------\n')

            summary.write('=======================================================================\n')
            break