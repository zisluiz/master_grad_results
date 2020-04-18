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

PROCESS_CPU_PERCENT_MATLAB_LABEL = "%CPU(s): "
PROCESS_MEMORY_USAGE_MATLAB_LABEL = "KB mem : "

classesList = ["__ignore__", "parede", "chao", "armario", "cama", "cadeira", "sofa", "mesa", "porta", "janela",
    "estante_livros", "quadro", "balcao", "persianas", "escrivaninha_carteira", "prateleira", "cortina", "comoda",
    "travesseiro", "espelho", "tapete", "roupa", "teto", "livro", "geladeira", "tv", "papel", "toalha",
    "cortina_chuveiro", "caixa", "quadro_aviso", "pessoa", "mesa_cabeceira_cama", "vaso_sanitario", "pia", "lampada",
   "banheira", "sacola_mochila", "mean"]


def convertClassDic(class_accuracies):
    classes = {}
    for key in class_accuracies.keys():
        classes[key] = {
            'points': 0,
            'accuracy': class_accuracies[key],
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'iou': 0
        }

    return classes


def insertMetrics(metricsmap, metric_type, accuracy, prec, rec, f1, iou, metricsPerClass, totalInstancesGt = 0, totalInstancesPred = 0):
    metricsByType = metricsmap.get(metric_type)
    if not metricsByType:
        metricsmap[metric_type] = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'iou': 0.0,
            'classes': {},
            'totalInstancesGt': 0,
            'totalInstancesPred': 0
        }
    metricsByType = metricsmap.get(metric_type)
    classes = metricsByType['classes']

    for classMetric in metricsPerClass:
        classMetricMap = classes.get(classMetric)

        if not classMetricMap:
            classes[classMetric] = {
                'count': 1,
                'accuracy': metricsPerClass[classMetric]['accuracy'],
                'precision': metricsPerClass[classMetric]['precision'],
                'recall': metricsPerClass[classMetric]['recall'],
                'f1': metricsPerClass[classMetric]['f1'],
                'iou': metricsPerClass[classMetric]['iou']
            }
        else:
            classes[classMetric] = {
                'count': classMetricMap['count'] + 1,
                'accuracy': classMetricMap['accuracy'] + metricsPerClass[classMetric]['accuracy'],
                'precision': classMetricMap['precision'] + metricsPerClass[classMetric]['precision'],
                'recall': classMetricMap['recall'] + metricsPerClass[classMetric]['recall'],
                'f1': classMetricMap['f1'] + metricsPerClass[classMetric]['f1'],
                'iou': classMetricMap['iou'] + metricsPerClass[classMetric]['iou']
            }

    metricsmap[metric_type] = {
        'accuracy': accuracy + metricsByType['accuracy'],
        'precision': prec + metricsByType['precision'],
        'recall': rec + metricsByType['recall'],
        'f1': f1 + metricsByType['f1'],
        'iou': iou + metricsByType['iou'],
        'classes': classes,
        'totalInstancesGt': totalInstancesGt + metricsByType['totalInstancesGt'],
        'totalInstancesPred': totalInstancesPred + metricsByType['totalInstancesPred']
    }

def collect_resources_metrics(summary, log_running_files):
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
    numLogFiles = len(log_running_files)

    for log_run in log_running_files:
        startTime = None
        endTime = None
        currentCpuCore = 0
        currentSecondsPerImage = 0

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
                    currentSecondsPerImage = float(line[len(SECONDS_PER_IMAGE_LABEL):len(line)])
                    secondsPerImage += currentSecondsPerImage
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

                if line.startswith(PROCESS_CPU_PERCENT_MATLAB_LABEL):
                    totalProcessCpuPercent += float(
                        line[len(PROCESS_CPU_PERCENT_MATLAB_LABEL):line.index('us')].strip())
                    numProcessCpuPercent += 1

                if line.startswith(PROCESS_MEMORY_USAGE_MATLAB_LABEL):
                    totalProcessMemUsage += float(
                        line[line.index('livre,') + 6:line.index('usados,')].strip()) / 1000  # to mb
                    numProcessMemUsage += 1

        try:
            timeElapsedInSeconds = (
                        datetime.datetime.strptime(endTime, '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(
                    startTime, '%Y-%m-%d %H:%M:%S.%f')).seconds
        except:
            timeElapsedInSeconds = (
                        datetime.datetime.strptime(endTime, '%d-%b-%Y %H:%M:%S') - datetime.datetime.strptime(
                    startTime, '%d-%b-%Y %H:%M:%S')).seconds

        totalElapsedTime += timeElapsedInSeconds

        if currentSecondsPerImage == 0:
            secondsPerImage += timeElapsedInSeconds / totalProcessedImages

    if numLogFiles > 0:
        summary.write(
            'Mean elapsed time to total processed files: ' + str(totalElapsedTime / numLogFiles) + ' seconds\n')
        summary.write('Mean total processed files: ' + str(totalProcessedImages / numProcessedImages) + '\n')
        summary.write('Total number of runs: ' + str(numLogFiles) + '\n')

        summary.write('Mean seconds to process per image: ' + str(secondsPerImage / numLogFiles) + '\n')
        if totalThreads > 0:
            summary.write('Mean opened threads during process: ' + str(totalThreads / numThreads) + '\n')

        if totalGpuPercent > 0:
            summary.write('Mean GPU Percent Usage during process: ' + str(totalGpuPercent / numGpuPercent) + '\n')
            summary.write('Mean GPU Memory Usage during process: ' + str(totalGpuMemUsage / numGpuMemUsage) + '\n')

        summary.write(
            'Mean CPU Memory Usage during process: ' + str(totalProcessMemUsage / numProcessMemUsage) + '\n')
        summary.write('Mean Process CPU Percent Usage during process: ' + str(
            totalProcessCpuPercent / numProcessCpuPercent) + '\n')

        if totalCpuPercentCore0 > 0:
            summary.write('Mean CPU(0) Percent Usage during process: ' + str(
                totalCpuPercentCore0 / numCpuPercentCore0) + '\n')
        if totalCpuPercentCore1 > 0:
            summary.write('Mean CPU(1) Percent Usage during process: ' + str(
                totalCpuPercentCore1 / numCpuPercentCore1) + '\n')
        if totalCpuPercentCore2 > 0:
            summary.write('Mean CPU(2) Percent Usage during process: ' + str(
                totalCpuPercentCore2 / numCpuPercentCore2) + '\n')
        if totalCpuPercentCore3 > 0:
            summary.write('Mean CPU(3) Percent Usage during process: ' + str(
                totalCpuPercentCore3 / numCpuPercentCore3) + '\n')

methods = glob.glob("results/*", recursive=True)

for method in methods:
    methodName = os.path.basename(method)
    if methodName == "gt":# or methodName != "graph_canny_segm_image":
        continue

    print("Processing method " + methodName)
    isClassPrediction = methodName == "fcn_tensorflow" or methodName == "fusenet_pytorch" or methodName == "rednet"  # prediction with class labeled
    isRedNet = methodName == "rednet"

    log_running_files = glob.glob(method + "/run_cpu_*.txt")
    numLogFiles = len(log_running_files)

    if numLogFiles > 0:
        with open(method + "/summary_cpu.txt", "w+") as summary:
            collect_resources_metrics(summary, log_running_files)

    with open(method+"/summary.txt", "w+") as summary:
        log_running_files = glob.glob(method + "/run_[0-9]*.txt")

        collect_resources_metrics(summary, log_running_files)

        datasets = glob.glob(method + "/*" + os.path.sep) #only directories

        for dataset in datasets:
            datasetName = os.path.basename(os.path.dirname(dataset))

            if "colored" in datasetName:
                continue

            print("Processing dataset " + datasetName)
            predictions = glob.glob(dataset + "/*.png")
            #predictions.extend(glob.glob(dataset + "/*.jpg"))

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

            isAvOrPutkk = datasetName == "active_vision" or datasetName == "putkk"
            isS3DSCT = datasetName == "semantics3d_mod"
            isS3DSST = datasetName == "semantics3d_raw"

            mapMetrics = {}
            totalImages = 0
            for prediction in predictions:
                imageName = os.path.basename(prediction)
                print("Processing file "+imageName)
                pred = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE if isClassPrediction else cv2.IMREAD_COLOR)
                gt = cv2.imread('results/gt/'+datasetName+'/'+imageName, cv2.IMREAD_GRAYSCALE)

                if not isClassPrediction:
                    if isAvOrPutkk:
                        gt = gt[0:1080, 240:1680]
                elif isRedNet:
                    if isAvOrPutkk:
                        gt = gt[0:1080, 240:1680]
                    elif isS3DSCT:
                        gt = gt[270:1080, 0:1080]
                    elif isS3DSST:
                        gt = gt[64:1024, 0:1280]
                else:
                    if isAvOrPutkk:
                        gt = gt[0:1080, 420:1500]
                    elif isS3DSST:
                        gt = gt[0:1024, 256:1280]

                h, w = pred.shape[:2]
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_regions = pred
                #pred_classed = pred

                if isClassPrediction:
                    pred_regions = region_evaluation.split_classes_to_regions(pred)

                gt_regions = region_evaluation.split_classes_to_regions(gt)

                #pred_classed = region_evaluation.join_regions_into_classes(pred_regions, gt)

                accuracy, prec, rec, f1, iou, metricsPerClass, totalInstancesGt, totalInstancesPred = region_evaluation.evaluate(
                    pred_regions, gt, gt_regions, 'all', True, methodName == "graph_canny_segm_objects")

                insertMetrics(mapMetrics, 'region_class_all', accuracy, prec, rec, f1, iou, metricsPerClass, totalInstancesGt, totalInstancesPred)

                accuracy, prec, rec, f1, iou, metricsPerClass, totalInstancesGt, totalInstancesPred = region_evaluation.evaluate(
                    pred_regions, gt, gt_regions, 'only_best_iou', True, methodName == "graph_canny_segm_objects")

                insertMetrics(mapMetrics, 'region_class_best_iou', accuracy, prec, rec, f1, iou, metricsPerClass, totalInstancesGt, totalInstancesPred)

                #unique_pixels = set(np.unique(pred_classed))
                #unique_pixels.update(np.unique(gt))

                #accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred_classed, gt,
                #                                                                               len(unique_pixels),
                #                                                                               labels=list(
                #                                                                                   unique_pixels),
                #                                                                               score_averaging="macro")

                #accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred_classed, gt, 39,
                #                                                                               score_averaging="weighted")
                #insertMetrics(mapMetrics, 'region_classed_prediction', accuracy, prec, rec, f1, iou,
                #              convertClassDic(class_accuracies))

                if isClassPrediction:
                    unique_pixels = set(np.unique(pred))
                    unique_pixels.update(np.unique(gt))

                    accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred, gt,
                                                                                                   len(unique_pixels),
                                                                                                   labels=list(
                                                                                                       unique_pixels),
                                                                                                   score_averaging="macro")

                    #accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred, gt, 39,
                    #                                                                               score_averaging="weighted")
                    insertMetrics(mapMetrics, 'class_prediction', accuracy, prec, rec, f1, iou,
                                  convertClassDic(class_accuracies))

                totalImages += 1
                #break

            summary.write('\n\n\n')
            summary.write('=======================================================================\n')
            summary.write('Dataset: ' + datasetName + '\n')
            for metricType in mapMetrics:
                summary.write('----------------------------------------------------------------------\n')
                summary.write('Metric type: ' + metricType + '\n')
                summary.write('\n')
                summary.write(' - Accuracy: ' + str((mapMetrics[metricType]['accuracy'] / totalImages * 100)) + '\n')
                summary.write(' - Precision: ' + str((mapMetrics[metricType]['precision'] / totalImages * 100)) + '\n')
                summary.write(' - Recall: ' + str((mapMetrics[metricType]['recall'] / totalImages * 100)) + '\n')
                summary.write(' - F1: ' + str((mapMetrics[metricType]['f1'] / totalImages * 100)) + '\n')
                summary.write(' - Iou: ' + str((mapMetrics[metricType]['iou'] / totalImages * 100)) + '\n')
                summary.write(' - Total instances gt: ' + str((mapMetrics[metricType]['totalInstancesGt'] / totalImages)) + '\n')
                summary.write(' - Total instances pred: ' + str((mapMetrics[metricType]['totalInstancesPred'] / totalImages)) + '\n')
                summary.write('\n')
                summary.write('Classes: \n')
                summary.write('\n')
                classes = mapMetrics[metricType]['classes']
                for classMetric in classes:
                    summary.write(' Class: ' + str(classesList[classMetric]) + '\n')
                    summary.write(' - Accuracy: ' + str((classes[classMetric]['accuracy'] / classes[classMetric]['count'] * 100)) + '\n')
                    summary.write(' - Precision: ' + str((classes[classMetric]['precision'] / classes[classMetric]['count'] * 100)) + '\n')
                    summary.write(' - Recall: ' + str((classes[classMetric]['recall'] / classes[classMetric]['count'] * 100)) + '\n')
                    summary.write(' - F1: ' + str((classes[classMetric]['f1'] / classes[classMetric]['count'] * 100)) + '\n')
                    summary.write(' - Iou: ' + str((classes[classMetric]['iou'] / classes[classMetric]['count'] * 100)) + '\n')
                summary.write('\n')
                summary.write('----------------------------------------------------------------------\n')

            summary.write('=======================================================================\n')
            #break