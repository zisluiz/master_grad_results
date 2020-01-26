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

            i = 0
            for prediction in predictions:
                imageName = os.path.basename(prediction)
                print("Processing file "+imageName)
                pred = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE if isClassPrediction else cv2.IMREAD_COLOR)
                gt = cv2.imread('results/gt/'+datasetName+'/'+imageName, cv2.IMREAD_GRAYSCALE)

                if cropImage:
                    gt = gt[0:1080, 419:1499]

                gt = cv2.resize(gt, pred.shape[:2])
                pred_regions = pred;
                if isClassPrediction:
                    pred_regions = region_evaluation.split_classes_to_regions(pred)
                    cv2.imwrite('tests/' + datasetName + '/region_' + imageName, pred_regions)

                accuracy, prec, rec, f1, iou = region_evaluation.evaluate(pred_regions, gt)

                totalRegionAccuracy += accuracy
                totalRegionPrecision += prec
                totalRegionIou += iou
                totalRegionRecall += rec
                totalRegionF1 += f1

                if isClassPrediction:
                    accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred, gt, 39,
                                                                                                   score_averaging="weighted")
                    totalClassAccuracy += accuracy
                    totalClassPrecision += prec
                    totalClassIou += iou
                    totalClassRecall += rec
                    totalClassF1 += f1
                #if preds is None:
                #    preds = np.zeros((pred.shape[0], pred.shape[1], len(predictions)), dtype=int)
                #    gts = np.zeros((pred.shape[0], pred.shape[1], len(predictions)), dtype=int)

                #preds[:, :, i] = pred
                #gts[:, :, i] = gt
                i += 1
                #break
            #break
            summary.write('Dataset ' + datasetName + 'Class Accuracy: ' + str(totalClassAccuracy / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Class Precision: ' + str(totalClassPrecision / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Class Precision: ' + str(totalClassRecall / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Class Precision: ' + str(totalClassF1 / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Class Iou: ' + str(totalClassIou / len(predictions)) + '\n')

            summary.write('Dataset ' + datasetName + 'Region Accuracy: ' + str(totalRegionAccuracy / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Region Precision: ' + str(totalRegionPrecision / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Region Precision: ' + str(totalRegionRecall / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Region Precision: ' + str(totalRegionF1 / len(predictions)) + '\n')
            summary.write('Dataset ' + datasetName + 'Region Iou: ' + str(totalRegionIou / len(predictions)) + '\n')