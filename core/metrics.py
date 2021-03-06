import numpy as np
#from keras import backend as K
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from core import eval_semantic_segmentation
from math import isnan

"""
def iou_coef(y_true, y_pred, smooth=1):
    axis = 0
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
    union = K.sum(y_true, axis)+K.sum(y_pred, axis)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
"""
# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, labels=None, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    #global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    #class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)
    global_accuracy = 0
    class_accuracies = 0

    eval_semantic_results = eval_semantic_segmentation.eval_semantic_segmentation(
        np.reshape(pred, (1, pred.shape[0], pred.shape[1])),
        np.reshape(label, (1, label.shape[0], label.shape[1])))

    prec = precision_score(flat_label, flat_pred, labels=labels, average=score_averaging)
    rec = recall_score(flat_label, flat_pred, labels=labels, average=score_averaging)
    f1 = f1_score(flat_label, flat_pred, labels=labels, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    class_results = dict(enumerate(eval_semantic_results["class_accuracy"]))
    #remove nan
    class_results = {k: class_results[k] for k in class_results if not isnan(class_results[k])}

    return eval_semantic_results["pixel_accuracy"], class_results, prec, rec, f1, iou