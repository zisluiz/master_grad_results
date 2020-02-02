import os
import cv2
import numpy as np
from core import metrics


#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.neighbors import KNeighborsClassifier
##from keras import backend as K
#from sklearn.metrics import precision_score, \
#    recall_score, confusion_matrix, classification_report, \
#    accuracy_score, f1_score




def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images
    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes)

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = cv2.imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis=-1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels == 0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights


pred = cv2.imread("results/rednet/active_vision/000210000010101.png", cv2.IMREAD_GRAYSCALE)
gt = cv2.imread("results/gt/active_vision/000210000010101.png", cv2.IMREAD_GRAYSCALE)

gt = gt[0:1080, 419:1499]
gt = cv2.resize(gt, (pred.shape[1],pred.shape[0]), interpolation=cv2.INTER_NEAREST)

cv2.imwrite("gt.png", gt)

preds = np.zeros((pred.shape[0], pred.shape[1], 1), dtype=int)
gts = np.zeros((pred.shape[0], pred.shape[1], 1), dtype=int)

preds[:, :, 0] = pred
gts[:, :, 0] = gt

#metrics = eval_semantic_segmentation.eval_semantic_segmentation(preds, gts)

intersection = np.logical_and(pred, gt)
union = np.logical_or(pred, gt)
iou_score = np.sum(intersection) / np.sum(union)

accuracy, class_accuracies, prec, rec, f1, iou = metrics.evaluate_segmentation(pred, gt, 38, score_averaging="weighted")

print(str(metrics)+'\n')
print('accuracy: ' + str(accuracy)+'\n')
print('class_accuracies: ' + str(class_accuracies)+'\n')
print('prec: ' + str(prec)+'\n')
print('iou_score2: ' + str(iou)+'\n')
#print('iou_score2: ' + str(iou_coef(gt.astype(np.int32), pred.astype(np.int32)))+'\n')




#labels = range(0, 39)
#conf_matrix = confusion_matrix(gt, pred, labels)
#accuracy = accuracy_score(gt, pred, sample_weight=labels)
#totalAccuracy += accuracy
#precision = precision_score(gt, pred, average='micro', labels = labels)
#totalPrecision += precision

#knn = KNeighborsClassifier(n_neighbors=38)
#classifier = MultiOutputClassifier(knn, n_jobs=-1)
#classifier.fit(pred, gt)
#predictions = classifier.predict(pred)
#score = classifier.score(pred, np.array(gt))

