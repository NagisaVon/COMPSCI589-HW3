from math import sqrt
from statistics import mean
import random
import numpy as np  

def build_confusion_matrix(true_tags, pred_tags, class_list):
    '''
    return a confusion matrix, row for predicted tags, column for true tags (or class)
    '''
    # init matrix
    confusion_matrix = []
    for i in range(len(class_list)):
        row = [0 for j in range(len(class_list))]
        confusion_matrix.append(row)
    
    for i in range(len(true_tags)):
        confusion_matrix[pred_tags[i]][true_tags[i]] += 1
    return confusion_matrix


def build_report(confusion_matrix, total_entry, class_list, binary_class):
    '''
    return a report of accuracy, precision, recall, f1
    if binary_class is True, then the report use class 1 for positive, and class 0 for negative
    '''
    acc = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))/total_entry
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(class_list)):
        tp = confusion_matrix[i][i]
        all_p = sum(confusion_matrix[i][j] for j in range(len(confusion_matrix)))
        fn = sum(confusion_matrix[j][i] for j in range(len(confusion_matrix)) if j != i)
        precisions.append(tp/all_p)
        recalls.append(tp/(tp+fn))
        f1s.append(2*precisions[i]*recalls[i]/(precisions[i]+recalls[i]))
    if binary_class:
        return [acc, precisions[1], recalls[1], f1s[1]]
    precision = mean(precisions)
    recall = mean(recalls)
    f1 = mean(f1s)
    return [acc, precision, recall, f1]

# true_tags = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
# pred_tags = [1, 0, 0, 0, 2, 1, 1, 0, 2, 2, 2, 2]
# class_list = [0, 1, 2]

true_tags = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
pred_tags = [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]
class_list = [0, 1]
cf_matrix = build_confusion_matrix(true_tags, pred_tags, class_list)
print(cf_matrix)

print(build_report(cf_matrix, len(true_tags), class_list, binary_class=True))


def create_bootstrap_replace(data, n_samples):
    bootstrap_data = random.sample(data, n_samples)
    n_replace = len(data) - n_samples
    replace_data = random.sample(data, n_replace)
    bootstrap_data += replace_data
    return bootstrap_data


da = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dat = np.array(da)
print(create_bootstrap_replace(da, 6))
print(np.array_split(dat, 3))
print(np.concatenate((np.array_split(dat, 3), np.array_split(dat, 3))))

