from math import sqrt
from statistics import mean
from decision_tree import * 


def create_bootstrap_replace(data, n_samples):
    # random.sample does not works for np.array
    data = data.tolist()
    bootstrap_data = random.sample(data, n_samples)
    n_replace = len(data) - n_samples
    replace_data = random.sample(data, n_replace)
    return np.array(bootstrap_data + replace_data)


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
        pred_ind = class_list.index(int(pred_tags[i]))
        true_ind = class_list.index(int(true_tags[i]))
        confusion_matrix[pred_ind][true_ind] += 1
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


def evaluate_random_forest(test, forest, tag_col, class_list, binary_class):
    true_tags = test.T[tag_col]
    pred_tags = []
    for row in test:
        preds = []
        for tree in forest:
            preds.append(predict(row, tree))
            # vote for the class
        pred_tags.append(Counter(preds).most_common(1)[0][0])
    confusion_matrix = build_confusion_matrix(true_tags, pred_tags, class_list)
    report = build_report(confusion_matrix, len(true_tags), class_list, binary_class)
    return report
    

def dispatch_random_forest(train, test, attr, attr_type, attr_opt, tag_col, minimal_size_for_split, minimal_gain, maximal_depth, algo, n_trees, binary_class, bootstrap_percentage):
    forest = []
    # a list of index, without the index for the class column
    attr_list = [i for i in range(len(attr)) if attr_type[i] != "class"]
    only_m_attr = int(sqrt(len(attr_list)))
    for _ in range(n_trees): # _ since we don't use the loop value
        train_bootstrap = create_bootstrap_replace(train, int(len(train)*bootstrap_percentage))
        tree = build_decision_tree(train_bootstrap, attr_list, attr_type, attr_opt, tag_col, algo, minimal_size_for_split, minimal_gain, maximal_depth, only_m_attr)
        forest.append(tree)
    
    eval_result = evaluate_random_forest(test, forest, tag_col, attr_opt[tag_col], binary_class)
    return eval_result


def dispatch_k_fold(data, attr, attr_type, attr_opt, tag_col, minimal_size_for_split=0, minimal_gain=0, maximal_depth=10000, algo="entropy", random_state=42, k_fold=10, n_trees=10, binary_class=False, bootstrap_percentage=0.9):
    # k_fold_size = int(len(data) / k_fold)
    data_by_class = [] 
    for i in range(len(attr_opt[tag_col])):
        tag = attr_opt[tag_col][i]
        data_by_class.append(data[data.T[tag_col] == tag])
        data_by_class[i] = np.array_split(data_by_class[i], k_fold)
    
    # [[accuracy, precision, recall, f1], ...]
    k_fold_eval = []
    for i in range(k_fold):
        train = np.array([]).reshape(0, len(data.T))
        test = np.array([]).reshape(0, len(data.T))
        # build train and test from different classes
        for j in range(len(attr_opt[tag_col])):
            # test set is the the i-th fold
            test = np.vstack([test, data_by_class[j][i]])
            # train set is all data except the i-th fold
            # this is so stupid, but it works
            before_i = np.array([]).reshape(0, len(data.T))
            after_i =  np.array([]).reshape(0, len(data.T))
            for k in range(i):
                before_i = np.vstack([before_i, data_by_class[j][k]])
            for k in range(i+1, k_fold):
                after_i = np.vstack([after_i, data_by_class[j][k]])
            if (len(before_i) != 0):
                train = np.vstack([train ,before_i])
            if (len(after_i) != 0):
                train = np.vstack([train ,after_i])
        # print(len(train)+len(test)==len(data))
        eval_result = dispatch_random_forest(train, test, attr, attr_type, attr_opt, tag_col, minimal_size_for_split, minimal_gain, maximal_depth, algo, n_trees, binary_class, bootstrap_percentage)
        k_fold_eval.append(eval_result)
    
    k_fold_eval =np.mean(k_fold_eval, axis=0)
    return k_fold_eval # np array .T then .mean
