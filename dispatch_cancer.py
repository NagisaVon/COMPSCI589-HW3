from decision_tree import * 
from random_forest import *

# load cancer data 
cancer_data, cancer_attribute = load_data("datasets/hw3_cancer.csv", csv_delimiter='\t')

cancer_attribute_type = ["numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","class"]
cancer_attribute_options = get_possible_options(cancer_data, cancer_attribute, cancer_attribute_type)
cancer_tag_col = -1
print("classes: " , cancer_attribute_options[cancer_tag_col])
# print(cancer_attribute_options)
# print(cancer_data[:5])

eval_train, eval_test = dispatch_decision_tree(cancer_data, \
    cancer_attribute, \
    cancer_attribute_type, \
    cancer_attribute_options, \
    cancer_tag_col, \
    "gini", printTree=True)
print(eval_train, eval_test)