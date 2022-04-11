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

# eval_train, eval_test = dispatch_decision_tree(cancer_data, \
#     cancer_attribute, \
#     cancer_attribute_type, \
#     cancer_attribute_options, \
#     cancer_tag_col, \
#     "gini", printTree=True)
# print(eval_train, eval_test)

report = dispatch_k_fold(cancer_data, 
    cancer_attribute, 
    cancer_attribute_type, 
    cancer_attribute_options, 
    cancer_tag_col, 
    minimal_size_for_split=0.,
    minimal_gain=0.,
    maximal_depth=10000,
    algo="entropy",  
    random_state=42, 
    k_fold=10,
    n_trees=10,
    binary_class=False, 
    bootstrap_percentage=0.9
)

print(report)