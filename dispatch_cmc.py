from decision_tree import * 
from random_forest import *

# load cmc data 
# I added attribute name to the first line of the csv file
# wife-age,wife-education,husband-education,n-children,wife-religion,wife-working,husband-occupation,sol-index,media-exposure,class
cmc_data, cmc_attribute = load_data("datasets/cmc.data", csv_delimiter=',')

cmc_attribute_type = ["numerical","categorical","categorical","numerical","categorical","categorical","categorical","categorical","categorical","class"]
cmc_attribute_options = get_possible_options(cmc_data, cmc_attribute, cmc_attribute_type)
cmc_tag_col = -1
# print("classes: " , cmc_attribute_options[cmc_tag_col])
# print(cmc_attribute_options)
# print(cmc_data[:5])

# eval_train, eval_test = dispatch_decision_tree(cmc_data, 
#     cmc_attribute, 
#     cmc_attribute_type, 
#     cmc_attribute_options, 
#     cmc_tag_col, 
#     "entropy", printTree=True, 
#     minimal_size_for_split=2, 
#     minimal_gain=0, 
#     maximal_depth=11
# )
# print(eval_train, eval_test)

report = dispatch_k_fold(cmc_data, 
    cmc_attribute, 
    cmc_attribute_type, 
    cmc_attribute_options, 
    cmc_tag_col, 
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
