from decision_tree import * 
from random_forest import *

# load wine data 
wine_data, wine_attribute = load_data("datasets/hw3_wine.csv", csv_delimiter='\t')

wine_attribute_type = ["class","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical"]
wine_attribute_options = get_possible_options(wine_data, wine_attribute, wine_attribute_type)
wine_tag_col = 0
print("classes: " , wine_attribute_options[wine_tag_col])
# print(wine_attribute_options)
# pp(wine_data[:5])

# eval_train, eval_test = dispatch_decision_tree(wine_data, \
#     wine_attribute, \
#     wine_attribute_type, \
#     wine_attribute_options, \
#     wine_tag_col, \
#     "entropy", printTree=True, random_state=100)
# print(eval_train, eval_test)

report = dispatch_k_fold(wine_data, 
    wine_attribute, 
    wine_attribute_type, 
    wine_attribute_options, 
    wine_tag_col, 
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