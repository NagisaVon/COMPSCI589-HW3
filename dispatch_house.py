from decision_tree import *
from math import sqrt

from random_forest import dispatch_k_fold

# load house data
house_data, house_attribute = load_data("datasets/hw3_house_votes_84.csv")
# house_possible_classes = [0, 1, 2]
house_attribute_type = ["categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","categorical","class"]
house_attribute_options = get_possible_options(house_data, house_attribute, house_attribute_type)
house_tag_col = -1
print("classes: " , house_attribute_options[house_tag_col])
# temporary test data
temp_data = house_data[:5, -5:]
print(temp_data)

# eval_train, eval_test = dispatch_decision_tree(house_data, 
#     house_attribute, 
#     house_attribute_type, 
#     house_attribute_options, 
#     house_tag_col, 
#     "entropy", 
#     printTree=True, 
#     random_state=42, 
#     minimal_size_for_split=0., 
#     minimal_gain=0., 
#     maximal_depth=10000,
#     only_m_attr=int(sqrt(len(house_attribute)-1)) # very important to convert to int
# )

# when rnd_state=42, gini and entropy are the same 
# for wine also, that was weird
# print(eval_train, eval_test)

report = dispatch_k_fold(house_data, 
    house_attribute, 
    house_attribute_type, 
    house_attribute_options, 
    house_tag_col, 
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