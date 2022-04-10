from decision_tree import *


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

eval_train, eval_test = dispatch_decision_tree(house_data, 
    house_attribute, 
    house_attribute_type, 
    house_attribute_options, 
    house_tag_col, 
    "entropy", 
    printTree=True, 
    random_state=42, 
    minimal_size_for_split=0., 
    minimal_gain=0., 
    maximal_depth=10000)

# when rnd_state=42, gini and entropy are the same 
# for wine also, that was weird
print(eval_train, eval_test)
