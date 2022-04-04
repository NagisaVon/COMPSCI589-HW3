from decision_tree import *


# load house data
house_data, house_attribute = load_data("datasets/hw3_house_votes_84.csv")
house_meta = load_meta("datasets/hw3_house_votes_84.json")
house_meta['possible_classes'] = [0, 1, 2]
house_meta['attributes'] = house_attribute
house_meta['classatcolumn'] = -1

# temporary test data
temp_data = house_data[:5, -5:]
print(temp_data)

eval_train, eval_test = dispatch(temp_data, house_meta, "entropy")
print(eval_train, eval_test)
