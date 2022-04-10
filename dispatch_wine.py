from decision_tree import * 
from pprint import pprint as pp

# load wine data 
wine_data, wine_attribute = load_data("datasets/hw3_wine.csv", csv_delimiter='\t')

wine_attribute_type = ["class","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical"]
wine_attribute_options = get_possible_options(wine_data, wine_attribute, wine_attribute_type)
wine_tag_col = 0
print(wine_attribute)
print(wine_attribute_options)

# pp(wine_data[:5])

eval_train, eval_test = dispatch(wine_data, \
    wine_attribute, \
    wine_attribute_type, \
    wine_attribute_options, \
    wine_tag_col, \
    "entropy")
print(eval_train, eval_test)
