from decision_tree import * 


# load wine data 
wine_data, wine_attribute = load_data("datasets/hw3_wine.csv")
func_generate_template_json(wine_attribute)
wine_meta = load_meta("datasets/hw3_house_votes_84.json")
wine_meta['possible_classes'] = [1, 2, 3]
wine_meta['attributes'] = wine_attribute
wine_meta['classatcolumn'] = 1