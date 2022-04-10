import csv
import json
from pprint import pprint
import numpy as np
import math
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

def get_possible_options(data, attr, attr_type):
    options = []
    for i in range(len(attr)):
        opt = []
        if attr_type[i] == "numerical":
            opt = {}
        else: # if the attribute is categorical or a class
            opt = set(data[i])
        options.append(opt)
    return options

def entropy(data):
    '''
    list of classes -> entropy
    '''
    cnt = list(Counter(data).values())
    total = len(data)
    return sum([-k/total * math.log(k/total, 2) for k in cnt])


def gini(data):
    cnt = list(Counter(data).values())
    total = len(data)
    return 1 - sum([k/total * (k/total) for k in cnt])


def entropy_in_list(data, attr_list, attr_type, attr_opt, tag_col, algo):
    '''
    house_data(w/ tags)->entropy_of_all_attributes[]
    '''
    attr_count = len(data.T)
    total_entry = len(data)
    ent_list = []
    # for each attributes, calculate information gain
    for i in range(attr_count):
        # to use argmin later, keep all attribute in the list
        # skip if not in attr_list
        if(not i in attr_list or attr_type[i] == 'class'):
            ent_list.append(1000)  # infinite high entropy
            continue
        # gather the class labels of the current attribute for each options
        ent = 0
        for j in attr_opt[i]:
            option_tags = [row[tag_col] for row in data if row[i] == j]
            ent += len(option_tags)/total_entry * algo(option_tags)
        ent_list.append(ent)
    return ent_list


class Node:
    # tag means different classes, since class is a reserved keyword
    def __init__(self, depth, isLeaf=False, attr_index=None, tag=None ):
        self.depth = depth
        self.attr_index = attr_index
        self.isLeaf = isLeaf
        self.tag = tag
        self.children = []
        self.option = None
        self.parent = None

    def print_tree(self, attribute_list, indent=0):
        ind_str = ' ' * indent
        if self.isLeaf:
            print( ind_str + "Leaf:CLASS" , self.tag, "OPTION", self.option , "DEPTH" , self.depth)
        else:
            print( ind_str + attribute_list[self.attr_index] , "OPTION" , self.option , "DEPTH" , self.depth)
            for child in self.children:
                child.print_tree(attribute_list, indent+2)


def build_decision_tree(data, attr_list, attr_type, attr_opt, tag_col, algo, _depth=0):
    # If there are no more attributes that can be tested, return the most common tag
    if (not attr_list): # equivelent to len(attr_list) == 0
        return Node( _depth, isLeaf=True, tag=Counter(data.T[-1]).most_common()[0][0])

    # If there are no more data
    if (not data.any()):
        return Node( _depth, isLeaf=True, tag=None)

    tags = data.T[-1]
    original_entropy = entropy(tags)

    # If there is only one tag, return the tag
    if (original_entropy == 0):
        return Node( _depth, isLeaf=True, tag=Counter(tags).most_common()[0][0])
    # this subtraction is not necessary, just to be justify the name 'info_gain'
    if algo=='gini':
        gini_val = entropy_in_list(data, attr_list, attr_type, attr_opt, tag_col, gini)
        decided_attr = np.argmin(gini_val)
    elif algo=='entropy':
        info_gain = [original_entropy - ent for ent in entropy_in_list(data, attr_list, attr_type, attr_opt, tag_col, entropy)]
        decided_attr = np.argmax(info_gain)

    # get the index of the attribute with highest information gain
    nd = Node( _depth, isLeaf=False, attr_index=decided_attr ) 
    # keep track of a majority tag in case the leaf is None
    # .most_common()[0][0] returns the most common tag (key of a dict)
    nd.tag = Counter(data.T[tag_col]).most_common()[0][0] 
    # remove decision attribute from attr_list
    new_attr_list = [x for x in attr_list if x != decided_attr]

    # build subtree for each possible option 
    for opt in attr_opt[decided_attr]:
        # filtered data
        new_data = data[data.T[decided_attr] == opt]
        subtree = build_decision_tree(new_data, new_attr_list, attr_type, attr_opt, tag_col, algo, _depth+1) 
        subtree.option = opt
        nd.children.append(subtree)
        subtree.parent = nd
    return nd


# depreciated function
def func_generate_template_json(attribute_list):
    """
    generate a json names file
    """
    attr_info = []
    for i in range(len(attribute_list)):
        attr_info.append({'attr_index': i, \
            'attr_name': attribute_list[i], \
            'attr_type': 'categorical', \
            'attr_possible_options': [0, 1, 2]})
    with open('temp.json', 'w') as f:
        json.dump(attr_info, f, indent=2)


def load_data(cvsfilename, csv_delimiter=','): 
    # import data, include encoding to ommit BOM  
    data = []
    with open(cvsfilename, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=csv_delimiter)
        for row in reader:
            if len(row) != 0: # skip empty lines
                data.append(row)
    # drop the attribute row from the list
    attributes = data.pop(0)
    data = np.array(data).astype(float)
    return (data, attributes)


# depreciated function
def load_meta(jsonfilename):
    meta = {};
    with open(jsonfilename) as json_file:
        meta['attribute_info'] = json.load(json_file)
    return meta


def predict(row, node):
    for child in node.children:
        if row[node.attr_index] == child.option:
            if child.isLeaf:
                if child.tag == None:
                    return node.tag
                else:
                    return child.tag
            else:
                return predict(row, child)
    return None;


def evaluate(data, tree):
    correct = 0
    for i in range(len(data)):
        if data[i][-1] == predict(data[i], tree):
            correct += 1
    return correct / len(data)


def dispatch(data, attr, attr_type, attr_opt, tag_col, algo, random_state=42, printTree=False):
    train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=random_state)
    # build a attr_list not including the class attribute
    attr_list = [i for i in range(len(attr)) if attr_type[i] != "class"]
    tree = build_decision_tree(train, attr_list, attr_type, attr_opt, tag_col, algo)
    if(printTree):
        tree.print_tree(attr)
    return evaluate(train, tree), evaluate(test, tree)


