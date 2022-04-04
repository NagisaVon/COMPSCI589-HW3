import csv
import json
from pprint import pprint
import numpy as np
import math
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt


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


def entropy_in_list(data, attr_list, algo):
    '''
    house_data(w/ tags)->entropy_of_all_attributes[]
    '''
    attr_count = len(data.T) - 1
    total_entry = len(data)
    ent_list = []
    # for each attributes, calculate information gain
    for i in range(attr_count):
        # skip if not in attr_list
        if(not i in attr_list):
            ent_list.append(1000)  # infinate high entropy
            continue
        # three possible classes are hard coded
        # organize data by class
        v_tag = {0: [], 1: [], 2: []}
        for j in range(len(data)):
            v_tag[data[j][i]].append(data[j][-1])
        # sum of entropy of each class
        ent = sum(len(v_tag[val])/total_entry * algo(v_tag[val])
                  for val in v_tag)
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
            print( ind_str + "Node:SPLIT_ATTI" , attribute_list[self.attr_index] , "OPTION" , self.option , "DEPTH" , self.depth)
            for child in self.children:
                child.print_tree(attribute_list, indent+2)


def build_decision_tree(data, attr_list, meta, algo,_depth=0):

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
        gini_val = entropy_in_list(data, attr_list, gini)
        decision_attr = np.argmin(gini_val)
    elif algo=='entropy':
        info_gain = [original_entropy - ent for ent in entropy_in_list(data, attr_list, entropy)]
        decision_attr = np.argmax(info_gain)

    # get the index of the attribute with highest information gain
    nd = Node( _depth, isLeaf=False, attr_index=decision_attr ) 
    # keep track of a majority tag in case the leaf is None
    nd.tag = Counter(data.T[-1]).most_common()[0][0]
    # remove decision attribute from attr_list
    new_attr_list = [x for x in attr_list if x != decision_attr]

    meta_decided_attribute = meta['attribute_info'][decision_attr]
    if (meta_decided_attribute['attr_type'] == 'categorical'):
        nd.possible_options = meta_decided_attribute['attr_possible_options']
    elif (meta_decided_attribute['attr_type'] == 'numerical'):
        return;  
    print(nd.possible_options)
    # build subtree for each possible option 
    for v in nd.possible_options:
        # filtered data
        new_data = data[data.T[decision_attr] == v]
        subtree = build_decision_tree(new_data, new_attr_list, meta, algo, _depth+1) 
        subtree.option = v
        nd.children.append(subtree)
        subtree.parent = nd

    return nd


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


def load_data(cvsfilename): 
    meta = {'classatcolumn': -1, 'possible_classes':[], 'attributes':[], 'attribute_info':[]}
    # import data, include encoding to ommit BOM  
    data = []
    with open(cvsfilename, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 0: # skip empty lines
                data.append(row)
    # drop the attribute row from the list
    attributes = data.pop(0)
    data = np.array(data).astype(int)
    return (data, attributes)


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


def dispatch(data, meta, algo, random_state=42):
    train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=random_state)
    tree = build_decision_tree(train, list(range(len(meta['attributes'])-1)), meta, algo)
    tree.print_tree(meta['attributes'])
    return evaluate(train, tree), evaluate(test, tree)


