import csv
from math import log

def main():
    data = getData("weather.csv")
    labels = data[0]
    tree = decision_tree(data[1:], labels)
    print("Decision Tree:", tree)

def getData(file):
    with open(file) as f:
        return [row for row in csv.reader(f)]

def decision_tree(data, labels):
    results = [row[-1] for row in data]
    if results.count(results[0]) == len(results):
        return results[0]
    
    max_gain_attribute = select_attribute(data)
    tree = {labels[max_gain_attribute]: {}}
    nodes = set(row[max_gain_attribute] for row in data)

    for node in nodes:
        sublabels = labels[:max_gain_attribute] + labels[max_gain_attribute+1:]
        subtree = decision_tree(split_data(data, max_gain_attribute, node), sublabels)
        tree[labels[max_gain_attribute]][node] = subtree

    return tree

def split_data(data, attribute, value):
    return [row[:attribute] + row[attribute+1:] for row in data if row[attribute] == value]

def select_attribute(data):
    base_entropy = entropy(data)
    attributes = len(data[0]) - 1
    best_attribute, max_info_gain = -1, -1

    for attribute in range(attributes):
        values = set(row[attribute] for row in data)
        attr_entropy = sum((len(split_data(data, attribute, v)) / len(data)) * entropy(split_data(data, attribute, v)) for v in values)
        info_gain = base_entropy - attr_entropy

        if info_gain > max_info_gain:
            max_info_gain, best_attribute = info_gain, attribute

    return best_attribute

def entropy(data):
    total_rows = len(data)
    outcomes = [row[-1] for row in data]
    probs = [outcomes.count(outcome) / total_rows for outcome in set(outcomes)]
    return -sum(p * log(p, 2) for p in probs)

main()
