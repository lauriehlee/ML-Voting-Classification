import sys
import csv
import copy
import numpy as np 
from anytree import Node
import math
import random
import matplotlib.pyplot as plt
import randomForest


def decision_tree(node, lst_of_classifications, attributes_list, original_attributes_list, is_categorical, idx_of_class, attribute_averages, minimal_size_for_split, minimal_gain, maximal_depth, curr_depth):
    # Train a decision tree based by recursively:
    # (1) Picking a random subset of m = sqrt(X) attributes
    # (2) Out of these, select the best attribute to split the current node (e.g., based on Information Gain)
    # (3) Add the node to the tree and use it to partition the data into disjoint subsets

    # If all instances in data set belong to same class, define node N as leaf node, label w class, and return it
    # IMPORTANT: leaf nodes will be represented by a tuple in decision tree to show classification: (class label, accompanying data, attribute value, index of most recent attribute that led to this node)
    # Edge case: If the root node is the leaf node, it will be represented by (class label, accompanying data)
    # Otherwise, non-leaf nodes in the tree's name will be identified by tuple (index of most recent attribute that led to this node, accompanying data, attribute value)

    homogenous = check_same_class(node.name[1], idx_of_class)
    if homogenous:
        # Edge case: If this is the root node of the entire decision tree, return the node itself classified as the homogenous class.
        if len(node.name) == 2:
            node.name = (node.name[1][0][idx_of_class], node.name[1])
            lst_of_classifications.append((node.name[1][0][idx_of_class], node.name[1]))
            return node
        else:
            node.name = (node.name[1][0][idx_of_class], node.name[1], node.name[2], node.name[0])
            lst_of_classifications.append(node.name[:2])
            return node
    
    # Stop splitting whenever the current branch of the tree becomes too deep—i.e., its depth becomes larger than maximal_depth
    # OR
    # Stop splitting nodes whenever the data partition associated with the current node contains less than minimal_size_for_split instances 
    # Then: define node N as leaf node labeled w majority class in data set and return it.
    if (curr_depth > maximal_depth) or (len(node.name[1]) < minimal_size_for_split):
        majority_class = find_majority_class(node.name[1], idx_of_class)

        # Edge case: If this is the root node of the entire decision tree, return the node itself classified as the homogenous class.
        if len(node.name) == 2:
            node.name = (majority_class, node.name[1])
            lst_of_classifications.append((majority_class, node.name[1]))
            return node
        else: 
            node.name = (majority_class, node.name[1], node.name[2], node.name[0])
            lst_of_classifications.append(node.name[:2])
            return node
    
    # Otherwise, select an attribute to add to the tree using the Information Gain criterion
    # (1) Pick a random subset of m = sqrt(X) attributes
    # (2) Out of these, select the best attribute to split the current node (e.g., based on Information Gain)
    random_subset_of_attributes = random.sample(original_attributes_list, math.floor(math.sqrt(len(original_attributes_list))))
    attributes_list = copy.deepcopy(random_subset_of_attributes)
    chosen_attribute = choose_next_attribute(node.name[1], attributes_list, original_attributes_list, is_categorical, idx_of_class, attribute_averages, minimal_gain)
    
    # Stop splitting nodes whenever the information gain resulting from a split is not sufficiently high
    if chosen_attribute == -1:
        majority_class = find_majority_class(node.name[1], idx_of_class)

        # Edge case: If this is the root node of the entire decision tree, return the node itself classified as majority class.
        if len(node.name) == 2:
            node.name = (majority_class, node.name[1])
            lst_of_classifications.append((majority_class, node.name[1]))
            return node
        else: 
            node.name = (majority_class, node.name[1], node.name[2], node.name[0])
            lst_of_classifications.append(node.name[:2])
            return node
    
    chosen_attribute_index = original_attributes_list.index(chosen_attribute) # This is so that you can always find corresponding index in data set even if attribute list is continually modified.

    # Add to node N, one branch for each possible value of the selected attribute with non-empty data set
    # Partition the instances / examples -- assign each instance to its corresponding branch, based on the value of that instance's attribute
    partitioned_instances = partition_based_on_attribute(node.name[1], chosen_attribute_index, is_categorical, attribute_averages)
    path_attributes_list = copy.deepcopy(attributes_list) # Modification after HW1: make a deep copy of attributes list so that you can use the same attribute in diff paths that stem from a common ancestor
    # If it results in an empty partition based on the attribute, define node N as leaf node labeled w majority class in data set and return it
    majority_class = find_majority_class(node.name[1], idx_of_class)

    # attribute_val for data corresponds to partition indexes (e.g., partition 0, partition 1, partition 2, etc.)
    if is_categorical:
        attribute_val = 0 
    else:
        attribute_val = ("leq", attribute_averages[chosen_attribute_index]) # Numerical attribute value described by a tuple where: (leq or gr, threshold)
    
    for partition in partitioned_instances:
        if len(partition) > 0:
            child = Node((chosen_attribute_index, partition, attribute_val), parent=node)
            child = decision_tree(child, lst_of_classifications, path_attributes_list, original_attributes_list, is_categorical, idx_of_class, attribute_averages, minimal_size_for_split, minimal_gain, maximal_depth, curr_depth + 1)
        
        if len(partition) == 0:
            "len partition 0"
            child = Node((majority_class, [], attribute_val, chosen_attribute_index), parent=node)
            lst_of_classifications.append(child.name[:2])
        

        if is_categorical:
            attribute_val += 1
        else:
            attribute_val = ("gr", attribute_averages[chosen_attribute_index]) 

    return node


def check_same_class(data_set, idx_of_class):
    classification = data_set[0][idx_of_class]
    for elem in data_set:
        if (float)(elem[idx_of_class]) != (float)(classification):
            return False
    
    return True

def find_majority_class(data_set, idx_of_class):
    class_count = randomForest.obtain_class_count(data_set, idx_of_class)
    majority_count = class_count[0][1]
    majority_class = class_count[0][0]
    for clss, count in class_count:
        if count == majority_count:
            # Randomly choose majority class if counts are equal.
            if random.randint(0, 1) == 0:
                majority_class = clss
        if count > majority_count:
            majority_count = count
            majority_class = clss
    
    return majority_class

def choose_next_attribute(data_set, attributes_list, original_attributes_list, is_categorical, idx_of_class, attribute_averages, minimal_gain):
    ig_of_attributes = []
    for attribute in attributes_list:
        # Calculate the Information Gain of each attribute
        ig_of_attributes.append((attribute, information_gain(data_set, original_attributes_list.index(attribute), is_categorical, idx_of_class, attribute_averages)))

    # Find the tuple that has the maximum IG value among all tuples
    informative_attribute = max(ig_of_attributes, key=lambda x: (x[1], x[0]))  

    # Return -1 if information gain resulting from a split is not sufficiently high
    if float(informative_attribute[1]) < minimal_gain:
        return -1
    
    return informative_attribute[0]

def information_gain(data_set, index_of_attribute, is_categorical, idx_of_class, attribute_averages):

    total_instances = (float)(len(data_set))
    # Find entropy of original data set.
    original_entropy = find_entropy(data_set, idx_of_class)
    
    # Partition instances based on attribute
    partitioned_instances = partition_based_on_attribute(data_set, index_of_attribute, is_categorical, attribute_averages)

    # Find entropy of each attribute
    # Find weighted average entropy according to resulting partitions
    weighted_average_entropy = 0
    for partition in partitioned_instances:
        entropy = find_entropy(partition, idx_of_class)
        weighted_average_entropy += ((float)(len(partition))/total_instances * entropy)
    
    return original_entropy - weighted_average_entropy

def find_entropy(data_set, idx_of_class):
    class_counts = randomForest.obtain_class_count(data_set, idx_of_class)
    total_instances = (float)(len(data_set))

    entropy = 0
    if total_instances > 0:
        for clss, count in class_counts:
            prob_of_class = 0
            log_of_class = 0
            prob_of_class = (float)(count) / total_instances
            if prob_of_class > 0:
                log_of_class = math.log2(prob_of_class)
            
            entropy += ((prob_of_class * -1) * log_of_class)

    return entropy

def partition_based_on_attribute(data_set, chosen_attribute_index, is_categorical, attribute_averages):
    if is_categorical:
        #attribute_val_counts = randomForest.obtain_class_count(data_set, chosen_attribute_index)
        # Order classes in correct order (class 0 counts, class 1 counts, class 2 counts)
        #attribute_val_counts = sorted(attribute_val_counts, key=lambda x: x[0])
        attribute_val_counts = [(0, 0), (1, 0), (2, 0)]
        partitions = randomForest.partition_dataset(data_set, attribute_val_counts, chosen_attribute_index)
    else:
        # First partition contains instances with chosen attribute value <= threshold, second partition represents > threshold.
        partitions = [[], []] 
        for instance in data_set:
            if (float)(instance[chosen_attribute_index]) <= (float)(attribute_averages[chosen_attribute_index]):
                partitions[0].append(instance)
            else:
                partitions[1].append(instance)
    
    return partitions

def make_it_pretty(lst):

    for elem in lst:
        print(elem)




def explore_decision_tree(instance, current_node, is_categorical):

    while len(current_node.children) > 0:
        for child in current_node.children:
            if len(child.children) > 0:
                attribute_index = (int)(child.name[0]) # If non-leaf node, child's name fields would be (attribute index, data set, attribute value)
            else:
                attribute_index = (int)(child.name[3]) # If leaf node, child's name fields would be (classification, data set, attribute value, attribute index)
            
            # If the value of attribute matches the value of attribute in one of the items in dataset, move onto this child node
            if is_categorical:
                if (((float)(instance[attribute_index])) == (float)(child.name[2])):
                    current_node = child
                    break
            else:
                # If value of attribute is <= threshold and the attribute value in dataset is (leq, threshold), move onto this child node
                if (((float)(instance[attribute_index])) <= child.name[2][1]) and (child.name[2][0] == "leq"):
                    current_node = child
                    break
                # If value of attribute is > threshold and the attribute value in dataset is (gr, threshold), move onto this child node
                if (((float)(instance[attribute_index])) > child.name[2][1]) and (child.name[2][0] == "gr"):
                    current_node = child
                    break

    return current_node.name[0] # Returns the class of the leaf node in decision tree
