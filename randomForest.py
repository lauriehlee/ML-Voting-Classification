
import csv
import math
import copy
import numpy as np
import random
import modifiedDecisionTree
from anytree import Node
import math
import matplotlib.pyplot as plt


def generate_random_forest(training_instances, ntrees, original_attributes_list, is_categorical, idx_of_class, minimal_size_for_split, minimal_gain, maximal_depth):

    random_forest = []
    # Let D be the original training set which contains N training instances, each with X attributes
    size_training_set = len(training_instances)
    
    # For each bootstrap b = 1,..., B:
    for n in range(ntrees):
        # Construct a bootstrap dataset of size N by sampling from D with replacement
        bootstrap = random.choices(training_instances, k=size_training_set)
        

        # Generate thresholds if numerical attributes by finding the average of that attribute’s values in the partition of instances associated with the current node
        attribute_averages = []
        if not is_categorical:
            bootstrap = [[float(item) for item in sub_array] for sub_array in bootstrap]
            attribute_averages = np.mean(np.array(bootstrap), axis=0)

        # Train a decision tree based on this bootstrap by recursively:
        # (1) Picking a random subset of m = sqrt(X) attributes
        # (2) Out of these, select the best attribute to split the current node (e.g., based on Information Gain)
        # (3) Add the node to the tree and use it to partition the data into disjoint subsets
        data_entries = copy.deepcopy(bootstrap) 
        attributes_list = copy.deepcopy(original_attributes_list)

        root = Node(("", data_entries))
        lst_of_classifications = []
        curr_depth = 0
        learned_tree = modifiedDecisionTree.decision_tree(root, lst_of_classifications, attributes_list, original_attributes_list, is_categorical, idx_of_class, attribute_averages, minimal_size_for_split, minimal_gain, maximal_depth, curr_depth)

        # Add learned tree to random forest.
        random_forest.append(learned_tree)
    
    # Return the ensemble of learned trees
    return random_forest

def make_decision(instance, random_forest, is_categorical):
    # Given the random forest, predict on each of these trees 
    decisions = []
    for tree in random_forest:
        decision = modifiedDecisionTree.explore_decision_tree(instance, tree, is_categorical)
        classification = []
        # Appended to singular array to be compatible with method obtain_class_count
        classification.append(decision) 
        decisions.append(classification)
    
    # Return the majority vote --> final classification
    class_counts = obtain_class_count(decisions, 0)

    maximum = -math.inf
    majority_class = None
    for clss, count in class_counts:
        if count == maximum:
            if random.randint(0,1) == 0:
                majority_class = clss
        if count > maximum:
            maximum = count
            majority_class = clss

    return (majority_class, instance)

def obtain_class_count(dataset, idx_of_class):
    # classes = list of tuples described by: (class, count of class)
    
    classes = []
    if len(dataset) > 0:
        classes.append((dataset[0][idx_of_class], 0))

        for instance in dataset:
            # Iterate through current possible classes. If current instance matches 
            # class whose count is currently tracked, increment count of class.
            if any(item[0] == instance[idx_of_class] for item in classes):
                for key, val in classes:
                    if instance[idx_of_class] == key:
                        classes.remove((key, val))
                        classes.append((key, val + 1))
                        break
            else:
                # If class of instance does not match any class whose count is currently tracked,
                # Add the instance's class to tracked classes and initialize with count 1.
                classes.append((instance[idx_of_class], 1))

    return classes

def partition_dataset(dataset, classes, idx_of_class):
    # Split original dataset into respective classes
    # split_dataset initialized to an array containing x empty arrays, where x = # of classes
    split_dataset = []
    for clss, count in classes:
        new_class = []
        split_dataset.append(new_class)
    
    for instance in dataset:
        index = -1
        # Find the index of class in classes to add instance to corresponding class in split_dataset
        for idx, (key, val) in enumerate(classes):
            if (float)(key) == (float)(instance[idx_of_class]):
                index = idx
                break
        split_dataset[index].append(instance)
    
    return split_dataset

def obtain_folds(original_dataset, idx_of_class):
    # classes = list of tuples described by: (class, count of class)
    classes = obtain_class_count(original_dataset, idx_of_class)

    # Find proportion of instances of each class in original dataset 
    # proportions = list of tuples described by: (class, proportion of class in relation to dataset)
    proportions = []
    total_instances = len(original_dataset)
    for clss, count in classes:
        proportions.append((clss, ((float)(count))/((float)(total_instances))))
    
    # Split original dataset into respective classes
    # split_dataset initialized to an array containing x empty arrays, where x = # of classes
    split_dataset = partition_dataset(original_dataset, classes, idx_of_class)

    # For each of the 10 folds:
    # Create fold, preserving proportion of each class in each fold
    # Append fold to folds
    folds = []
    size_of_fold = math.floor((float)(total_instances) / (float)(10))

    for j in range(9):
        new_fold = []
        curr_class_idx = 0
        for clss, proportion in proportions:
            num_to_add = proportion * size_of_fold
            # If number's tenth place < .5, round down. Otherwise, round up.
            if num_to_add - math.floor(proportion * size_of_fold) < 0.5:
                num_to_add = math.floor(proportion * size_of_fold)
            else:
                num_to_add = math.ceil(proportion * size_of_fold)

            new_fold.append(copy.deepcopy(split_dataset[curr_class_idx][:num_to_add]))
            # Change the data set so it remains disjoint to select new instances for following folds 
            split_dataset[curr_class_idx] = copy.deepcopy(split_dataset[curr_class_idx][num_to_add:])
            curr_class_idx += 1

        new_fold = [element for sublist in new_fold for element in sublist]

        folds.append(new_fold)

    # For the last fold, put in the rest of data.
    new_fold = []
    for class_data in split_dataset:
        new_fold.append(class_data)
    
    new_fold = [element for sublist in new_fold for element in sublist]

    folds.append(new_fold)

    # Return folds
    return folds


def accuracy(classifications, idx_of_class):
    # Classifications organized as: (class, instance)
    correct = 0
    total_instances = 0
    for clss, instance in classifications:
        total_instances += 1
        if (float)(clss) == (float)(instance[idx_of_class]):
            correct += 1
    
    #print("Accuracy: ", (float)(correct) / (float)(total_instances))
    return (float)(correct) / (float)(total_instances)

def precision(classifications, idx_of_class, classes_list):

    sum = 0
    for curr_class in classes_list:
        predicted_positive = 0
        true_positive = 0
        for clss, instance in classifications:
            if (float)(clss) == (float)(curr_class):
                predicted_positive += 1
                if (float)(instance[idx_of_class]) == (float)(clss):
                    true_positive += 1

        if predicted_positive == 0:
            sum += 0
        else:
            sum += ((float)(true_positive) / (float)(predicted_positive))
    
    #print("Precision: ", sum / (float)(len(classes_list)))
    return sum / (float)(len(classes_list))

def recall(classifications, idx_of_class, classes_list):
    
    sum = 0
    for curr_class in classes_list:
        correct_prediction = 0
        positive_instances = 0
        for clss, instance in classifications:
            if (float)(instance[idx_of_class]) == (float)(curr_class):
                positive_instances += 1
                if (float)(clss) == (float)(curr_class):
                    correct_prediction += 1
                
        if positive_instances == 0:
            sum += 0
        else:
            sum += ((float)(correct_prediction) / (float)(positive_instances))
    
    #print("Recall: ", sum / (float)(len(classes_list)))
    return sum / (float)(len(classes_list))

def f1_score(precision, recall, beta):
    numerator = ((1 + math.pow(beta, 2)) * (precision) * (recall))
    denominator = (((math.pow(beta, 2)) * precision) + recall)
    if denominator == 0:
        #print("F1 Score: ", 0)
        return 0
    
    f1_score = numerator / denominator
    #print("F1 Score: ", f1_score)
    return f1_score

def show_graph(points, measurement_type, title):

    x = np.array([coord[0] for coord in points])
    y = np.array([coord[1] for coord in points])

    plt.plot(x, y)
    plt.xlabel('ntree Value')
    plt.ylabel(measurement_type)
    plt.title(title)
    #ax0.set_ylim(0.72, 1.0)    
    
    plt.show()

if __name__ == "__main__":
    #Can modify these stopping criteria values
    minimal_size_for_split = 7      # Depends on the size of the dataset. Too little size (e.g., 0) -> could go on forever and would hit maximal depth, overfitting, poor generalization; Too large size (e.g., 400 for both data sets since both are < 500 instances) --> tree too shallow, poor predictions
    minimal_gain = 0                 # Too little gain (e.g., 0) -> could go on forever and would hit maximal depth, overfitting, poor generalization; Too much gain (e.g., 0.9) --> tree too shallow, poor predictions
    maximal_depth = 16          # Too little depth (e.g. 0) -> tree too shallow, poor predictions; Too much depth (e.g., 10000) --> overfitting, poor generalization
    # (10, 0, 10 is great for wine dataset, retesting new thresholds for numerical attributes)
    # (7, 0, 16 is great for vote dataset - categorical, dont need to keep retesting categorical attributes)
    
    print("Minimal size for split set to: ", minimal_size_for_split)
    print("Minimal gain set to: ", minimal_gain)
    print("Maximal depth set to: ", maximal_depth)

    
    # Extract instances from Wine Dataset.
    # Establish reading through csv: [0] = class; [1, 13] = attributes
    wine_csv_name = "datasets/hw3_wine.csv"
    wine_original_dataset = []
    with open(wine_csv_name, mode = 'r') as csvfile:
        file = csv.reader(csvfile)
        i = 0
        for line in file: 
            if i == 0:
                wine_original_attributes_list = line[0].split('\t')[1:]
                i += 1
                continue
            if len(line) > 0: 
                wine_original_dataset.append(line[0].split('\t'))

    # Implement the stratified cross-validation technique with k = 10 folds to obtain training set & testing set
    idx_of_wine_class = 0
    wine_folds = obtain_folds(wine_original_dataset, idx_of_wine_class)
    wine_classes = [elem[0] for elem in obtain_class_count(wine_original_dataset, idx_of_wine_class)]
    print("Number of wine instances: ", len(wine_original_dataset))
    print("Length of wine folds: ", len(wine_folds))
    print("Length of 1st wine fold: ", len(wine_folds[0]))
    print("Length of 3rd wine fold: ", len(wine_folds[2]))
    print("Length of 7th wine fold: ", len(wine_folds[6]))
    print("Length of 10th wine fold: ", len(wine_folds[9]))

    # For each fold as testing set and the rest as a conglomerate training set:
        # For each ntree value:
        # Predict on the Wine Dataset using Random Forest Algorithm.
        # Measure accuracy, precision, recall and F1 score of resulting random forest.
        # Accumulate these (x,y) = (ntree value, measurement) into their respective data-point arrays.
    
    ntree_val = [1, 5, 10, 20, 30, 40, 50]
    ntree_accuracy_data = [[], [], [], [], [], [], []] # Each array corresponds to the accuracy data collected across 10 folds for each of the 7 ntree values.
    ntree_precision_data = [[], [], [], [], [], [], []] # Each array corresponds to the precision data collected across 10 folds for each of the 7 ntree values.
    ntree_recall_data = [[], [], [], [], [], [], []] # Each array corresponds to the recall data collected across 10 folds for each of the 7 ntree values.
    ntree_f1_data = [[], [], [], [], [], [], []] # Each array corresponds to the F1 data collected across 10 folds for each of the 7 ntree values.
    wine_is_categorical = False 
    for fold in wine_folds:
        wine_testing_set = fold
        excluded_idx = wine_folds.index(fold)
        wine_training_set = [arr for idx, arr in enumerate(wine_folds) if idx != excluded_idx]
        wine_training_set = [element for sublist in wine_training_set for element in sublist]

        for val in ntree_val:
            learned_wine_trees = generate_random_forest(wine_training_set, val, wine_original_attributes_list, wine_is_categorical, idx_of_wine_class, minimal_size_for_split, minimal_gain, maximal_depth)
            decisions = []
            for instance in wine_testing_set:
                decisions.append(make_decision(instance, learned_wine_trees, wine_is_categorical))
            
            idx_of_ntree_val = ntree_val.index(val)
            ntree_accuracy_data[idx_of_ntree_val].append(accuracy(decisions, idx_of_wine_class))
            precision_val = precision(decisions, idx_of_wine_class, wine_classes)
            ntree_precision_data[idx_of_ntree_val].append(precision_val) 
            recall_val = recall(decisions, idx_of_wine_class, wine_classes)
            ntree_recall_data[idx_of_ntree_val].append(recall_val) 
            beta = 1
            ntree_f1_data[idx_of_ntree_val].append(f1_score(precision_val, recall_val, beta)) 
  
    # Find average accuracy across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_accuracy_data = [np.average(nums) for nums in ntree_accuracy_data]
    ntree_accuracy_data = list(zip(ntree_val, ntree_accuracy_data))
    print("Avg Accuracy of Wine Dataset based on ntree Value: ", ntree_accuracy_data)
    show_graph(ntree_accuracy_data, "Avg Accuracy", "Average Accuracy of Wine Dataset based on ntree Value")

    # Find average precision across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_precision_data = [np.average(nums) for nums in ntree_precision_data]
    ntree_precision_data = list(zip(ntree_val, ntree_precision_data))
    print("Avg Precision of Wine Dataset based on ntree Value: ", ntree_precision_data)
    show_graph(ntree_accuracy_data, "Avg Precision", "Average Precision of Wine Dataset based on ntree Value")


    # Find average recall across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_recall_data = [np.average(nums) for nums in ntree_recall_data]
    ntree_recall_data = list(zip(ntree_val, ntree_recall_data))
    print("Avg Recall of Wine Dataset based on ntree Value: ", ntree_recall_data)
    show_graph(ntree_accuracy_data, "Avg Recall", "Average Recall of Wine Dataset based on ntree Value")

    # Find average F1 score across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_f1_data = [np.average(nums) for nums in ntree_f1_data]
    ntree_f1_data = list(zip(ntree_val, ntree_f1_data))
    print("Avg F1 Scores of Wine Dataset based on ntree Value: ", ntree_f1_data)
    show_graph(ntree_accuracy_data, "Avg F1", "Average F1 of Wine Dataset based on ntree Value")

    
    # Extract instances from Voting Dataset.
    # Establish reading through csv: [0, 15] = attributes; [16] = political party
    votes_csv_name = "datasets/hw3_house_votes_84.csv"
    votes_original_dataset = []
    with open(votes_csv_name, mode = 'r') as csvfile:
        file = csv.reader(csvfile)
        i = 0
        for line in file: 
            if i == 0:
                votes_original_attributes_list = line[:16]
                i += 1
                continue
            if len(line) > 0: 
                votes_original_dataset.append(line)


    
    # Implement the stratified cross-validation technique with k = 10 folds to obtain training set & testing set
    idx_of_vote_class = 16
    vote_folds = obtain_folds(votes_original_dataset, 16)
    vote_classes = [elem[0] for elem in obtain_class_count(votes_original_dataset, idx_of_vote_class)]
    print("Number of vote instances: ", len(votes_original_dataset))
    print("Length of vote folds: ", len(vote_folds))
    print("Length of 1st vote fold: ", len(vote_folds[0]))
    print("Length of 3rd vote fold: ", len(vote_folds[2]))
    print("Length of 7th vote fold: ", len(vote_folds[6]))
    print("Length of 10th vote fold: ", len(vote_folds[9]))

    # For each fold as testing set and the rest as a conglomerate training set:
        # For each ntree value:
        # Predict on the Vote Dataset using Random Forest Algorithm.
        # Measure accuracy, precision, recall and F1 score of resulting random forest.
        # Accumulate these (x,y) = (ntree value, measurement) into their respective data-point arrays.
    ntree_val = [1, 5, 10, 20, 30, 40, 50]
    ntree_accuracy_data = [[], [], [], [], [], [], []] # Each array corresponds to the accuracy data collected across 10 folds for each of the 7 ntree values.
    ntree_precision_data = [[], [], [], [], [], [], []] # Each array corresponds to the precision data collected across 10 folds for each of the 7 ntree values.
    ntree_recall_data = [[], [], [], [], [], [], []] # Each array corresponds to the recall data collected across 10 folds for each of the 7 ntree values.
    ntree_f1_data = [[], [], [], [], [], [], []] # Each array corresponds to the F1 data collected across 10 folds for each of the 7 ntree values.
    vote_is_categorical = True 
    for fold in vote_folds:
        vote_testing_set = fold
        excluded_idx = vote_folds.index(fold)
        vote_training_set = [arr for idx, arr in enumerate(vote_folds) if idx != excluded_idx]
        vote_training_set = [element for sublist in vote_training_set for element in sublist]

        for val in ntree_val:
            learned_vote_trees = generate_random_forest(vote_training_set, val, votes_original_attributes_list, vote_is_categorical, idx_of_vote_class, minimal_size_for_split, minimal_gain, maximal_depth)
            decisions = []
            for instance in vote_testing_set:
                decisions.append(make_decision(instance, learned_vote_trees, vote_is_categorical))
            
            idx_of_ntree_val = ntree_val.index(val)
            ntree_accuracy_data[idx_of_ntree_val].append(accuracy(decisions, idx_of_vote_class))
            precision_val = precision(decisions, idx_of_vote_class, vote_classes)
            ntree_precision_data[idx_of_ntree_val].append(precision_val) 
            recall_val = recall(decisions, idx_of_vote_class, vote_classes)
            ntree_recall_data[idx_of_ntree_val].append(recall_val) 
            beta = 1 
            ntree_f1_data[idx_of_ntree_val].append(f1_score(precision_val, recall_val, beta)) 


    # Find average accuracy across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_accuracy_data = [np.average(nums) for nums in ntree_accuracy_data]
    ntree_accuracy_data = list(zip(ntree_val, ntree_accuracy_data))
    print("Avg Accuracy of Vote Dataset based on ntree Value: ", ntree_accuracy_data)
    show_graph(ntree_accuracy_data, "Avg Accuracy", "Average Accuracy of Vote Dataset based on ntree Value")

    # Find average precision across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_precision_data = [np.average(nums) for nums in ntree_precision_data]
    ntree_precision_data = list(zip(ntree_val, ntree_precision_data))
    print("Avg Precision of Vote Dataset based on ntree Value: ", ntree_precision_data)
    show_graph(ntree_accuracy_data, "Avg Precision", "Average Precision of Vote Dataset based on ntree Value")


    # Find average recall across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_recall_data = [np.average(nums) for nums in ntree_recall_data]
    ntree_recall_data = list(zip(ntree_val, ntree_recall_data))
    print("Avg Recall of Vote Dataset based on ntree Value: ", ntree_recall_data)
    show_graph(ntree_accuracy_data, "Avg Recall", "Average Recall of Vote Dataset based on ntree Value")

    # Find average F1 score across the 10 folds and accumuluate these (x,y) = (ntree value, avg measurement) into their respective data-point arrays.
    ntree_f1_data = [np.average(nums) for nums in ntree_f1_data]
    ntree_f1_data = list(zip(ntree_val, ntree_f1_data))
    print("Avg F1 Scores of Vote Dataset based on ntree Value: ", ntree_f1_data)
    show_graph(ntree_accuracy_data, "Avg F1", "Average F1 of Vote Dataset based on ntree Value")

    # Show the 8 graphs that are generated from these calculations


