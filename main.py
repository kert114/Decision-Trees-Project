import random as random
import numpy as np
import matplotlib.pyplot as plt
import abc

class TreeNode(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def depth(self):
        pass

    # @abc.abstractmethod
    # def visualize(self):
    #     pass
    
    @abc.abstractmethod
    def predict(self, input):
        pass

    def predict_batch(self, input_batch):
        return [self.predict(x) for x in input_batch]

    # @abc.abstractmethod
    # def prune(self):
    #     pass

class LeafNode(TreeNode):
    def __init__(self, prediction):
        self.prediction = prediction

    def depth(self):
        return 1

    def predict(self, input):
        return self.prediction

class DecisionNode(TreeNode):
    def __init__(self, left_branch, right_branch, split_value, split_column):

        self.split_value = split_value
        self.split_column = split_column

        self.left_branch = left_branch
        self.right_branch = right_branch

    def depth(self):
        return max(self.right_branch.depth(), self.left_branch.depth()) + 1

    def predict(self, input):
        if input[self.split_column] > self.split_value:
            return self.right_branch.predict(input)
        else:
            return self.left_branch.predict(input)

def build_decision_tree(dataset):
    # Find the best split.
    # If the best split is not good enough, return a leaf node.
    # Otherwise, split the dataset and build a decision node.
    # Recursively build the left and right branches.
    # Return the decision node.

    # best_split = find_split(dataset)
    # if best_split is None:
    #     return LeafNode(majority_vote(dataset))

    labels = np.unique(dataset[:, -1])

    if len(labels) == 1:
        return LeafNode(labels[0])
        
    best_split = find_split(dataset)

    split_column, split_value = best_split

    left_of_split, right_of_split = split_dataset(dataset, split_column, split_value)

    left_branch = build_decision_tree(left_of_split)
    right_branch = build_decision_tree(right_of_split)

    return DecisionNode(left_branch, right_branch, split_value, split_column)


def load_data(filepath):
    return np.loadtxt(filepath)

def find_split(dataset):
    # Find the best split for the dataset.
    # This is the column and value that will give the most information gain.
    # Return the column and value of the best split.
    
    # Find the entropy of the full dataset.
    full_dataset_entropy = entropy(dataset)

    max_information_gain = 0
    best_split = None

    for col in range(dataset.shape[1] - 1):
        # Find the unique values in the column.
        # Sort them so we can find the midpoints.
        unique_values = np.unique(dataset[:, col])
        unique_values.sort()

        # For each adjacent pair of unique values, find their midpoint.
        # Split the dataset around this value.
        # Calculate the entropy of each split.
        # Calculate the information gain of the split.
        # Find the split with the highest information gain.
        for lower in range(len(unique_values) - 1):
            upper = lower + 1
            midpoint = (unique_values[lower] + unique_values[upper]) / 2

            left_of_split, right_of_split = split_dataset(dataset, col, midpoint)
            information_gain = information_gained(full_dataset_entropy, left_of_split, right_of_split)
            
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_split = (col, midpoint)
                
    return best_split

def split_dataset(dataset, column, value):
    left_of_split = dataset[dataset[:, column] <= value]
    right_of_split = dataset[dataset[:, column] > value]

    return left_of_split, right_of_split

# full_dataset_entropy = float
# left_of_split = np.array
# right_of_split = np.array
def information_gained(full_dataset_entropy, left_of_split, right_of_split):
    size_left = len(left_of_split)
    size_right = len(right_of_split)

    proportion_left = size_left / (size_left + size_right)
    proportion_right = size_right / (size_left + size_right)

    scaled_entropy_left = proportion_left * entropy(left_of_split)
    scaled_entropy_right = proportion_right * entropy(right_of_split)

    remainder = scaled_entropy_left + scaled_entropy_right

    return full_dataset_entropy - remainder

def entropy(dataset):
    labels = dataset[:, -1]

    total_elements = len(labels)

    _, num_labels = np.unique(labels, return_counts=True)

    label_proportions = num_labels / total_elements

    entropy = - sum(label_proportions * np.log2(label_proportions))
    return entropy

# Worth noting that this should split the data into num_splits arrays.
# This is so we can verify it with 10-fold cross-validation.
# Dataset should also be shuffled.
def split_training_data(dataset, num_splits=10, random_generator=np.random.default_rng()):
    shuffled_array = np.copy(dataset)
    random_generator.shuffle(shuffled_array)

    return np.array_split(shuffled_array, num_splits)

# true_labels = np.array
# predicted_labels = np.array
# returned confusion matrix has rows as true labels and columns as predicted labels
def create_confusion_matrix(true_labels, predicted_labels):
    confusion_matrix = np.zeros((4, 4))

    for i in range(len(true_labels)):
        confusion_matrix[int(true_labels[i]-1)][int(predicted_labels[i]-1)] += 1

    return confusion_matrix

def calculate_accuracy(confusion_matrix):
    accuracy = sum(np.diagonal(confusion_matrix)) / sum(confusion_matrix)
    return accuracy



if __name__ == "__main__":
    print("This is the main body!")
    dataset = load_data("./wifi_db/clean_dataset.txt")

    split_data = split_training_data(dataset)

    ds = build_decision_tree(np.concatenate(split_data[:9]))

    x = [x[:-1] for x in split_data[9]]

    y_pred = ds.predict_batch(x)

    y_true = [x[-1] for x in split_data[9]]

    cm = create_confusion_matrix(y_true, y_pred)

    print(cm)

    print(ds.depth())