import random as random
import numpy as np
import matplotlib.pyplot as plt

class DescisionTree:
    def __init__(self, dataset):
        split_value, split_column, left_dataset, right_dataset = find_split(dataset)

        self.split_value = split_value
        self.split_column = split_column
        self.is_leaf = False
        self.leaf_prediction = 0

        self.left_branch = DescisionTree(left_dataset)
        self.right_branch = DescisionTree(right_dataset)

    def depth(self):
        if self.is_leaf:
            return 1
        else:
            return self.right_branch.depth() + self.left_branch.depth() + 1

    def visualize(self):
        pass
    
    def predict(self, inputs):
        pass

    def predict_batch(self, input_batch):
        return [self.predict(x) for x in input_batch]

    def prune(self):
        pass

def load_data(filepath):
    pass

def find_split(dataset):
    pass

def information_gained(full_dataset_entropy, left_of_split, right_of_split):
    pass

def entropy(dataset):
    pass

# Worth noting that this should split the data into num_splits arrays.
# This is so we can verify it with 10-fold cross-validation.
# Dataset should also be shuffled.
def split_training_data(dataset, num_splits=10, random_generator=np.random.default_rng()):
    pass

def create_confusion_matrix(true_labels, predicted_labels):
    pass

if __name__ == "__main__":
    print("This is the main body!")