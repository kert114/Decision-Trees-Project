import numpy as np
import abc
import sys

class TreeNode(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def depth(self):
        pass

    @abc.abstractmethod
    def node_count(self):
        pass

    # @abc.abstractmethod
    # def visualize(self):
    #     pass
    
    @abc.abstractmethod
    def predict(self, input):
        pass

    def predict_batch(self, input_batch):
        return [self.predict(x) for x in input_batch]

    # A prune method for improving the accuracy of our decision tree when the
    # dataset contains noise.
    # Returns a tuple TreeNode where TreeNode is the pruned node.
    @abc.abstractmethod
    def prune(self, validation_dataset):
        pass

class LeafNode(TreeNode):
    def __init__(self, prediction):
        self.prediction = prediction

    def depth(self):
        return 1

    def node_count(self):
        return 1

    def predict(self, input):
        return self.prediction

    def prune(self, validation_dataset):
        return self

class DecisionNode(TreeNode):
    def __init__(self, left_branch, right_branch, split_value, split_column):

        self.split_value = split_value
        self.split_column = split_column

        self.left_branch = left_branch
        self.right_branch = right_branch

    def depth(self):
        return max(self.right_branch.depth(), self.left_branch.depth()) + 1

    def node_count(self):
        return 1 + self.left_branch.node_count() + self.right_branch.node_count()

    def predict(self, input):
        if input[self.split_column] > self.split_value:
            return self.right_branch.predict(input)
        else:
            return self.left_branch.predict(input)

    def prune(self, validation_dataset):
        left_dataset, right_dataset = split_dataset(validation_dataset, self.split_column, self.split_value)
        # Call prune on children with their split of the validation dataset
        new_left = self.left_branch.prune(left_dataset)
        new_right = self.right_branch.prune(right_dataset)
        self.left_branch = new_left
        self.right_branch = new_right
        # If they're both leaves, then calculate accuracy and then merge them together
        if isinstance(new_left, LeafNode) and isinstance(new_right, LeafNode):
            if len(validation_dataset) == 0:
                return new_left
            # calculate accuracy of self
            current_confusion_matrix = create_confusion_matrix(validation_dataset[:, -1], self.predict_batch(validation_dataset))
            current_accuracy = calculate_accuracy(current_confusion_matrix)

            # create a new leaf node with majority vote
            values, counts = np.unique(validation_dataset[:, -1], return_counts=True)
            majority_vote = values[np.argmax(counts)]
            candidate_node = LeafNode(majority_vote)

            # Test the new merged node on the validation dataset
            candidate_confusion_matrix = create_confusion_matrix(validation_dataset[:, -1], candidate_node.predict_batch(validation_dataset))
            candidate_accuracy = calculate_accuracy(candidate_confusion_matrix)

            # If the accuracy increases or is the same, then we return the merged node
            if candidate_accuracy >= current_accuracy:
                return candidate_node
            
        # Else return self
        return self

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
    accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
    return accuracy

# Calculate recall per class
# confusion_matrix: confusion matrix (4x4 numpy matrix)
# Returns a 4-element array with recall for each class
def calculate_recall(confusion_matrix):
    # Recall = TP / (TP + FN)
    # TP = diagonal element
    # FN = sum of row - diagonal element
    row_sums = confusion_matrix.sum(axis=1)
    diagonals = confusion_matrix.diagonal()
    return diagonals/row_sums

# Calculate precision per class
# confusion_matrix: confusion matrix (4x4 numpy matrix)
# Returns a 4-element array with precision for each class
def calculate_precision(confusion_matrix):
    # Precision = TP / (TP + FP)
    # TP = diagonal element
    # FP = sum of column - diagonal element
    return calculate_recall(confusion_matrix.T)

# Calculate f1 measure per class
# confusion_matrix: confusion matrix (4x4 numpy matrix)
# beta: the weight of precision in the metric
# Returns a 4-element array with f measure for each class
def calculate_f_measures(confusion_matrix, beta=1):
    # F1 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    precision = calculate_precision(confusion_matrix)
    recall = calculate_recall(confusion_matrix)
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def evaluate(test_dataset, trained_tree):
    confusion_matrix = create_confusion_matrix(test_dataset[:, -1], trained_tree.predict_batch(test_dataset))
    return calculate_accuracy(confusion_matrix)

def train_on_dataset(datset, pruning=False):
    split_data = split_training_data(dataset)

    cumulative_confusion_matrix = np.zeros((4, 4))
    cumulative_depth = 0

    # 10-fold cross-validation, so we train on 9/10 of the data and test on 1/10
    # Repeat for each 1/10 test set
    # Average evaluation metrics to estimate global performance (estimate for if tree was trained on whole dataset)
    for i in range(10):
        test_data = split_data[i]
        # print(split_data[:i] + split_data[i+1:])
        training_validation_data = split_data[:i] + split_data[i+1:]

        best_tree = None
        best_accuracy = -1

        # Validation to select best tree
        for j in range(9):
            training_data = np.concatenate(training_validation_data[:j] + training_validation_data[j+1:])
            validation_data = training_validation_data[j]
            
            ds = build_decision_tree(training_data)
            if pruning:
                ds = ds.prune(validation_data)
            candidate_accuracy = evaluate(validation_data, ds)

            if candidate_accuracy > best_accuracy:
                best_accuracy = candidate_accuracy
                best_tree = ds
            
        confusion_matrix = create_confusion_matrix(test_data[:, -1], best_tree.predict_batch(test_data))
        cumulative_confusion_matrix += confusion_matrix
        cumulative_depth += best_tree.depth()
    
    
    avg_accuracy = calculate_accuracy(cumulative_confusion_matrix)
    avg_recall = calculate_recall(cumulative_confusion_matrix)
    avg_precision = calculate_precision(cumulative_confusion_matrix)
    avg_f1_measure = calculate_f_measures(cumulative_confusion_matrix)
    avg_depth = cumulative_depth / 10

    return cumulative_confusion_matrix, avg_accuracy, avg_recall, avg_precision, avg_f1_measure, avg_depth

def pretty_print(confusion_matrix, avg_accuracy, avg_recall, avg_precision, avg_f1_measure, avg_depth):
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Average Accuracy:", avg_accuracy)
    print("Average Per-Class Recall:", avg_recall)
    print("Average Per-Class Precision:", avg_precision)
    print("Average Per-Class F1 Measure:", avg_f1_measure)
    print("Average Depth:", avg_depth)

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    dataset = load_data(dataset_path)

    print("=====Non pruned=====")
    evaluation_metrics = train_on_dataset(dataset)
    pretty_print(*evaluation_metrics)
    print("====================")
    print()
    print("=======Pruned=======")
    evaluation_metrics = train_on_dataset(dataset, pruning=True)
    pretty_print(*evaluation_metrics)
    print("====================")
