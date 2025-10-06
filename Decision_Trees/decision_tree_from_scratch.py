import numpy as np
from collections import Counter

# Node class for the tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # column index to split on
        self.threshold = threshold      # value to compare
        self.left = left                # left child node
        self.right = right              # right child node
        self.value = value              # class label (for leaf node)

    def is_leaf_node(self):
        return self.value is not None


# Decision Tree class
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split # it means no further splitting if number of rows is less than min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features  # which column is going to be split
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)  # it means use all features(columns) if not specified; but in betn of total columns and specified by the user
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))  # it means unique features from label column

        # stopping conditions
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples <= self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # choose random subset of features
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        # n_features = total number of features (columns) in your dataset
        # self.n_features = number of features you want to randomly select for this node (set by you or defaulted to all)
        # The function then randomly chooses self.n_features distinct feature indices from the total n_features.
        # the 'replace = False' removes the duplicates


        # find the best split
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)
        # This finds which feature (like Age or Salary) and what threshold (like Age = 30) gives maximum information gain (i.e., purest split)

        # create child nodes recursively
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]   #select all rows based on the featured indexes
            thresholds = np.unique(X_column)  # select the unique values b'coz this could be potential thresholds for splitting

            # suppose we have three features_0,features_1,features_2
            # we want feat_idx = 1
            # so we got thresholds = [30000, 60000, 80000] <= example

            for thr in thresholds:
                # For each loop, we ask:
                # “If I split the dataset at this threshold, how much better (or purer) will my groups become?”

                # calculate information gain
                gain = self._info_gain(y, X_column, thr)
                # How much uncertainty (entropy) do we reduce if we split at this threshold?
                # Higher gain = better split ✅

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _info_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        # Why?
        # If a split puts all rows into one side and leaves the other side empty,
        # then that’s not a valid split — no actual division happened.
        # So, we just return 0 information gain for that case.

        # weighted average child entropy
        n = len(y)  #n = total samples (say 4)
        n_l, n_r = len(left_idx), len(right_idx)   #n_l = samples in left node (say 2)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])   # e_l = entropy of left node #e_r = entropy of right node
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r   

        # information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        # It divides the dataset into two groups — based on the split_thresh (threshold).
        # left_idxs = indexes of rows where the feature value ≤ threshold
        # right_idxs = indexes of rows where the feature value > threshold
        # These indices are used to select the corresponding rows from X and y

        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y) #np.bincount(y) → counts how many of each class
        ps = hist / len(y)    #ps → converts counts to probabilities
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
