# Example data (simple binary classification)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from decision_tree_from_scratch import DecisionTree

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)

acc = np.sum(preds == y_test) / len(y_test)
print("Accuracy:", acc)
