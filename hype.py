import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Define hyperparameter search space
param_dist = {
    "max_depth": [3, None],
    "max_features": randint(1, 9),
    "min_samples_leaf": randint(1, 9),
    "criterion": ["gini", "entropy"]
}

# Create and tune decision tree
tree = DecisionTreeClassifier()
search = RandomizedSearchCV(tree, param_dist, cv=5, n_iter=10, random_state=42)
search.fit(X, y)

# Print best results
print("Best Parameters:", search.best_params_)
print("Best Score:", search.best_score_)
