import pandas as pd
import numpy as np
import random
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def logreg(X_train, y_train, X_test, y_test, n_iter=20):

    """
    Hyperparameter search for Logistic Regression models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : array-like
        Training target values.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : array-like
        Test target values.
    n_iter : int
        Number of random hyperparameter configurations to evaluate.

    Returns
    -------
    pd.DataFrame
        A DataFrame sorted by descending F1 score containing metrics.
    """

    param_distributions = [
        {'C': [0.001, 0.01, 0.1, 1, 10, 100],
         'penalty': ['l1', 'l2'],
         'solver': ['liblinear'],
         'class_weight': [None, 'balanced']},
        {'C': [0.001, 0.01, 0.1, 1, 10, 100],
         'penalty': ['l2', None],
         'solver': ['lbfgs'],
         'class_weight': [None, 'balanced']}
    ]

    valid_params = []
    for param_group in param_distributions:
        keys, values = zip(*param_group.items())
        for combination in product(*values):
            valid_params.append(dict(zip(keys, combination)))

    sampled_params = random.sample(valid_params, min(n_iter, len(valid_params)))

    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []

    for i, params in enumerate(sampled_params):
        model = LogisticRegression(**params, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        best_thresh, best_f1 = 0.5, 0
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_thresh, best_f1 = t, f1

        y_pred = (y_proba >= best_thresh).astype(int)
        results.append({
            "params": params,
            "optimal_threshold": best_thresh,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": best_f1,
            "roc_auc": roc_auc_score(y_test, y_proba)
        })

    df = pd.DataFrame(results).sort_values(by='f1', ascending=False).reset_index(drop=True)
    return df
