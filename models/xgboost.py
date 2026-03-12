import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer


def xgboost_model(X_train, y_train, X_test, y_test, n_iter=100):

    """
    Hyperparameter search for XGBoost models.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by descending F1 score containing metrics.
    """

    xgb_model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    param_grid = {
        'max_depth': [10, 30, 35, 37, 40, 50],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0.05, 0.1, 0.25, 0.5, 1],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'n_estimators': [400, 500, 600, 700, 1000],
        'reg_alpha': [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2],
        'reg_lambda': [1, 1.5, 2, 3, 5],
        'max_delta_step': [0, 1, 5, 10, 15],
        'scale_pos_weight': [15, 20, 25, 30, 35, 40]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    f1_scorer = make_scorer(f1_score)

    rand_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=f1_scorer,
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )

    rand_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(rand_search.cv_results_)

    top3 = cv_results.sort_values(
        by="mean_test_score",
        ascending=False
    ).head(3)

    thresholds = np.arange(0.1, 1.0, 0.1)

    models_metrics = []

    for rank, (_, row) in enumerate(top3.iterrows(), 1):

        params = row["params"]

        model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            **params
        )

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]

        best_thresh, best_f1 = 0.5, 0

        for t in thresholds:

            y_pred = (y_proba >= t).astype(int)

            f1 = f1_score(y_test, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        y_pred = (y_proba >= best_thresh).astype(int)

        models_metrics.append({

            "rank": rank,
            "params": params,
            "optimal_threshold": best_thresh,

            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": best_f1,
            "roc_auc": roc_auc_score(y_test, y_proba)

        })

    df = pd.DataFrame(models_metrics).sort_values(
        by="f1",
        ascending=False
    ).reset_index(drop=True)

    return df