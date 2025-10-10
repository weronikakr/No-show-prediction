import pandas as pd
from models.logreg import logreg

if __name__ == "__main__":
    
    X_train = pd.read_csv('data/X_train_res.csv')
    y_train = pd.read_csv('data/y_train_res.csv').values.ravel()
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    metrics_df = logreg(X_train, y_train, X_test, y_test, n_iter=20)

    metrics_df.to_csv('results/models_metrics.csv', index=False)
    print(metrics_df.head(3))