import pandas as pd
from models.logreg import logreg
from models.neural import neural_network

if __name__ == "__main__":
    
    X_train = pd.read_csv('data/X_train_res.csv')
    y_train = pd.read_csv('data/y_train_res.csv').values.ravel()
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    metrics_df = logreg(X_train, y_train, X_test, y_test, n_iter=20)

    metrics_df.to_csv('results/models_metrics_log.csv', index=False)
    print(metrics_df.head(3))

    metrics_df_nn = neural_network(
        X_train.values,
        y_train,
        X_test.values,
        y_test,
        n_iter=5
    )
    metrics_df_nn.to_csv('results/models_metrics_nn.csv', index=False)
    print(metrics_df_nn.head(3))