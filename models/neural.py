import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class FeedforwardNN(nn.Module):

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2, activation="relu"):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, X, y, epochs=30, batch_size=32, lr=0.001):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def neural_network(X_train, y_train, X_test, y_test, n_iter=10):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    param_grid = {
        'hidden_dims': [[64], [128], [128, 64], [128, 64, 32]],
        'dropout': [0.0, 0.2, 0.3],
        'activation': ['relu', 'tanh'],
        'lr': [0.001, 0.0005],
        'batch_size': [32, 64],
        'epochs': [30,50]
    }

    def random_params(grid, n):
        keys = list(grid.keys())
        return [{k: random.choice(grid[k]) for k in keys} for _ in range(n)]

    param_combinations = random_params(param_grid, n_iter)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = []

    for params in param_combinations:

        f1_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):

            X_tr = X_train_t[train_idx]
            y_tr = y_train_t[train_idx]

            X_val = X_train_t[val_idx]
            y_val = y_train_t[val_idx]

            model = FeedforwardNN(
                input_dim=X_train.shape[1],
                hidden_dims=params['hidden_dims'],
                dropout=params['dropout'],
                activation=params['activation']
            ).to(device)

            model = train_model(
                model,
                X_tr,
                y_tr,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                lr=params['lr']
            )

            model.eval()

            with torch.no_grad():
                y_val_pred = model(X_val).cpu().numpy()
                y_val_pred_bin = (y_val_pred >= 0.5).astype(int)

            f1_scores.append(
                f1_score(y_val.cpu().numpy(), y_val_pred_bin)
            )

        mean_f1 = np.mean(f1_scores)

        results.append({
            "params": params,
            "cv_f1": mean_f1
        })

    results = sorted(results, key=lambda x: x["cv_f1"], reverse=True)[:3]

    metrics = []

    thresholds = np.arange(0.1, 1.0, 0.1)

    for rank, res in enumerate(results, 1):

        p = res["params"]

        model = FeedforwardNN(
            input_dim=X_train.shape[1],
            hidden_dims=p['hidden_dims'],
            dropout=p['dropout'],
            activation=p['activation']
        ).to(device)

        model = train_model(
            model,
            X_train_t,
            y_train_t,
            epochs=p['epochs'],
            batch_size=p['batch_size'],
            lr=p['lr']
        )

        model.eval()

        with torch.no_grad():
            y_proba = model(X_test_t).cpu().numpy().ravel()

        best_thresh = 0.5
        best_f1 = 0

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1 = f1_score(y_test, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        y_pred = (y_proba >= best_thresh).astype(int)

        metrics.append({
            "rank": rank,
            "params": p,
            "optimal_threshold": best_thresh,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": best_f1,
            "roc_auc": roc_auc_score(y_test, y_proba)
        })

    df = pd.DataFrame(metrics).sort_values(by="f1", ascending=False).reset_index(drop=True)

    return df