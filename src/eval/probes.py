import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score
from typing import Dict, Tuple

def train_and_evaluate_continuous_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[float, float]:
    """
    Train a Ridge regression probe for continuous variables (mu or sigma).
    Returns mean R^2 and standard deviation across K folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
        
    return np.mean(scores), np.std(scores)

def train_and_evaluate_categorical_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[float, float]:
    """
    Train a Logistic Regression probe for categorical variables (family).
    Returns mean accuracy and standard deviation across K folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scaling is usually helpful but for simplicity in MVP we use LogisticRegression directly.
        # Max_iter increased to ensure convergence.
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        
    return np.mean(scores), np.std(scores)

def evaluate_layer_probes(X: np.ndarray, y_mu: np.ndarray, y_sigma: np.ndarray, y_family: np.ndarray) -> Dict[str, float]:
    """
    Evaluate all probes for a single layer's hidden states.
    X: shape (N, hidden_size)
    """
    mu_r2, _ = train_and_evaluate_continuous_probe(X, y_mu)
    sigma_r2, _ = train_and_evaluate_continuous_probe(X, y_sigma)
    family_acc, _ = train_and_evaluate_categorical_probe(X, y_family)
    
    return {
        "mu_r2": float(mu_r2),
        "sigma_r2": float(sigma_r2),
        "family_accuracy": float(family_acc)
    }
