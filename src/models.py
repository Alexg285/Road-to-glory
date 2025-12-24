from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logreg(X_train, y_train, random_state: int = 42):
    model = LogisticRegression(
        max_iter=5000,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
