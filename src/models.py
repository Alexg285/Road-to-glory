from __future__ import annotations
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logreg(X, y):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    model.fit(X, y)
    return model

def train_rf(X_train, y_train, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
