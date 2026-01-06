import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

def train_logreg(train_df, features):
    X = train_df[features]
    y = train_df["y_ord"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])
    model.fit(X, y)
    return model


def train_xgb(train_df, features, random_state=42):
    X = train_df[features]
    y = train_df["y_ord"].astype(int)

    classes = np.sort(y.unique())
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = dict(zip(classes, cw))
    sample_weight = y.map(class_weight).values

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        ))
    ])

    model.fit(X, y, clf__sample_weight=sample_weight)
    return model