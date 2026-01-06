import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from .config import LABELS, LABEL_IDS


def evaluate_model(model, test_df, features, model_name):
    X_test = test_df[features]
    y_true = test_df["y_ord"].astype(int).values
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n" + "-" * 60)
    print(f"{model_name}")
    print("-" * 60)
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1 : {macro_f1:.3f}\n")
    print(classification_report(
        y_true, y_pred,
        labels=LABEL_IDS,
        target_names=LABELS,
        digits=3,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)

    return {
        "name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_df": cm_df
    }