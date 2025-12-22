from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate(model, X_test, y_test, name: str, out_dir: str) -> dict:
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=3)

    os.makedirs(out_dir, exist_ok=True)

    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_{name.lower()}.png"), dpi=200)
    plt.close()

    return {"model": name, "macro_f1": float(macro_f1), "report": report}
