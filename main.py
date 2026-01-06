import pandas as pd
from pathlib import Path
from src.config import FEATURES_BASELINE, FEATURES_ENRICHED
from src.data_loader import load_and_split
from src.models import train_logreg, train_xgb
from src.evaluation import evaluate_model


def main():
    print("=" * 60)
    print("Road to Glory — Model Comparison")
    print("=" * 60)

    # 1) Load + split
    print("\n1) Loading data...")
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "clean" / "final_dataset1.csv"

    train_df, test_df = load_and_split(data_path)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    # Quick sanity check: class distribution in test
    print("\nTest set class distribution (y_ord):")
    print(test_df["y_ord"].value_counts().sort_index())

    # 2) Train
    print("\n2) Training models...")
    lr_base = train_logreg(train_df, FEATURES_BASELINE)
    lr_enr = train_logreg(train_df, FEATURES_ENRICHED)
    xgb_base = train_xgb(train_df, FEATURES_BASELINE)
    xgb_enr = train_xgb(train_df, FEATURES_ENRICHED)
    print("✓ Models trained")

    # 3) Evaluate
    print("\n3) Evaluating on TEST...")
    results = {}
    results_full = {}

    r1 = evaluate_model(lr_base, test_df, FEATURES_BASELINE, "LogReg baseline")
    r2 = evaluate_model(lr_enr, test_df, FEATURES_ENRICHED, "LogReg enriched")
    r3 = evaluate_model(xgb_base, test_df, FEATURES_BASELINE, "XGB baseline")
    r4 = evaluate_model(xgb_enr, test_df, FEATURES_ENRICHED, "XGB enriched")

    for r in [r1, r2, r3, r4]:
        results[r["name"]] = {
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"]
        }
        results_full[r["name"]] = r

        
    rows = []
    for r in [r1, r2, r3, r4]:
        rows.append({
            "model": r["name"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
        })

    results_df = pd.DataFrame(rows)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    metrics_path = results_dir / "metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    print(f"\nMetrics saved to {metrics_path}")

    for name, scores in results.items():
        print(
            f"{name:20s} | "
            f"Accuracy = {scores['accuracy']:.3f} | "
            f"Macro F1 = {scores['macro_f1']:.3f}"
        )

    # Show confusion matrices for all models
    for name, r in results_full.items():
        print("\n" + "-" * 60)
        print(f"Confusion matrix — {name}")
        print("-" * 60)

        cm_df = r["confusion_df"]
        print(cm_df)

        cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)
        print("\nRow-normalized (%):")
        print((cm_norm * 100).round(1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())