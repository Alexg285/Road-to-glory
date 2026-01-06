from pathlib import Path
import pandas as pd

from .config import REQUIRED_COLUMNS


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_and_split(data_path: Path):
    """Load final dataset, split using 'split', and apply train-median imputation (no leakage)."""
    if not data_path.exists():
        raise FileNotFoundError(f"final dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    require_columns(df, REQUIRED_COLUMNS, "final_dataset1.csv")

    df = df[df["split"].isin(["train", "test"])].copy()
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Train-median imputation (safe, no leakage)
    for col in ["strength_pre", "avg_team_age"]:
        if train_df[col].isna().any() or test_df[col].isna().any():
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            test_df[col] = test_df[col].fillna(med)

    return train_df, test_df