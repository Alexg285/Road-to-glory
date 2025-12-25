from __future__ import annotations
import os
import pandas as pd

from src.data_loader import load_matches, load_fifa_rankings, load_soccer_rankings
from src.features import build_group_features
from src.labels import build_progression_labels
from src.split import temporal_split
from src.models import train_logreg, train_rf
from src.evaluation import evaluate
from src.strength_features import add_pre_tournament_strength
from src.utils import add_tournament_year
from src.team_names import TEAM_RENAME
from src.geo_features import add_host_features
from src.age_features import add_team_age_features



FEATURE_COLS = [
    "games", "wins", "draws", "losses", "points",
    "gf", "ga", "gd", "gf_per_game", "ga_per_game",
    "strength_pre", "is_host", "avg_team_age"
]


def jls_extract_def():
    
    return 


def main() -> None:
    print("=" * 70)
    print("Road to Glory — Group-stage performance → World Cup progression")
    print("=" * 70)

    # 1) Load data
    print("\n[1] Loading matches.csv...")
    matches = load_matches("data/raw/matches.csv")
    print(f"✓ Loaded {len(matches):,} matches")

    # 2) Build features from group stage only
    print("\n[2] Building group-stage features (X)...")
    Xdf = build_group_features(matches)
    print(f"✓ Features rows: {len(Xdf):,}")

    # 3) Build labels from final tournament outcomes
    print("\n[3] Building progression labels (y)...")
    ydf = build_progression_labels(matches)
    print(f"✓ Labels rows: {len(ydf):,}")

    # 4) Merge
    print("\n[4] Merging X and y...")
    df = Xdf.merge(ydf, on=["year", "tournament_id", "team"], how="inner")
    print(f"✓ Final dataset: {len(df):,} rows")

    # --- Build team_dates (min match_date per team & tournament) ---
    matches_dt = matches.copy()
    matches_dt["match_date"] = pd.to_datetime(matches_dt["match_date"], errors="coerce")

    team_dates = (
        pd.concat(
            [
                matches_dt[["tournament_id", "match_date", "home_team_name"]].rename(
                    columns={"home_team_name": "team"}
                ),
                matches_dt[["tournament_id", "match_date", "away_team_name"]].rename(
                    columns={"away_team_name": "team"}
                ),
            ],
            ignore_index=True,
        )
        .dropna(subset=["match_date"])
        .groupby(["tournament_id", "team"], as_index=False)["match_date"]
        .min()
    )
    team_dates = add_tournament_year(team_dates)

    df = df.merge(team_dates, on=["tournament_id", "team"], how="left")

    # Fix possible year column duplication after merge (year_x / year_y)
    if "year" not in df.columns:
        if "year_x" in df.columns:
            df = df.rename(columns={"year_x": "year"})
        elif "year_y" in df.columns:
            df = df.rename(columns={"year_y": "year"})

    # If both exist, keep 'year' and drop the others
    for c in ["year_x", "year_y"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # --- Geographical features: host team ---
    worldcups = pd.read_csv("data/raw/WorldCups.csv", sep=";", encoding="utf-8-sig")
    df = add_host_features(df, worldcups)

    # --- Strength feature: FIFA fallback to Soccer ---
    print("\n[+] Loading FIFA + Soccer rankings...")
    fifa = load_fifa_rankings("data/raw/Ranking fifa 1993-2024.csv")
    soccer = load_soccer_rankings("data/raw/Ranking soccer 1901-2023.csv")

    # Harmonize team names across sources (small alias table)
    df["team"] = df["team"].replace(TEAM_RENAME)
    fifa["team"] = fifa["team"].replace(TEAM_RENAME)
    soccer["team"] = soccer["team"].replace(TEAM_RENAME)

    print("[+] Adding pre-tournament strength (FIFA -> Soccer fallback)...")
    df = add_pre_tournament_strength(df, fifa, soccer)

    miss_strength = int(df["strength_pre"].isna().sum())
    print("Missing strength_pre:", miss_strength)
    print(df[["team", "year", "strength_source", "strength_pre"]].head())

    if miss_strength > 0:
        missing_rows = (
            df[df["strength_pre"].isna()][["year", "tournament_id", "team"]]
            .drop_duplicates()
        )
        print("\nTeams with missing strength_pre:")
        print(missing_rows.to_string(index=False))

    # Impute remaining missing strength values (usually name mismatches)
    df["strength_pre"] = pd.to_numeric(df["strength_pre"], errors="coerce")
    df["strength_pre"] = df["strength_pre"].fillna(df["strength_pre"].median())

    # --- Age feature: average team age at tournament start ---
    print("\n[+] Adding average squad age (avg_team_age)...")

    players = pd.read_csv("data/raw/players.csv", sep=",")
    squads  = pd.read_csv("data/raw/squads.csv", sep=";")
    matches = pd.read_csv("data/raw/matches.csv", sep=";")

    df = add_team_age_features(df, players, squads, matches)

    print("Missing avg_team_age:", df["avg_team_age"].isna().sum())

    # Imputation simple (pour modèles)
    df["avg_team_age"] = df["avg_team_age"].fillna(df["avg_team_age"].median())

    # Quick sanity checks
    missing = df[FEATURE_COLS + ["stage_label"]].isna().sum().sum()
    print(f"NaN cells in (features + strength + label): {int(missing)}")

    # 5) Temporal split
    print("\n[5] Temporal split: train 1970–2014, test 2018 & 2022...")
    train_df, test_df = temporal_split(df)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check year parsing or filters.")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["stage_label"].astype(int).values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["stage_label"].astype(int).values

    # 6) Train
    print("\n[6] Training models...")
    lr = train_logreg(X_train, y_train)
    rf = train_rf(X_train, y_train)
    print("✓ Models trained")

    # 7) Evaluate
    print("\n[7] Evaluating on 2018 & 2022...")
    os.makedirs("results/plots", exist_ok=True)

    lr_res = evaluate(lr, X_test, y_test, "logreg", "results/plots")
    rf_res = evaluate(rf, X_test, y_test, "rf", "results/plots")

    metrics = pd.DataFrame(
        [
            {"model": lr_res["model"], "macro_f1": lr_res["macro_f1"]},
            {"model": rf_res["model"], "macro_f1": rf_res["macro_f1"]},
        ]
    )
    metrics.to_csv("results/metrics.csv", index=False)

    print("\n=== Macro F1 on test (2018 & 2022) ===")
    print(metrics.to_string(index=False))
    print("\nSaved: results/metrics.csv and confusion matrices in results/plots/")
    print("\nDone.")


if __name__ == "__main__":
    main()
