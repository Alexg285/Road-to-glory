"""Data loading for the Road to Glory project."""
from __future__ import annotations
import pandas as pd

def load_matches(path: str = "data/raw/matches.csv") -> pd.DataFrame:
    return pd.read_csv(path, sep=";", engine="python", encoding="utf-8")

def load_fifa_rankings(path: str = "data/raw/Ranking fifa 1993-2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python", encoding="latin1")
    df = df.rename(columns={
        "country_full": "team",
        "rank_date": "ranking_date"
    })
    df["ranking_date"] = pd.to_datetime(df["ranking_date"], errors="coerce", format="%d.%m.%y")
    if df["ranking_date"].isna().mean() > 0.2:
        df["ranking_date"] = pd.to_datetime(df["ranking_date"], errors="coerce")
    return df[["team", "rank", "ranking_date"]].copy()

def load_soccer_rankings(path: str = "data/raw/Ranking soccer 1901-2023.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python", encoding="latin1")
    return df[["year", "team", "rank", "rating"]].copy()
