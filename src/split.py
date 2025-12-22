from __future__ import annotations
import pandas as pd

def temporal_split(df: pd.DataFrame):
    train = df[df["year"].between(1970, 2014)].copy()
    test = df[df["year"].isin([2018, 2022])].copy()
    return train, test
