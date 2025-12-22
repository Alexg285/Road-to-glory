from __future__ import annotations
import pandas as pd

def add_pre_tournament_fifa_rank(df: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Add FIFA rank for each team, using the last ranking strictly before tournament start date.
    df must have: year, tournament_id, team, match_date
    rankings must have: ranking_date, team, rank
    """
    out = []

    for (year, tid), g in df.groupby(["year", "tournament_id"]):
        start_date = g["match_date"].min()

        r = rankings[rankings["ranking_date"] < start_date]
        if r.empty:
            # No FIFA rankings available before this tournament (e.g., pre-1993)
            g = g.copy()
            g["rank"] = pd.NA
            out.append(g)
            continue

        last_r = (
            r.sort_values("ranking_date")
             .groupby("team")
             .last()
             .reset_index()
        )

        merged = g.merge(last_r[["team", "rank"]], on="team", how="left")
        out.append(merged)

    return pd.concat(out, ignore_index=True)
