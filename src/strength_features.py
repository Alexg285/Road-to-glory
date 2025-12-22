from __future__ import annotations
import pandas as pd

def add_pre_tournament_strength(
    df: pd.DataFrame,
    fifa_rankings: pd.DataFrame,
    soccer_rankings: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds unified pre-tournament strength.
    FIFA: uses last ranking strictly before tournament start date.
    Soccer: uses yearly rating for tournament year.
    strength_pre: higher = stronger.
    """

    out = []

    fifa = fifa_rankings.copy()
    fifa["ranking_date"] = pd.to_datetime(fifa["ranking_date"], errors="coerce")

    soccer = soccer_rankings.copy()
    soccer["year"] = pd.to_numeric(soccer["year"], errors="coerce")

    for (year, tid), g in df.groupby(["year", "tournament_id"]):
        g = g.copy()
        start_date = g["match_date"].min()

        # FIFA pre-tournament rank
        r = fifa[fifa["ranking_date"] < start_date]
        if not r.empty:
            last_r = (
                r.sort_values("ranking_date")
                 .groupby("team")
                 .last()
                 .reset_index()
                 .rename(columns={"rank": "fifa_rank"})
            )
            g = g.merge(last_r[["team", "fifa_rank"]], on="team", how="left")
        else:
            g["fifa_rank"] = pd.NA

        # Soccer yearly rating
        s_year = soccer[soccer["year"] == year].copy().rename(columns={"rating": "soccer_rating"})
        g = g.merge(s_year[["team", "soccer_rating"]], on="team", how="left")

        g["fifa_rank"] = pd.to_numeric(g["fifa_rank"], errors="coerce")
        g["soccer_rating"] = pd.to_numeric(g["soccer_rating"], errors="coerce")

        # Unified strength
        g["strength_pre"] = pd.NA
        g["strength_source"] = pd.NA

        fifa_ok = g["fifa_rank"].notna()
        g.loc[fifa_ok, "strength_pre"] = 1.0 / g.loc[fifa_ok, "fifa_rank"]
        g.loc[fifa_ok, "strength_source"] = "fifa"

        soccer_ok = (~fifa_ok) & g["soccer_rating"].notna()
        g.loc[soccer_ok, "strength_pre"] = g.loc[soccer_ok, "soccer_rating"]
        g.loc[soccer_ok, "strength_source"] = "soccer"

        out.append(g)

    return pd.concat(out, ignore_index=True)
