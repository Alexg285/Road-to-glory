from __future__ import annotations
import pandas as pd
from src.team_names import TEAM_RENAME


def add_team_age_features(
    df: pd.DataFrame,
    players: pd.DataFrame,
    squads: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds avg_team_age to df using players.csv, squads.csv and matches.csv
    """

    # ---------- Normalize column names ----------
    for d in (df, players, squads, matches):
        d.columns = (
            d.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

    # ---------- Keep only useful columns ----------
    players = players[["player_id", "birth_date"]].copy()
    squads = squads[["tournament_id", "team_name", "player_id"]].copy()
    matches = matches[["tournament_id", "match_date"]].copy()

    # ---------- Parse dates ----------
    players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")

    # ---------- Tournament start date ----------
    tournament_start = (
        matches
        .groupby("tournament_id", as_index=False)["match_date"]
        .min()
        .rename(columns={"match_date": "tournament_start"})
    )

    # ---------- Merge squads → players ----------
    sp = squads.merge(players, on="player_id", how="left")

    # ---------- Merge tournament start ----------
    sp = sp.merge(tournament_start, on="tournament_id", how="left")

    # ---------- Compute age ----------
    sp["age_at_tournament"] = (
        (sp["tournament_start"] - sp["birth_date"])
        .dt.days / 365.25
    )

    # ---------- Average per team & tournament ----------
    team_age = (
        sp.groupby(["tournament_id", "team_name"], as_index=False)["age_at_tournament"]
        .mean()
        .rename(columns={"age_at_tournament": "avg_team_age"})
    )

    # ---------- Normalize team names ----------
    team_age["team_name"] = team_age["team_name"].replace(TEAM_RENAME)
    df["team"] = df["team"].replace(TEAM_RENAME)

    # ---------- Merge into df ----------
    out = df.merge(
        team_age,
        left_on=["tournament_id", "team"],
        right_on=["tournament_id", "team_name"],
        how="left",
    )

    out = out.drop(columns=["team_name"])
    return out