from __future__ import annotations
import pandas as pd
from src.utils import add_tournament_year

def build_progression_labels(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    stage_col = "stage_name"
    home_team_col = "home_team_name"
    away_team_col = "away_team_name"
    home_goals_col = "home_team_score"
    away_goals_col = "away_team_score"
    match_date_col = "match_date"

    df[match_date_col] = pd.to_datetime(df[match_date_col], errors="coerce")
    df = add_tournament_year(df)

    # Map common stage names to ordinal labels (adjustable)
    def stage_to_label(s: str) -> int | None:
        s = str(s).lower()
        if "group" in s or "first round" in s:
            return 0
        if "round of 16" in s:
            return 1
        if "quarter" in s:
            return 2
        if "semi" in s:
            return 3
        if s.strip() == "final":
            return 4
        return None

    rows = []
    for (year, tid), g in df.groupby(["year", "tournament_id"]):
        teams = pd.unique(pd.concat([g[home_team_col], g[away_team_col]]))

        for team in teams:
            tm = g[(g[home_team_col] == team) | (g[away_team_col] == team)]

            labels = tm[stage_col].map(stage_to_label).dropna()
            if labels.empty:
                continue

            max_stage = int(labels.max())

            finals = tm[tm[stage_col].astype(str).str.lower().str.strip() == "final"]
            is_champion = (
                ((finals[home_team_col] == team) & (finals[home_goals_col] > finals[away_goals_col])) |
                ((finals[away_team_col] == team) & (finals[away_goals_col] > finals[home_goals_col]))
            ).any()

            rows.append({
                "year": int(year) if year is not None else year,
                "tournament_id": tid,
                "team": team,
                "stage_label": 5 if bool(is_champion) else max_stage
            })

    return pd.DataFrame(rows)
