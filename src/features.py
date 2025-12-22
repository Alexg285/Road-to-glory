from __future__ import annotations
import pandas as pd
from src.utils import add_tournament_year

def build_group_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    # Your dataset columns
    stage_col = "stage_name"
    group_flag_col = "group_stage"
    home_team_col = "home_team_name"
    away_team_col = "away_team_name"
    home_goals_col = "home_team_score"
    away_goals_col = "away_team_score"
    match_date_col = "match_date"

    df[match_date_col] = pd.to_datetime(df[match_date_col], errors="coerce")
    df = add_tournament_year(df)

    # ✅ Robust group-stage selection
    if group_flag_col in df.columns:
        df = df[df[group_flag_col] == 1]
    else:
        # fallback if group flag missing
        df = df[df[stage_col].astype(str).str.contains("group", case=False, na=False)]

    rows = []
    for (year, tid), g in df.groupby(["year", "tournament_id"]):
        teams = pd.unique(pd.concat([g[home_team_col], g[away_team_col]]))

        for team in teams:
            home = g[g[home_team_col] == team]
            away = g[g[away_team_col] == team]

            games = len(home) + len(away)

            gf = home[home_goals_col].sum() + away[away_goals_col].sum()
            ga = home[away_goals_col].sum() + away[home_goals_col].sum()

            wins = ((home[home_goals_col] > home[away_goals_col]).sum()
                    + (away[away_goals_col] > away[home_goals_col]).sum())
            draws = ((home[home_goals_col] == home[away_goals_col]).sum()
                     + (away[away_goals_col] == away[home_goals_col]).sum())
            losses = games - wins - draws
            points = wins * 3 + draws

            rows.append({
                "year": int(year) if year is not None else year,
                "tournament_id": tid,
                "team": team,
                "games": int(games),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "points": int(points),
                "gf": float(gf),
                "ga": float(ga),
                "gd": float(gf - ga),
                "gf_per_game": float(gf / games) if games else 0.0,
                "ga_per_game": float(ga / games) if games else 0.0,
            })

    return pd.DataFrame(rows)

HOSTS = {
    1970: ["Mexico"],
    1974: ["Germany FR"],
    1978: ["Argentina"],
    1982: ["Spain"],
    1986: ["Mexico"],
    1990: ["Italy"],
    1994: ["USA"],
    1998: ["France"],
    2002: ["Korea Republic", "Japan"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
}

def add_is_host(df, team_col="team", year_col="year"):
    hosts_series = df[year_col].map(lambda y: HOSTS.get(int(y), []))
    df["is_host"] = [
        1 if team in host_list else 0
        for team, host_list in zip(df[team_col], hosts_series)
    ]
    return df


def add_wc_experience_before(df, team_col="team", year_col="year"):
    df = df.sort_values([team_col, year_col]).copy()
    df["wc_experience_before"] = (
        df.groupby(team_col).cumcount()
    )
    return df
