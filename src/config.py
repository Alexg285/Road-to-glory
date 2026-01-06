LABELS = ["Group", "R16", "QF", "SF", "Final", "Winner"]
LABEL_IDS = [0, 1, 2, 3, 4, 5]

FEATURES_BASELINE = [
    "wins", "draws", "losses", "points","goal_diff",
    "avg_goals_for", "avg_goals_against",
    "win_ratio"
]

FEATURES_ENRICHED = FEATURES_BASELINE + ["strength_pre", "is_host", "avg_team_age"]

REQUIRED_COLUMNS = ["split", "y_ord"] + FEATURES_ENRICHED