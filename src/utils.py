from __future__ import annotations
import re
import pandas as pd

_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

def year_from_tournament_id(tournament_id: str) -> int | None:
    if tournament_id is None:
        return None
    m = _YEAR_RE.search(str(tournament_id))
    return int(m.group(1)) if m else None

def add_tournament_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["tournament_id"].map(year_from_tournament_id)
    return df
